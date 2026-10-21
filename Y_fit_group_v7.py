
import ast
import re
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ==========================================================
# SETTINGS
# ==========================================================

FREQ = 10000.0
B_PER_PF = 2*np.pi*FREQ*1e-3

G_UNIT = "S"
G_FACTOR = {"S": 1e9, "uS": 1e3, "nS": 1.0}

C_SCALE = 0.5       # pF
G_SCALE = 1.0       # nS

N_FOLDS = 3
TEST_SIZE = 0.25
SEED = 42

PIN_G_REG = 3.0
PIN_B_REG = 30.0
SHARED_Z_REG = 2000.0

# pF coefficient priors.
# SMU-dependent parameters need stronger shrinkage.
LOCAL_SIGMA = {
    "Local-Global": 0.5,
    "Local-SMU": 0.35,
    "Local-Interaction": 0.35
}

SMUS = [
    range(1,7), range(7,15), range(15,23),
    range(23,31), range(31,39), range(39,45)
]

SMU_OF = {
    p: g for g, group in enumerate(SMUS)
    for p in group
}

# ==========================================================
# DATA PREPARATION
# ==========================================================

def parse_pins(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            x = re.findall(r"\d+", x)

    if isinstance(x, (int, np.integer)):
        x = [x]

    out = tuple(sorted(int(p) for p in x))

    if not out or len(set(out)) != len(out):
        raise ValueError(f"Invalid pin selection: {x}")

    if any(p < 1 or p > 44 for p in out):
        raise ValueError("Pin outside 1..44")

    return out


data = df.copy().reset_index(drop=True)

HIGH = [parse_pins(x) for x in data["high_pins"]]
LOW = [parse_pins(x) for x in data["low_pins"]]

N = len(data)
H = np.zeros((N,44), dtype=float)
L = np.zeros((N,44), dtype=float)

config = []
signature = []

for k, (hh,ll) in enumerate(zip(HIGH,LOW)):
    if set(hh) & set(ll):
        raise ValueError(f"H/L overlap at row {k}")

    H[k,np.array(hh)-1] = 1
    L[k,np.array(ll)-1] = 1

    config.append(str((hh,ll)))

    signature.append(str(tuple(
        [sum(p in hh for p in group) for group in SMUS]
        + [sum(p in ll for p in group) for group in SMUS]
    )))

CONFIG = np.asarray(config)
SIGNATURE = np.asarray(signature)

C_TRUE = (
    pd.to_numeric(data["C"]).to_numpy(dtype=float)*1e12
)
G_TRUE = (
    pd.to_numeric(data["G"]).to_numpy(dtype=float)
    *G_FACTOR[G_UNIT]
)

if not (np.isfinite(C_TRUE).all()
        and np.isfinite(G_TRUE).all()):
    raise ValueError("Nonfinite measured values")

# ==========================================================
# BASE MODEL: SHARED PATH + INDIVIDUAL
#
# Parameters:
#   0:2   common G,B [nS]
#   2:4   shared Z real,imag [MOhm]
#   4:48  individual dG [nS]
#   48:92 individual dB [nS]
# ==========================================================

def unpack_base(t):
    dg = t[4:48] - np.mean(t[4:48])
    db = t[48:92] - np.mean(t[48:92])

    yp = (t[0]+dg) + 1j*(t[1]+db)
    zs = complex(t[2],t[3])
    return yp,zs,dg,db


def predict_base(idx,t):
    yp,zs,_,_ = unpack_base(t)

    sh = H[idx] @ yp
    sl = L[idx] @ yp

    yh = sh/(1+1e-3*zs*sh)
    yl = sl/(1+1e-3*zs*sl)

    return yh*yl/(yh+yl)


def predict_safe(idx,t):
    with np.errstate(all="ignore"):
        y = predict_base(idx,t)

    if not np.isfinite(y).all():
        return np.full(len(idx),1e12+1e12j)

    return y


def measurement_residual(idx,y):
    return np.r_[
        (y.imag/B_PER_PF-C_TRUE[idx])/C_SCALE,
        (y.real-G_TRUE[idx])/G_SCALE
    ]


def regularize_base(t):
    _,zs,dg,db = unpack_base(t)

    return np.r_[
        dg/PIN_G_REG,
        db/PIN_B_REG,
        zs.real/SHARED_Z_REG,
        zs.imag/SHARED_Z_REG,
        t[4:48].mean(),
        t[48:92].mean()
    ]


BASE_LB = np.r_[
    [-100,-1000,-100,-100],
    np.full(44,-100),
    np.full(44,-300)
]

BASE_UB = np.r_[
    [100,1000,100,100],
    np.full(44,100),
    np.full(44,300)
]


def fit_base(train):
    # First fit common pin admittance.
    def common_res(x):
        t = np.zeros(92)
        t[:4] = x
        return np.r_[
            measurement_residual(
                train,predict_safe(train,t)
            ),
            x[2:]/SHARED_Z_REG
        ]

    init = [0.1,0.5*B_PER_PF,1.0,0.0]

    common = least_squares(
        common_res,init,
        bounds=(BASE_LB[:4],BASE_UB[:4]),
        max_nfev=400,
        x_scale="jac"
    )

    # Fit all pin parameters and shared path jointly.
    t0 = np.zeros(92)
    t0[:4] = common.x

    def objective(t):
        return np.r_[
            measurement_residual(
                train,predict_safe(train,t)
            ),
            regularize_base(t)
        ]

    opt = least_squares(
        objective,t0,
        bounds=(BASE_LB,BASE_UB),
        max_nfev=400,
        x_scale="jac"
    )

    return opt.x


# ==========================================================
# LOCAL PAIR FEATURES
#
# By SMU:
#    0 Even adjacent
#    1 Odd adjacent
#    2 Distance 2
#
# Interaction:
#    High pair distance 2,
#    with Low adjacent to either High pin.
#
# Features are computed only from High/Low pins.
# ==========================================================

ZONE_MASK = np.zeros((6,3,44,44),dtype=float)

for a in range(44):
    for b in range(44):
        if a == b:
            continue

        pa,pb = a+1,b+1

        if SMU_OF[pa] != SMU_OF[pb]:
            continue

        g = SMU_OF[pa]
        d = abs(pa-pb)

        if d == 1:
            z = 0 if max(pa,pb)%2 == 0 else 1
        elif d == 2:
            z = 2
        else:
            continue

        ZONE_MASK[g,z,a,b] = 1


def build_local_features():
    # Per-SMU zone counts (N x 18).
    base = np.column_stack([
        np.sum((H @ ZONE_MASK[g,z])*L,axis=1)
        for g in range(6)
        for z in range(3)
    ])

    # An additional same-SMU interaction for
    # High pins separated by 2 and adjacent Low.
    #
    # Counts one activation per High pair when
    # at least one selected Low pin is adjacent
    # to either High pin.
    interaction = np.zeros((N,6))

    for k,(hh,ll) in enumerate(zip(HIGH,LOW)):
        low_set = set(ll)

        for i,a in enumerate(hh):
            for b in hh[i+1:]:
                if SMU_OF[a] != SMU_OF[b]:
                    continue
                if abs(a-b) != 2:
                    continue

                adjacent = (
                    (a-1 in low_set) or
                    (a+1 in low_set) or
                    (b-1 in low_set) or
                    (b+1 in low_set)
                )

                if adjacent:
                    interaction[k,SMU_OF[a]] += 1

    return base,interaction


SMU_ZONE_X, INTERACTION_X = build_local_features()

FEATURES = {
    "Local-Global":
        SMU_ZONE_X.reshape(N,6,3).sum(axis=1),

    "Local-SMU":
        SMU_ZONE_X,

    "Local-Interaction":
        np.column_stack([SMU_ZONE_X,INTERACTION_X])
}

# ==========================================================
# LOCAL CORRECTION FIT
#
# Stage 1: Base trained on training data.
# Stage 2: Fit local correction to training residuals.
#
# Holdout predictions never enter fitting.
# ==========================================================

def fit_local(tr,t_base,model):
    x = FEATURES[model][tr]
    y_base = predict_safe(tr,t_base)
    target = C_TRUE[tr]-y_base.imag/B_PER_PF

    sigma = LOCAL_SIGMA[model]

    # Ridge, centered at zero.
    x_aug = np.vstack([
        x,
        np.eye(x.shape[1])*(C_SCALE/sigma)
    ])
    y_aug = np.r_[
        target,np.zeros(x.shape[1])
    ]

    alpha = np.linalg.lstsq(
        x_aug,y_aug,rcond=None
    )[0]

    return alpha


def predict_local(idx,t_base,model,alpha):
    y = predict_safe(idx,t_base).copy()

    delta_c = FEATURES[model][idx]@alpha

    return y+1j*B_PER_PF*delta_c


def calc_metrics(idx,y):
    err = (y.imag/B_PER_PF-C_TRUE[idx])*1000

    return {
        "RMSE_fF": float(np.sqrt(np.mean(err**2))),
        "MAE_fF": float(np.mean(np.abs(err))),
        "P95_fF": float(np.percentile(np.abs(err),95)),
        "Bias_fF": float(np.mean(err))
    }


# ==========================================================
# GROUPED CV
# ==========================================================

rows = []
coef_rows = []
prediction_rows = []

for split_name,groups in [
    ("RandomConfig",CONFIG),
    ("UnseenSignature",SIGNATURE)
]:
    cv = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold,(tr,te) in enumerate(
        cv.split(H,groups=groups),1
    ):
        t_base = fit_base(tr)

        predictions = {
            "Base":predict_safe(te,t_base)
        }

        for model in FEATURES:
            alpha = fit_local(tr,t_base,model)
            predictions[model] = predict_local(
                te,t_base,model,alpha
            )

            coef_rows.append({
                "Split":split_name,
                "Fold":fold,
                "Model":model,
                "Coeff":alpha.copy(),
                "CoeffAbsMean_pF":np.mean(np.abs(alpha)),
                "CoeffAbsMax_pF":np.max(np.abs(alpha))
            })

        for model,y in predictions.items():
            rows.append({
                "Split":split_name,
                "Fold":fold,
                "Model":model,
                **calc_metrics(te,y)
            })

            for row_id,yi in zip(te,y):
                prediction_rows.append({
                    "Split":split_name,
                    "Fold":fold,
                    "Row":int(row_id),
                    "Model":model,
                    "High":str(HIGH[row_id]),
                    "Low":str(LOW[row_id]),
                    "Actual_pF":C_TRUE[row_id],
                    "Pred_pF":yi.imag/B_PER_PF,
                    "Error_fF":(
                        yi.imag/B_PER_PF-C_TRUE[row_id]
                    )*1000
                })

        print(f"\n{split_name} / Fold {fold}")
        for name,y in predictions.items():
            print(
                f"{name:20} "
                f"RMSE={calc_metrics(te,y)['RMSE_fF']:.2f} fF"
            )


# ==========================================================
# RESULTS
# ==========================================================

scores = pd.DataFrame(rows)
coeffs = pd.DataFrame(coef_rows)
pred_df = pd.DataFrame(prediction_rows)

summary = (
    scores.groupby(["Split","Model"])
    .agg(
        RMSE_Mean=("RMSE_fF","mean"),
        RMSE_Std=("RMSE_fF","std"),
        MAE_Mean=("MAE_fF","mean"),
        P95_Mean=("P95_fF","mean"),
        Bias_Mean=("Bias_fF","mean")
    )
    .reset_index()
)

print("\n=== LOCAL COUPLING COMPARISON ===")
print(
    summary.sort_values(
        ["Split","RMSE_Mean"]
    ).round(3).to_string(index=False)
)

wide = scores.pivot(
    index=["Split","Fold"],
    columns="Model",
    values="RMSE_fF"
).reset_index()

for model in FEATURES:
    wide[model+"_Benefit_fF"] = (
        wide["Base"]-wide[model]
    )

print("\n=== IMPROVEMENT OVER BASE ===")
print(
    wide.round(3).to_string(index=False)
)

# ==========================================================
# TARGET CONFIGURATION DIAGNOSTICS
#
# Look at High = {7,9} in holdout predictions.
# No additional measurements required.
# ==========================================================

target_rows = np.array([
    k for k,hh in enumerate(HIGH)
    if set(hh) == {7,9}
])

target = pred_df[
    pred_df["Row"].isin(target_rows)
].copy()

print("\n=== HIGH 7,9 HOLDOUT ERRORS ===")

if target.empty:
    print(
        "No High={7,9} rows in these holdout folds."
    )
else:
    print(
        target.sort_values(
            ["Row","Split","Fold","Model"]
        )[
            [
                "Split","Fold","High","Low",
                "Model","Actual_pF","Pred_pF",
                "Error_fF"
            ]
        ].round(3).to_string(index=False)
    )

# ==========================================================
# FULL-DATA FIT AND COEFFICIENT REPORT
#
# Diagnostic only; CV is the performance measure.
# ==========================================================

all_idx = np.arange(N)
t_full = fit_base(all_idx)

final_coeff = {
    model:fit_local(all_idx,t_full,model)
    for model in FEATURES
}

smu_coeff = final_coeff["Local-SMU"].reshape(6,3)

coeff_report = pd.DataFrame(
    smu_coeff,
    columns=["Even_pF","Odd_pF","Dist2_pF"]
)

coeff_report.insert(
    0,"SMU",np.arange(1,7)
)

print("\n=== SMU-SPECIFIC PAIR COEFFICIENTS ===")
print(coeff_report.round(5).to_string(index=False))

int_coeff = final_coeff["Local-Interaction"]

print("\n=== INTERACTION COEFFICIENTS ===")
for g,a in enumerate(int_coeff[18:],1):
    print(f"SMU{g}: {a:+.6f} pF")

# Full-data comparison specifically for High={7,9}
print("\n=== HIGH 7,9 FULL DATA COMPARISON ===")

if len(target_rows):
    output = pd.DataFrame({
        "High":[str(HIGH[k]) for k in target_rows],
        "Low":[str(LOW[k]) for k in target_rows],
        "Actual_pF":C_TRUE[target_rows]
    })

    output["Base_pF"] = (
        predict_safe(target_rows,t_full).imag/B_PER_PF
    )

    for name,alpha in final_coeff.items():
        output[name+"_pF"] = (
            predict_local(target_rows,t_full,name,alpha)
            .imag/B_PER_PF
        )

    print(
        output.round(4).to_string(index=False)
    )

scores.to_csv("local_coupling_scores.csv",index=False)
pred_df.to_csv("local_coupling_predictions.csv",index=False)
coeff_report.to_csv("local_smu_coefficients.csv",index=False)
