
import ast
import re
import warnings
import numpy as np
import pandas as pd
import joblib

from itertools import product
from scipy.optimize import least_squares, minimize
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

# ============================================================
# 0. SETTINGS - SI ONLY
# ============================================================

FREQ = 10000.0
W = 2 * np.pi * FREQ

COL_CK = "C_keysight"
COL_GK = "G_keysight"
COL_CV = "C_v93k"
COL_GV = "G_v93k"
COL_H = "high_pins"
COL_L = "low_pins"

# Error normalization, SI units
C_SCALE = 5e-13
G_SCALE = 1e-9

# Ridge regularization in dimensionless regression space
RIDGE_VALUES = [0.01, 0.1, 1.0, 10.0]

# Initial fast screening
SCREEN_TEST_SIZE = 0.25
SCREEN_SEED = 42

# Detailed validation for finalists
FINALISTS = 12
FINAL_FOLDS = 5
FINAL_TEST_SIZE = 0.25

# Avoid tiny groups dominating model selection
MIN_TRAIN_ROWS = 10

# Optimization scaling for Zs [Ohm]
Z_REF = 1e6
Z_MAX = 1e8

# SharedPath parameter refinement
REFINE_MAXITER = 60

# Choose final winner by C RMSE.
# Change to "Combined" to weigh G too.
RANK_METRIC = "C_RMSE_F"

# Split by actual pin configuration.
# Alternative: "Signature"
GROUP_MODE = "Config"

OUTDIR = Path("dut_autofit_results")
OUTDIR.mkdir(exist_ok=True)

SMUS = [
    list(range(1, 7)),
    list(range(7, 15)),
    list(range(15, 23)),
    list(range(23, 31)),
    list(range(31, 39)),
    list(range(39, 45))
]

SMU = np.zeros(45, dtype=int)

for g, pins in enumerate(SMUS):
    for p in pins:
        SMU[p] = g


# ============================================================
# 1. INPUT
# ============================================================

def parse_pins(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = re.findall(r"\d+", v)

    if isinstance(v, (int, np.integer)):
        v = [v]

    v = tuple(sorted(int(p) for p in v))

    if not v or len(set(v)) != len(v):
        raise ValueError(f"Invalid pins: {v}")

    if any(p < 1 or p > 44 for p in v):
        raise ValueError(f"Pin outside 1..44: {v}")

    return v


d = df.copy().reset_index(drop=True)

required = [
    COL_CK, COL_GK, COL_CV, COL_GV,
    COL_H, COL_L
]

missing = set(required) - set(d.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

HIGH = [parse_pins(x) for x in d[COL_H]]
LOW = [parse_pins(x) for x in d[COL_L]]

N = len(d)

H = np.zeros((N, 44), dtype=float)
L = np.zeros((N, 44), dtype=float)

CONFIG = []
SIGNATURE = []

for k, (hh, ll) in enumerate(zip(HIGH, LOW)):
    if set(hh) & set(ll):
        raise ValueError(f"High/Low overlap at row {k}")

    H[k, np.array(hh)-1] = 1
    L[k, np.array(ll)-1] = 1

    CONFIG.append(str((hh, ll)))

    hc = [sum(p in hh for p in s) for s in SMUS]
    lc = [sum(p in ll for p in s) for s in SMUS]

    SIGNATURE.append(str(tuple(hc + lc)))

CONFIG = np.asarray(CONFIG)
SIGNATURE = np.asarray(SIGNATURE)

GROUPS = CONFIG if GROUP_MODE == "Config" else SIGNATURE

CK = d[COL_CK].to_numpy(float)
GK = d[COL_GK].to_numpy(float)

CV = d[COL_CV].to_numpy(float)
GV = d[COL_GV].to_numpy(float)

YK = GK + 1j*W*CK
YV = GV + 1j*W*CV

if not (
    np.isfinite(YK).all()
    and np.isfinite(YV).all()
):
    raise ValueError("Invalid measurement values")

print("Rows:", N)
print("Configurations:", len(set(CONFIG)))
print("Signatures:", len(set(SIGNATURE)))
print("C range [F]:", CK.min(), CK.max())
print("G range [S]:", GK.min(), GK.max())


# ============================================================
# 2. PIN AGGREGATION FEATURES
#
# These encode selection-dependent complex correction
# coefficients, fitted in SI admittance.
#
# PinSum     : additive pin contribution
# PinMean    : averaged side contributions
# PinSeries  : H/L series-equivalent participation weights
# PinShared  : count-saturated side participation
#
# PinShared is an empirical SharedPath-shaped feature.
# Explicit complex Zs is fitted separately below.
# ============================================================

nh = H.sum(axis=1, keepdims=True)
nl = L.sum(axis=1, keepdims=True)

def safe_ratio(a, b):
    return np.divide(
        a, b,
        out=np.zeros_like(a, dtype=float),
        where=np.abs(b) > 1e-15
    )


PIN_FEATURES = {
    "PinSum": np.column_stack([H, L]),

    "PinMean": np.column_stack([
        H / nh, L / nl
    ]),

    "PinSeries": np.column_stack([
        H * nl / (nh + nl),
        L * nh / (nh + nl)
    ]),

    "PinShared": np.column_stack([
        H / (1 + nh),
        L / (1 + nl)
    ])
}

# Alternative lower-dimensional grouping.
SMU_H = np.column_stack([
    H[:, np.asarray(s)-1].sum(axis=1)
    for s in SMUS
])

SMU_L = np.column_stack([
    L[:, np.asarray(s)-1].sum(axis=1)
    for s in SMUS
])


# ============================================================
# 3. PAIR ZONES
#
# 0 even adjacent
# 1 odd adjacent
# 2 distance 2
# 3 same-SMU far
# 4 cross-SMU
#
# All relevant zones, including cross-SMU,
# are included in the fit.
# ============================================================

def zone(h, l):
    if SMU[h] != SMU[l]:
        return 4

    dist = abs(h-l)

    if dist == 1:
        return 0 if max(h,l)%2 == 0 else 1

    if dist == 2:
        return 2

    return 3


PAIR_MODES = [
    "Sum", "Mean", "MaxPerPin",
    "Sqrt", "PartialSum"
]

PAIR_FEATURES = {
    mode: np.zeros((N, 5), dtype=float)
    for mode in PAIR_MODES
}

PAIR_SMU_FEATURES = {
    mode: np.zeros((N, 30), dtype=float)
    for mode in PAIR_MODES
}


def aggregate_binary_zone(zmat, mode):
    # zmat: 0/1 matrix for one zone.
    s = zmat.sum()

    if s == 0:
        return 0.0

    mh = np.count_nonzero(zmat.sum(axis=1))
    ml = np.count_nonzero(zmat.sum(axis=0))

    m = 0.5*(mh + ml)

    if mode == "Sum":
        return float(s)

    if mode == "Mean":
        return float(s/zmat.size)

    if mode == "MaxPerPin":
        return float(m)

    if mode == "Sqrt":
        return float(np.sqrt(s*m))

    if mode == "PartialSum":
        return float(m + 0.25*(s-m))

    raise ValueError(mode)


for k, (hh, ll) in enumerate(zip(HIGH, LOW)):
    zz = np.array([
        [zone(h,l) for l in ll]
        for h in hh
    ])

    for mode in PAIR_MODES:
        for z in range(5):
            mask = (zz == z).astype(float)

            PAIR_FEATURES[mode][k,z] = (
                aggregate_binary_zone(mask, mode)
            )

            for g in range(6):
                # Group by High-side SMU
                hi = [
                    i for i,h in enumerate(hh)
                    if SMU[h] == g
                ]

                if not hi:
                    continue

                sub = mask[hi,:]

                PAIR_SMU_FEATURES[mode][k,5*g+z] = (
                    aggregate_binary_zone(sub, mode)
                )


# ============================================================
# 4. CANDIDATE FEATURE STRUCTURES
# ============================================================

FEATURE_BANK = {}
FEATURE_INFO = {}

for pin_mode, pair_mode, structure in product(
    PIN_FEATURES.keys(),
    PAIR_MODES,
    ["Global", "Pin", "SMU", "Pin+SMU", "Pair", "All"]
):

    blocks = []

    if structure in ("Pin", "Pin+SMU", "All"):
        blocks.append(PIN_FEATURES[pin_mode])

    if structure in ("SMU", "Pin+SMU", "All"):
        blocks.extend([SMU_H, SMU_L])

    if structure in ("Pair", "All"):
        blocks.append(PAIR_FEATURES[pair_mode])

    if structure == "All":
        blocks.append(PAIR_SMU_FEATURES[pair_mode])

    if structure == "Global":
        X = np.zeros((N, 0))
    else:
        X = np.column_stack(blocks)

    key = (pin_mode, pair_mode, structure)

    FEATURE_BANK[key] = X

    FEATURE_INFO[key] = {
        "Pin": pin_mode,
        "Pair": pair_mode,
        "Structure": structure
    }

# Remove duplicate feature matrices, e.g. Global
unique = {}
seen = set()

for key, x in FEATURE_BANK.items():
    fingerprint = (
        x.shape,
        np.round(x, 10).tobytes()
    )

    if fingerprint not in seen:
        seen.add(fingerprint)
        unique[key] = x

FEATURE_BANK = unique

print("Unique feature structures:", len(FEATURE_BANK))


# ============================================================
# 5. COMPLEX REGRESSION
#
# Y_corrected = Y_base + X @ beta
#
# beta is complex admittance [S].
#
# C/G are simultaneously fitted using a
# weighted real-valued least-squares system.
#
# Internal optimizer coefficients are dimensionless
# but the returned beta is always in S.
# ============================================================

def fit_linear_correction(
    x_train,
    y_target,
    y_base,
    ridge,
    affine=False
):
    # Additional complex gain on Y_base.
    if affine:
        x = np.column_stack([
            x_train.astype(complex),
            y_base
        ])
    else:
        x = x_train.astype(complex)

    # Complex intercept.
    x = np.column_stack([
        np.ones(len(x), dtype=complex),
        x
    ])

    # Scale real-valued configuration features
    # for well-conditioned regression.
    scales = np.sqrt(np.mean(np.abs(x)**2, axis=0))
    scales = np.maximum(scales, 1e-20)

    # For Y_base affine column, choose its natural
    # data magnitude as its numerical scale.
    xn = x / scales

    # Target and baseline in SI.
    residual_y = y_target - y_base

    # Fit beta = beta_re + j beta_im.
    # Real and imag responses are weighted
    # in their native S units.
    ar, ai = xn.real, xn.imag

    a_top = np.column_stack([
        ar/G_SCALE,
        -ai/G_SCALE
    ])

    a_bottom = np.column_stack([
        ai/(W*C_SCALE),
        ar/(W*C_SCALE)
    ])

    A = np.vstack([a_top, a_bottom])

    b = np.r_[
        residual_y.real/G_SCALE,
        residual_y.imag/(W*C_SCALE)
    ]

    ncoef = x.shape[1]

    # No ridge on intercept; other columns penalized.
    reg = np.sqrt(ridge) * np.eye(2*ncoef)
    reg[0,0] = 0
    reg[ncoef,ncoef] = 0

    A_aug = np.vstack([A, reg])
    b_aug = np.r_[b, np.zeros(2*ncoef)]

    theta = np.linalg.lstsq(
        A_aug, b_aug, rcond=None
    )[0]

    beta_normalized = (
        theta[:ncoef] + 1j*theta[ncoef:]
    )

    beta = beta_normalized / scales

    return beta


def predict_linear(x, y_base, beta, affine=False):
    if affine:
        features = np.column_stack([
            np.ones(len(x), dtype=complex),
            x.astype(complex),
            y_base
        ])
    else:
        features = np.column_stack([
            np.ones(len(x), dtype=complex),
            x.astype(complex)
        ])

    return y_base + features @ beta


# ============================================================
# 6. SHARED PATH / SERIES CORRECTION
#
# z [Ohm], y [S]
#
# Base:
#   Direct = yv
#   Series = yv/(1-z*yv)
#
# Inference:
#   post: transform first, then feature correction
#   pre : feature correction first, then transform
# ============================================================

def series_transform(y, z):
    den = 1.0 - z*y

    with np.errstate(
        divide="ignore",
        invalid="ignore"
    ):
        return y/den


def base_transform(y, z, transform):
    if transform == "Direct":
        return y

    if transform == "Series":
        return series_transform(y, z)

    raise ValueError(transform)


def correction_from_beta(x, base, beta, affine):
    return predict_linear(
        x, base, beta, affine
    ) - base


def predict_model(x, yv, model):
    z = model["z"]
    beta = model["beta"]
    affine = model["affine"]
    transform = model["transform"]
    order = model["order"]

    if order == "Post":
        ybase = base_transform(yv,z,transform)

        return predict_linear(
            x,ybase,beta,affine
        )

    # Pre correction:
    # Feature correction is calculated from raw Yv,
    # then transformed by the series path.
    delta = correction_from_beta(
        x,yv,beta,affine
    )

    return base_transform(
        yv + delta,z,transform
    )


# ============================================================
# 7. FIT A MODEL
#
# The fast screening stage uses an initial
# discrete set of shared path impedances.
#
# The final stage refines Z continuously.
# ============================================================

Z_CANDIDATES = [
    0j,
    complex(1e5,0),
    complex(-1e5,0),
    complex(0,1e5),
    complex(0,-1e5),
    complex(1e6,0),
    complex(0,1e6)
]


def objective_score(ytrue, ypred):
    ec = (
        (ypred.imag-ytrue.imag)
        / (W*C_SCALE)
    )

    eg = (
        (ypred.real-ytrue.real)
        / G_SCALE
    )

    if not (
        np.isfinite(ec).all()
        and np.isfinite(eg).all()
    ):
        return 1e30

    return np.mean(ec**2 + eg**2)


def fit_candidate(
    tr, feature_key, transform, affine, ridge,
    z_candidates=None
):
    X = FEATURE_BANK[feature_key]

    if z_candidates is None:
        z_candidates = (
            [0j]
            if transform == "Direct"
            else Z_CANDIDATES
        )

    best = None
    best_loss = np.inf

    for z in z_candidates:
        # Screening uses Post ordering,
        # which admits stable linear coefficient fitting.
        base = base_transform(
            YV[tr],z,transform
        )

        if not np.isfinite(base).all():
            continue

        if np.any(
            np.abs(1-z*YV[tr]) < 0.05
        ):
            continue

        beta = fit_linear_correction(
            X[tr],YK[tr],base,
            ridge,affine
        )

        pred = predict_linear(
            X[tr],base,beta,affine
        )

        loss = objective_score(YK[tr],pred)

        if loss < best_loss:
            best_loss = loss

            best = {
                "feature_key":feature_key,
                "transform":transform,
                "affine":affine,
                "order":"Post",
                "ridge":ridge,
                "z":z,
                "beta":beta
            }

    return best


# ============================================================
# 8. METRICS
# ============================================================

def metrics(actual,pred):
    ec = (pred.imag-actual.imag)/W
    eg = pred.real-actual.real

    return {
        "C_RMSE_F":float(np.sqrt(np.mean(ec**2))),
        "C_MAE_F":float(np.mean(np.abs(ec))),
        "C_P95_F":float(np.quantile(np.abs(ec),0.95)),
        "G_RMSE_S":float(np.sqrt(np.mean(eg**2))),
        "G_MAE_S":float(np.mean(np.abs(eg))),
        "Combined":float(
            np.sqrt(objective_score(actual,pred))
        )
    }


def model_id(model):
    p, pair, structure = model["feature_key"]

    return (
        f"{p}|{pair}|{structure}|"
        f"{model['transform']}|"
        f"Affine={model['affine']}|"
        f"Ridge={model['ridge']}"
    )


# ============================================================
# 9. FAST FULL SCREENING
#
# One fixed unseen-configuration split.
# All candidates evaluated on SAME holdout.
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=SCREEN_TEST_SIZE,
    random_state=SCREEN_SEED
)

tr,te = next(
    splitter.split(np.arange(N),groups=GROUPS)
)

if len(tr)<MIN_TRAIN_ROWS:
    raise ValueError("Insufficient training rows")

screen_rows = []
screen_models = []

counter = 0

for feature_key in FEATURE_BANK:
    for transform,affine,ridge in product(
        ["Direct","Series"],
        [False,True],
        RIDGE_VALUES
    ):
        model = fit_candidate(
            tr,feature_key,transform,
            affine,ridge
        )

        if model is None:
            continue

        X = FEATURE_BANK[feature_key]

        pred = predict_model(
            X[te],YV[te],model
        )

        m = metrics(YK[te],pred)

        score = {
            "ID":model_id(model),
            **FEATURE_INFO[feature_key],
            "Transform":transform,
            "Affine":affine,
            "Ridge":ridge,
            "Z_real_Ohm":model["z"].real,
            "Z_imag_Ohm":model["z"].imag,
            **m
        }

        screen_rows.append(score)
        screen_models.append(model)
        counter += 1

        if counter%50 == 0:
            print("Screened:",counter)


screen = pd.DataFrame(screen_rows)

screen["_model_idx"] = np.arange(len(screen))

screen = screen.sort_values(
    RANK_METRIC
).reset_index(drop=True)

print("\n=== FAST SCREENING TOP 20 ===")

print(
    screen.head(20)[[
        "Pin","Pair","Structure",
        "Transform","Affine","Ridge",
        "C_RMSE_F","G_RMSE_S"
    ]].to_string(
        index=False,
        float_format=lambda x:f"{x:.6e}"
    )
)


# ============================================================
# 10. SHARED Z REFINEMENT
#
# Optimize complex series Z.
# After every Z change, linear coefficients
# are refitted using TRAIN data only.
# ============================================================

def refine_model(tr,model):
    if model["transform"] == "Direct":
        return model.copy()

    X = FEATURE_BANK[model["feature_key"]]

    def fit_at_z(z):
        if np.any(np.abs(1-z*YV[tr])<0.05):
            return None,1e30

        base = series_transform(YV[tr],z)

        if not np.isfinite(base).all():
            return None,1e30

        beta = fit_linear_correction(
            X[tr],YK[tr],base,
            model["ridge"],model["affine"]
        )

        pred = predict_linear(
            X[tr],base,beta,model["affine"]
        )

        return beta,objective_score(YK[tr],pred)

    def loss(q):
        z = Z_REF*complex(q[0],q[1])
        _,score = fit_at_z(z)
        return score

    q0 = np.array([
        model["z"].real/Z_REF,
        model["z"].imag/Z_REF
    ])

    result = minimize(
        loss,q0,
        method="Powell",
        bounds=[
            (-Z_MAX/Z_REF,Z_MAX/Z_REF),
            (-Z_MAX/Z_REF,Z_MAX/Z_REF)
        ],
        options={
            "maxiter":REFINE_MAXITER,
            "xtol":1e-4,
            "ftol":1e-6
        }
    )

    z_new = Z_REF*complex(*result.x)
    beta_new,new_loss = fit_at_z(z_new)

    _,old_loss = fit_at_z(model["z"])

    if beta_new is None or new_loss>=old_loss:
        return model.copy()

    updated = model.copy()
    updated["z"] = z_new
    updated["beta"] = beta_new

    return updated


# ============================================================
# 11. FINALISTS - MATCHED REPEATED HOLDOUT
#
# Candidate structure selected from fast screening.
#
# Refit all parameters independently inside each fold.
# ============================================================

top_indices = screen.head(FINALISTS)[
    "_model_idx"
].to_numpy(int)

finalist_specs = [
    screen_models[i]
    for i in top_indices
]

cv = GroupShuffleSplit(
    n_splits=FINAL_FOLDS,
    test_size=FINAL_TEST_SIZE,
    random_state=SEED+100
)

final_rows = []

for fold,(tr,te) in enumerate(
    cv.split(np.arange(N),groups=GROUPS),1
):
    for j,spec in enumerate(finalist_specs):
        fitted = fit_candidate(
            tr,
            spec["feature_key"],
            spec["transform"],
            spec["affine"],
            spec["ridge"],
            z_candidates=(
                [0j]
                if spec["transform"]=="Direct"
                else Z_CANDIDATES+[spec["z"]]
            )
        )

        if fitted is None:
            continue

        fitted = refine_model(tr,fitted)

        X = FEATURE_BANK[fitted["feature_key"]]

        pred = predict_model(
            X[te],YV[te],fitted
        )

        m = metrics(YK[te],pred)

        final_rows.append({
            "Candidate":j,
            "Fold":fold,
            "ID":model_id(spec),
            "Z_real_Ohm":fitted["z"].real,
            "Z_imag_Ohm":fitted["z"].imag,
            **m
        })

        print(
            f"Fold {fold} Candidate {j:02d} "
            f"C={m['C_RMSE_F']:.6e} F "
            f"G={m['G_RMSE_S']:.6e} S"
        )


final_cv = pd.DataFrame(final_rows)

ranking = (
    final_cv.groupby(["Candidate","ID"])
    .agg(
        C_RMSE_mean_F=("C_RMSE_F","mean"),
        C_RMSE_std_F=("C_RMSE_F","std"),
        G_RMSE_mean_S=("G_RMSE_S","mean"),
        G_RMSE_std_S=("G_RMSE_S","std"),
        Combined_mean=("Combined","mean")
    )
    .reset_index()
)

ranking = ranking.sort_values(
    "C_RMSE_mean_F"
).reset_index(drop=True)

print("\n=== FINAL MODEL RANKING ===")

print(
    ranking.to_string(
        index=False,
        float_format=lambda x:f"{x:.6e}"
    )
)


# ============================================================
# 12. REFIT WINNER ON ALL DATA
# ============================================================

best_candidate = int(ranking.iloc[0]["Candidate"])
best_spec = finalist_specs[best_candidate]

all_idx = np.arange(N)

winner = fit_candidate(
    all_idx,
    best_spec["feature_key"],
    best_spec["transform"],
    best_spec["affine"],
    best_spec["ridge"],
    z_candidates=(
        [0j]
        if best_spec["transform"]=="Direct"
        else Z_CANDIDATES+[best_spec["z"]]
    )
)

winner = refine_model(all_idx,winner)

X_best = FEATURE_BANK[winner["feature_key"]]

Y_PRED = predict_model(
    X_best,YV,winner
)

C_PRED = Y_PRED.imag/W
G_PRED = Y_PRED.real

print("\n=== WINNING MODEL ===")
print(model_id(winner))
print("Shared Z [Ohm]:",winner["z"])

print("\n=== FULL DATA METRICS ===")
print(metrics(YK,Y_PRED))

print("\n=== CROSS-VALIDATED WINNER METRICS ===")
print(
    ranking.iloc[0].to_string()
)


# ============================================================
# 13. OUTLIER REPORT FOR WINNER
# ============================================================

result = d.copy()

result["C_pred"] = C_PRED
result["G_pred"] = G_PRED

result["C_error_F"] = C_PRED-CK
result["G_error_S"] = G_PRED-GK

result["Abs_C_error_F"] = np.abs(
    result["C_error_F"]
)

print("\n=== TOP 20 ERRORS ===")

print(
    result.nlargest(
        20,"Abs_C_error_F"
    )[
        [
            COL_H,COL_L,
            COL_CK,"C_pred","C_error_F"
        ]
    ].to_string(
        index=False,
        float_format=lambda x:f"{x:.6e}"
    )
)


# ============================================================
# 14. SAVE
# ============================================================

screen.to_csv(
    OUTDIR/"all_screening_results.csv",
    index=False
)

final_cv.to_csv(
    OUTDIR/"finalist_fold_results.csv",
    index=False
)

ranking.to_csv(
    OUTDIR/"final_ranking.csv",
    index=False
)

result.to_csv(
    OUTDIR/"winner_predictions.csv",
    index=False
)

joblib.dump({
    "model":winner,
    "frequency_hz":FREQ,
    "feature_key":winner["feature_key"],
    "training_feature_matrix":X_best,
    "metrics":ranking.iloc[0].to_dict()
},OUTDIR/"best_model.joblib")

print("\nSAVED:",OUTDIR.resolve())
