
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit

# ======================================================
# 0. CONFIGURATION
# ======================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

C_UNIT = "F"
G_UNIT = "S"      # "S", "uS", "nS"

N_SPLITS = 5
TEST_SIZE = 0.25
SEED = 42

# Local coefficients are fixed [pF].
LOCAL_COEFF = np.array([1.544, 0.605, 0.175])

# N1 nonlinear parameters
INITIAL_C = 1.0     # pF / SMU pin
C_LOWER = 0.005      # pF
C_UPPER = 20.0       # pF

# Regularization:
# penalizes large differences between SMU coefficients
REG_LOG_SIGMA = 0.7
REG_WEIGHT_PF = 0.10

# Baseline regularization
RIDGE_ALPHA = 20.0

SMU_GROUPS = [
    list(range(1, 7)),
    list(range(7, 15)),
    list(range(15, 23)),
    list(range(23, 31)),
    list(range(31, 39)),
    list(range(39, 45))
]

PIN_SMU = {
    pin: g
    for g, pins in enumerate(SMU_GROUPS)
    for pin in pins
}

G_FACTORS = {
    "S": 1.0,
    "uS": 1e-6,
    "nS": 1e-9
}


# ======================================================
# 1. PARSE PINS
# ======================================================

def parse_pins(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        pins = [int(x) for x in value]
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                pins = [int(x) for x in parsed]
            else:
                pins = [
                    int(x)
                    for x in re.findall(r"\d+", value)
                ]
        except (SyntaxError, ValueError):
            pins = [
                int(x)
                for x in re.findall(r"\d+", value)
            ]
    else:
        raise ValueError(f"Invalid pins: {value}")

    if not pins or len(pins) != len(set(pins)):
        raise ValueError(f"Empty or duplicate pins: {value}")

    if any(p not in PIN_SMU for p in pins):
        raise ValueError(f"Invalid pin number: {value}")

    return tuple(sorted(pins))


# ======================================================
# 2. FEATURE EXTRACTION
# ======================================================

def extract_features(high, low):
    high = parse_pins(high)
    low = parse_pins(low)

    if set(high) & set(low):
        raise ValueError("High/Low overlap")

    hc = np.zeros(6)
    lc = np.zeros(6)

    for p in high:
        hc[PIN_SMU[p]] += 1

    for p in low:
        lc[PIN_SMU[p]] += 1

    ne = no = n2 = 0

    for a in high:
        for b in low:
            if PIN_SMU[a] != PIN_SMU[b]:
                continue

            d = abs(a - b)

            if d == 1:
                if max(a, b) % 2 == 0:
                    ne += 1
                else:
                    no += 1
            elif d == 2:
                n2 += 1

    local_pf = (
        LOCAL_COEFF[0] * ne
        + LOCAL_COEFF[1] * no
        + LOCAL_COEFF[2] * n2
    )

    signature = tuple(
        np.r_[hc, lc].astype(int)
    )

    configuration = (high, low)

    return hc, lc, local_pf, signature, configuration


# ======================================================
# 3. BASELINE: LINEAR SMU FEATURES
# ======================================================

def smu_linear_features(h, l):
    h = np.asarray(h, dtype=float)
    l = np.asarray(l, dtype=float)

    # Accept either one sample or a matrix.
    if h.ndim == 1:
        h = h[None, :]
        l = l[None, :]

    parts = [h, l, h*l]

    cross = []

    for g in range(6):
        for k in range(g + 1, 6):
            cross.append(
                h[:, g] * l[:, k]
                + h[:, k] * l[:, g]
            )

    parts.append(np.stack(cross, axis=1))

    nh = h.sum(axis=1)
    nl = l.sum(axis=1)

    parts.append(np.column_stack([
        nh * nl,
        np.log1p(nh),
        np.log1p(nl)
    ]))

    return np.column_stack(parts)


def fit_linear(X, target):
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=RIDGE_ALPHA)
    )
    model.fit(X, target)
    return model


# ======================================================
# 4. NONLINEAR SMU MODEL
#
# C_line = C_H * C_L / (C_H + C_L)
#
# C_H = sum(c_g * h_g)
# C_L = sum(c_g * l_g)
#
# theta = [C0, c1, ..., c6]
#
# Units: pF
# ======================================================

def nonlinear_c_base(theta, h, l):
    c0 = theta[0]
    c_smu = theta[1:]

    ch = h @ c_smu
    cl = l @ c_smu

    denominator = ch + cl

    c_line = np.divide(
        ch * cl,
        denominator,
        out=np.zeros_like(ch, dtype=float),
        where=denominator > 0
    )

    return c0 + c_line


def fit_nonlinear(h, l, target_base):
    # Starting value of C0 estimated from residual target.
    start_smu = np.full(6, INITIAL_C)

    initial_line = nonlinear_c_base(
        np.r_[0.0, start_smu], h, l
    )

    start_c0 = np.median(
        target_base - initial_line
    )

    theta0 = np.r_[start_c0, start_smu]

    lower = np.r_[-100.0, np.full(6, C_LOWER)]
    upper = np.r_[100.0, np.full(6, C_UPPER)]

    def residual(theta):
        predicted = nonlinear_c_base(theta, h, l)
        data_resid = predicted - target_base

        # Soft regularization:
        # encourage similar SMU effective coefficients
        # without fixing a particular SMU.
        logs = np.log(theta[1:])
        reg = (
            logs - np.mean(logs)
        ) / REG_LOG_SIGMA * REG_WEIGHT_PF

        return np.r_[data_resid, reg]

    result = least_squares(
        residual,
        theta0,
        bounds=(lower, upper),
        max_nfev=10000,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11
    )

    if not result.success:
        print("Warning: nonlinear fit:", result.message)

    return result.x


# ======================================================
# 5. DATA PREPARATION
# ======================================================

data = df.copy().reset_index(drop=True)

required = ["high_pins", "low_pins", "C", "G"]

for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

if G_UNIT not in G_FACTORS:
    raise ValueError("Invalid G_UNIT")

features = [
    extract_features(h, l)
    for h, l in zip(
        data["high_pins"], data["low_pins"]
    )
]

H = np.vstack([f[0] for f in features])
L = np.vstack([f[1] for f in features])

LOCAL = np.array([f[2] for f in features])

SIGNATURE = np.array([
    str(f[3]) for f in features
])

# Exact High/Low configurations as group IDs.
CONFIG = np.array([
    str(f[4]) for f in features
])

# C must be in F.
C_TRUE = pd.to_numeric(
    data["C"]
).to_numpy(dtype=float) * 1e12

# Convert G to nS.
G_TRUE = (
    pd.to_numeric(
        data["G"]
    ).to_numpy(dtype=float)
    * G_FACTORS[G_UNIT] * 1e9
)

if not (
    np.isfinite(C_TRUE).all()
    and np.isfinite(G_TRUE).all()
):
    raise ValueError("Non-finite C/G data")

C_BASE = C_TRUE - LOCAL

X_LINEAR = smu_linear_features(H, L)

print("Measurements:", len(data))
print("Unique Configurations:", len(set(CONFIG)))
print("Unique Signatures:", len(set(SIGNATURE)))
print("C range [pF]:", C_TRUE.min(), C_TRUE.max())
print("G range [nS]:", G_TRUE.min(), G_TRUE.max())


# ======================================================
# 6. METRICS
# ======================================================

def calc_metrics(actual, pred):
    err = pred - actual

    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MedianAE": np.median(np.abs(err)),
        "P95AE": np.percentile(np.abs(err), 95),
        "Bias": np.mean(err)
    }


# ======================================================
# 7. EVALUATE ONE SPLIT
# ======================================================

def evaluate_once(train, test, split_type, fold):
    # Model M0: existing SMU linear baseline
    model_m0 = fit_linear(
        X_LINEAR[train],
        C_BASE[train]
    )

    pred_m0 = (
        model_m0.predict(X_LINEAR[test])
        + LOCAL[test]
    )

    # Model N1: nonlinear capacitance network
    theta = fit_nonlinear(
        H[train],
        L[train],
        C_BASE[train]
    )

    pred_n1 = (
        nonlinear_c_base(
            theta,
            H[test],
            L[test]
        )
        + LOCAL[test]
    )

    # G is currently independent of the C network:
    # retain SMU linear model for absolute G.
    model_g = fit_linear(
        X_LINEAR[train],
        G_TRUE[train]
    )

    pred_g = model_g.predict(
        X_LINEAR[test]
    )

    rows = []

    for name, c_pred in [
        ("M0_Linear", pred_m0),
        ("N1_Nonlinear", pred_n1)
    ]:
        cm = calc_metrics(C_TRUE[test], c_pred)
        gm = calc_metrics(G_TRUE[test], pred_g)

        rows.append({
            "Split": split_type,
            "Fold": fold,
            "Model": name,
            "C_RMSE_fF": cm["RMSE"]*1000,
            "C_MAE_fF": cm["MAE"]*1000,
            "C_P95_fF": cm["P95AE"]*1000,
            "C_Bias_fF": cm["Bias"]*1000,
            "G_RMSE_nS": gm["RMSE"]
        })

    parameter_row = {
        "Split": split_type,
        "Fold": fold,
        "C0_pF": theta[0]
    }

    for i in range(6):
        parameter_row[f"c{i+1}_pF"] = theta[i+1]

    return rows, parameter_row


# ======================================================
# 8. MULTIPLE HOLDOUT VALIDATION
#
# Random:
#   exact High/Low configs are grouped together
#   to prevent repeated readings leaking into test.
#
# Unseen:
#   entire SMU signatures are held out.
# ======================================================

all_scores = []
all_parameters = []

for split_type, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        splitter.split(X_LINEAR, groups=groups),
        start=1
    ):
        rows, parameters = evaluate_once(
            train, test, split_type, fold
        )

        all_scores.extend(rows)
        all_parameters.append(parameters)

scores = pd.DataFrame(all_scores)
parameters = pd.DataFrame(all_parameters)


# ======================================================
# 9. SCORE REPORT
# ======================================================

summary = (
    scores.groupby(["Split", "Model"])
    .agg(
        C_RMSE_mean=("C_RMSE_fF", "mean"),
        C_RMSE_std=("C_RMSE_fF", "std"),
        C_MAE_mean=("C_MAE_fF", "mean"),
        C_P95_mean=("C_P95_fF", "mean"),
        G_RMSE_mean=("G_RMSE_nS", "mean")
    )
    .reset_index()
)

print("\n" + "="*78)
print("ABSOLUTE CAPACITANCE MODEL COMPARISON")
print("="*78)
print(summary.round(3).to_string(index=False))


# ======================================================
# 10. SMU PARAMETER STABILITY
# ======================================================

param_cols = (
    ["C0_pF"]
    + [f"c{i}_pF" for i in range(1, 7)]
)

print("\n" + "="*78)
print("NONLINEAR PARAMETER STABILITY")
print("="*78)

param_summary = (
    parameters.groupby("Split")[param_cols]
    .agg(["mean", "std"])
)

print(param_summary.round(4).to_string())


# ======================================================
# 11. PLOT: CV C RMSE
# ======================================================

fig, ax = plt.subplots(figsize=(9, 5))

for model_name, marker in [
    ("M0_Linear", "o"),
    ("N1_Nonlinear", "s")
]:
    subset = scores[
        scores["Model"] == model_name
    ]

    x = np.where(
        subset["Split"] == "RandomConfig",
        0, 1
    ).astype(float)

    offset = (
        -0.09 if model_name == "M0_Linear"
        else 0.09
    )

    ax.scatter(
        x + offset,
        subset["C_RMSE_fF"],
        label=model_name,
        marker=marker,
        alpha=0.75
    )

ax.set_xticks([0, 1])
ax.set_xticklabels([
    "Random Config",
    "Unseen Signature"
])
ax.set_ylabel("C RMSE (fF)")
ax.set_title("Absolute C Prediction: 5-Fold Repeated Holdout")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()


# ======================================================
# 12. FIT DEPLOYMENT MODELS ON ALL DATA
#
# No reference measurement required at inference.
# ======================================================

theta_final = fit_nonlinear(
    H, L, C_BASE
)

g_model_final = fit_linear(
    X_LINEAR, G_TRUE
)

m0_final = fit_linear(
    X_LINEAR, C_BASE
)

print("\nFINAL NONLINEAR PARAMETERS")

print(f"C0 : {theta_final[0]:.5f} pF")

for i in range(6):
    print(
        f"SMU{i+1}: {theta_final[i+1]:.5f} pF/pin"
    )

joblib.dump(
    {
        "theta": theta_final,
        "g_model": g_model_final,
        "m0_model": m0_final,
        "frequency": FREQ,
        "version": "SMU_Nonlinear_v2"
    },
    "smu_nonlinear_v2.joblib"
)


# ======================================================
# 13. INDEPENDENT ABSOLUTE ADMITTANCE PREDICTION
# ======================================================

def predict_admittance(high, low):
    hc, lc, local_pf, _, _ = (
        extract_features(high, low)
    )

    h = hc.reshape(1, -1)
    l = lc.reshape(1, -1)

    c_base_pf = nonlinear_c_base(
        theta_final, h, l
    )[0]

    c_pred_pf = c_base_pf + local_pf

    x = smu_linear_features(h, l)

    g_pred_ns = g_model_final.predict(x)[0]

    y_pred = (
        g_pred_ns * 1e-9
        + 1j * OMEGA * c_pred_pf * 1e-12
    )

    return {
        "C_pF": float(c_pred_pf),
        "G_nS": float(g_pred_ns),
        "Y_S": complex(y_pred)
    }


# ======================================================
# 14. EXAMPLE
# ======================================================

test_high = list(range(15, 45))
test_low = list(range(7, 14))

result = predict_admittance(
    test_high, test_low
)

print("\nINDEPENDENT PREDICTION")
print("High:", test_high)
print("Low :", test_low)

print(f"C = {result['C_pF']:.6f} pF")
print(f"G = {result['G_nS']:.6f} nS")
print(f"Y = {result['Y_S']} S")

# Optional exports
scores.to_csv("smu_nonlinear_scores.csv", index=False)
parameters.to_csv("smu_nonlinear_parameters.csv", index=False)
