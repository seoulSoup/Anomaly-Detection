
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
B_PER_PF = 2 * np.pi * FREQ * 1e-3

C_UNIT = "F"
G_UNIT = "S"

C_FACTORS = {"F": 1e12, "pF": 1.0}
G_FACTORS = {"S": 1e9, "uS": 1e3, "nS": 1.0}

N_FOLDS = 5
SEED = 42
TEST_SIZE = 0.25

# Normalize C/G residuals.
C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

# Parameter units:
# Zs: MOhm
# Yp: nS
#
# Initial broad exploratory bounds.
ZS_BOUND_MOHM = 100.0
YP_BOUND_NS = 100.0

MODELS = [
    "Identity",
    "Open-offset",
    "Series-only",
    "Series+Shunt"
]

# ==========================================================
# DATA
# ==========================================================

data = df.copy().reset_index(drop=True)

required = [
    "C_keysight", "G_keysight",
    "C_v93k", "G_v93k"
]

for name in required:
    if name not in data:
        raise ValueError(f"Missing column: {name}")

ck = data["C_keysight"].to_numpy(float) * C_FACTORS[C_UNIT]
gk = data["G_keysight"].to_numpy(float) * G_FACTORS[G_UNIT]

cv = data["C_v93k"].to_numpy(float) * C_FACTORS[C_UNIT]
gv = data["G_v93k"].to_numpy(float) * G_FACTORS[G_UNIT]

# Complex admittances in nS
YK = gk + 1j * B_PER_PF * ck
YV = gv + 1j * B_PER_PF * cv

if not (np.isfinite(YK).all() and np.isfinite(YV).all()):
    raise ValueError("Nonfinite measurement")

# Hold out entire H/L combinations.
# This tests generalization to unseen configurations.
def parse_pins(x):
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            x = re.findall(r"\d+", x)

    return tuple(sorted(map(int, x)))

if {"high_pins", "low_pins"}.issubset(data.columns):
    GROUPS = np.array([
        str((parse_pins(h), parse_pins(l)))
        for h, l in zip(
            data["high_pins"], data["low_pins"]
        )
    ])
else:
    raise ValueError(
        "For grouped CV, provide high_pins and low_pins"
    )

# ==========================================================
# COMPLEX CORRECTION
#
# Zs in MOhm, Y in nS.
# Zs * Y requires factor 1e-3.
# ==========================================================

def decode_params(theta, mode):
    if mode == "Identity":
        return 0j, 0j

    if mode == "Open-offset":
        yp = theta[0] + 1j * theta[1]
        return 0j, yp

    if mode == "Series-only":
        zs = theta[0] + 1j * theta[1]
        return zs, 0j

    if mode == "Series+Shunt":
        zs = theta[0] + 1j * theta[1]
        yp = theta[2] + 1j * theta[3]
        return zs, yp

    raise ValueError(mode)


def correct_y(yv, theta, mode):
    zs, yp = decode_params(theta, mode)

    denominator = 1 - 1e-3 * zs * yv

    with np.errstate(divide="ignore", invalid="ignore"):
        y_corrected = yv / denominator - yp

    return y_corrected


# ==========================================================
# OBJECTIVE
# ==========================================================

def residual(indices, theta, mode):
    yp = correct_y(YV[indices], theta, mode)

    if not np.isfinite(yp).all():
        return np.full(2 * len(indices), 1e8)

    dc = (yp.imag - YK[indices].imag) / B_PER_PF
    dg = yp.real - YK[indices].real

    # Penalize near-singular inversions.
    zs, _ = decode_params(theta, mode)
    denominator = 1 - 1e-3 * zs * YV[indices]

    if np.min(np.abs(denominator)) < 0.05:
        return np.full(2 * len(indices), 1e8)

    return np.r_[
        dc / C_SCALE_PF,
        dg / G_SCALE_NS
    ]


def fit_model(indices, mode):
    if mode == "Identity":
        return np.array([])

    n = 4 if mode == "Series+Shunt" else 2

    # Bounds follow parameter order.
    if mode == "Series+Shunt":
        bound = np.array([
            ZS_BOUND_MOHM, ZS_BOUND_MOHM,
            YP_BOUND_NS, YP_BOUND_NS
        ])
    elif mode == "Open-offset":
        bound = np.array([
            YP_BOUND_NS, YP_BOUND_NS
        ])
    else:
        bound = np.array([
            ZS_BOUND_MOHM, ZS_BOUND_MOHM
        ])

    starts = [np.zeros(n)]

    if mode in ("Series-only", "Series+Shunt"):
        x = np.zeros(n)
        x[0] = 1.0
        starts.append(x)

    fitted = []

    for start in starts:
        opt = least_squares(
            lambda t: residual(indices, t, mode),
            start,
            bounds=(-bound, bound),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=800,
            x_scale="jac"
        )

        fitted.append(opt)

    return min(fitted, key=lambda o: o.cost).x


# ==========================================================
# EVALUATION
# ==========================================================

def evaluate(indices, theta, mode):
    yp = correct_y(YV[indices], theta, mode)

    c_pred = yp.imag / B_PER_PF
    g_pred = yp.real

    c_actual = YK[indices].imag / B_PER_PF
    g_actual = YK[indices].real

    ec = (c_pred - c_actual) * 1000
    eg = g_pred - g_actual

    return {
        "C_RMSE_fF": np.sqrt(np.mean(ec**2)),
        "C_MAE_fF": np.mean(np.abs(ec)),
        "G_RMSE_nS": np.sqrt(np.mean(eg**2)),
        "G_MAE_nS": np.mean(np.abs(eg))
    }


# ==========================================================
# GROUPED CROSS VALIDATION
# ==========================================================

rows = []
param_rows = []

cv_splitter = GroupShuffleSplit(
    n_splits=N_FOLDS,
    test_size=TEST_SIZE,
    random_state=SEED
)

for fold, (train, test) in enumerate(
    cv_splitter.split(YV, groups=GROUPS),
    start=1
):
    for mode in MODELS:
        theta = fit_model(train, mode)

        train_score = evaluate(train, theta, mode)
        test_score = evaluate(test, theta, mode)

        zs, yp = decode_params(theta, mode)

        rows.append({
            "Fold": fold,
            "Model": mode,
            "Train_C_RMSE_fF": train_score["C_RMSE_fF"],
            **test_score
        })

        param_rows.append({
            "Fold": fold,
            "Model": mode,
            "Zs_real_MOhm": zs.real,
            "Zs_imag_MOhm": zs.imag,
            "Yp_real_nS": yp.real,
            "Yp_imag_nS": yp.imag
        })

        print(
            f"Fold {fold} {mode:14} "
            f"C RMSE={test_score['C_RMSE_fF']:.2f} fF "
            f"G RMSE={test_score['G_RMSE_nS']:.4f} nS"
        )


scores = pd.DataFrame(rows)
params = pd.DataFrame(param_rows)

summary = (
    scores.groupby("Model")
    .agg(
        C_RMSE_mean=("C_RMSE_fF", "mean"),
        C_RMSE_std=("C_RMSE_fF", "std"),
        G_RMSE_mean=("G_RMSE_nS", "mean"),
        G_RMSE_std=("G_RMSE_nS", "std"),
        Train_C_RMSE=("Train_C_RMSE_fF", "mean")
    )
    .sort_values("C_RMSE_mean")
)

print("\n=== DUT COMPLEX Y CALIBRATION ===")
print(summary.round(4).to_string())

print("\n=== PARAMETER STABILITY ===")
print(
    params.groupby("Model")[
        [
            "Zs_real_MOhm",
            "Zs_imag_MOhm",
            "Yp_real_nS",
            "Yp_imag_nS"
        ]
    ].agg(["mean", "std"]).round(6).to_string()
)


# ==========================================================
# FULL-DATA FIT
# ==========================================================

all_idx = np.arange(len(YV))

final_models = {}

for mode in MODELS:
    theta = fit_model(all_idx, mode)

    zs, yp = decode_params(theta, mode)

    final_models[mode] = {
        "theta": theta,
        "Zs_MOhm": zs,
        "Yp_nS": yp
    }

    print(
        mode,
        "Zs =", zs,
        "Yp =", yp
    )


# ==========================================================
# CORRECT V93K TO KEYSIGHT-EQUIVALENT C/G
# ==========================================================

def predict_keysight(c_v93k, g_v93k, mode="Series+Shunt"):
    theta = final_models[mode]["theta"]

    yv = (
        np.asarray(g_v93k) * G_FACTORS[G_UNIT]
        + 1j * B_PER_PF
        * np.asarray(c_v93k) * C_FACTORS[C_UNIT]
    )

    yp = correct_y(yv, theta, mode)

    c_corrected_pf = yp.imag / B_PER_PF
    g_corrected_ns = yp.real

    return c_corrected_pf, g_corrected_ns


scores.to_csv("dut_calibration_folds.csv", index=False)
params.to_csv("dut_calibration_parameters.csv", index=False)
summary.to_csv("dut_calibration_summary.csv")
