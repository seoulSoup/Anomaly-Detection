
import ast
import re
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 0. SETTINGS - ALL PHYSICAL QUANTITIES IN SI UNITS
# ============================================================

FREQ_HZ = 10000.0
OMEGA = 2.0 * np.pi * FREQ_HZ

N_FOLDS = 5
TEST_SIZE = 0.25
SEED = 42

MODELS = [
    "Identity",
    "Open-offset",
    "Series-only",
    "Series+Shunt"
]

# Objective residual scales (SI)
# These determine relative importance of C and G.
C_SCALE_F = 5e-13
G_SCALE_S = 1e-9

# Physical parameter bounds (SI)
Z_BOUND_OHM = 1e8
YP_BOUND_S = 1e-7

# Numerical optimization references.
# These are used ONLY to parameterize dimensionless
# optimizer variables. All physical calculations use
# Ohm, S, and F.
Z_REF_OHM = 1e6
YP_REF_S = 1e-10

MAX_NFEV = 1000
MIN_DENOMINATOR = 0.05

# ============================================================
# 1. DATA
# ============================================================

data = df.copy().reset_index(drop=True)

required = [
    "C_keysight",
    "G_keysight",
    "C_v93k",
    "G_v93k",
    "high_pins",
    "low_pins"
]

missing = set(required) - set(data.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

C_K = data["C_keysight"].to_numpy(dtype=float)
G_K = data["G_keysight"].to_numpy(dtype=float)

C_V = data["C_v93k"].to_numpy(dtype=float)
G_V = data["G_v93k"].to_numpy(dtype=float)

# Complex admittance [S]
Y_K = G_K + 1j * OMEGA * C_K
Y_V = G_V + 1j * OMEGA * C_V

if not (
    np.isfinite(Y_K).all()
    and np.isfinite(Y_V).all()
):
    raise ValueError("Nonfinite C/G data")


def parse_pins(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = re.findall(r"\d+", value)

    if isinstance(value, (int, np.integer)):
        value = [value]

    pins = tuple(sorted(int(p) for p in value))

    if not pins:
        raise ValueError("Empty pin selection")

    if len(set(pins)) != len(pins):
        raise ValueError("Duplicate pins")

    return pins


HIGH = [
    parse_pins(x) for x in data["high_pins"]
]

LOW = [
    parse_pins(x) for x in data["low_pins"]
]

GROUPS = np.array([
    str((h, l))
    for h, l in zip(HIGH, LOW)
])

N = len(data)

print("Rows:", N)
print("Configurations:", len(np.unique(GROUPS)))

print("\n=== MEASUREMENT RANGE (SI) ===")
print("Keysight C [F]:", C_K.min(), C_K.max())
print("V93K C [F]:", C_V.min(), C_V.max())
print("Keysight G [S]:", G_K.min(), G_K.max())
print("V93K G [S]:", G_V.min(), G_V.max())


# ============================================================
# 2. PARAMETERIZATION
#
# Optimizer variables x are DIMENSIONLESS.
#
# Physical parameters:
#   Zs [Ohm]
#   Yp [S]
#
# This keeps numerical derivatives stable without
# changing physical calculation units.
# ============================================================

def physical_params(x, mode):
    if mode == "Identity":
        return 0j, 0j

    if mode == "Open-offset":
        yp = (
            x[0] + 1j*x[1]
        ) * YP_REF_S

        return 0j, yp

    if mode == "Series-only":
        zs = (
            x[0] + 1j*x[1]
        ) * Z_REF_OHM

        return zs, 0j

    if mode == "Series+Shunt":
        zs = (
            x[0] + 1j*x[1]
        ) * Z_REF_OHM

        yp = (
            x[2] + 1j*x[3]
        ) * YP_REF_S

        return zs, yp

    raise ValueError(mode)


def parameter_bounds(mode):
    if mode == "Identity":
        return np.array([]), np.array([])

    z_limit = Z_BOUND_OHM / Z_REF_OHM
    y_limit = YP_BOUND_S / YP_REF_S

    if mode == "Open-offset":
        upper = np.array([y_limit, y_limit])

    elif mode == "Series-only":
        upper = np.array([z_limit, z_limit])

    else:
        upper = np.array([
            z_limit, z_limit,
            y_limit, y_limit
        ])

    return -upper, upper


# ============================================================
# 3. CORRECTION MODEL
#
# All admittances: S
# All impedances: Ohm
#
# Ycorrected = Yv/(1-Zs*Yv) - Yp
# ============================================================

def correct_y(yv, x, mode):
    zs, yp = physical_params(x, mode)

    denominator = 1.0 - zs*yv

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore"
    ):
        corrected = yv / denominator - yp

    return corrected


# ============================================================
# 4. OBJECTIVE
#
# Residuals are dimensionless:
#   C_error [F] / C_SCALE_F
#   G_error [S] / G_SCALE_S
# ============================================================

def residual(indices, x, mode):
    yv = Y_V[indices]

    zs, _ = physical_params(x, mode)
    denominator = 1.0 - zs*yv

    if (
        not np.isfinite(denominator).all()
        or np.any(np.abs(denominator) < MIN_DENOMINATOR)
    ):
        return np.full(2*len(indices), 1e8)

    yc = correct_y(yv, x, mode)

    if not np.isfinite(yc).all():
        return np.full(2*len(indices), 1e8)

    error_c = (yc.imag - Y_K[indices].imag) / OMEGA
    error_g = yc.real - Y_K[indices].real

    return np.r_[
        error_c / C_SCALE_F,
        error_g / G_SCALE_S
    ]


# ============================================================
# 5. FIT
# ============================================================

def fit_model(indices, mode):
    if mode == "Identity":
        return np.array([], dtype=float)

    lower, upper = parameter_bounds(mode)

    n = len(lower)

    starts = [np.zeros(n)]

    if mode in ("Series-only", "Series+Shunt"):
        x = np.zeros(n)
        x[0] = 1.0
        starts.append(x)

        x = np.zeros(n)
        x[1] = -1.0
        starts.append(x)

    results = []

    for start in starts:
        opt = least_squares(
            lambda x: residual(indices, x, mode),
            start,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=MAX_NFEV,
            x_scale="jac"
        )

        results.append(opt)

    best = min(results, key=lambda o: o.cost)

    return best.x


# ============================================================
# 6. EVALUATION - RAW SI VALUES
# ============================================================

def evaluate(indices, x, mode):
    yc = correct_y(Y_V[indices], x, mode)

    c_pred = yc.imag / OMEGA
    g_pred = yc.real

    ec = c_pred - C_K[indices]
    eg = g_pred - G_K[indices]

    return {
        "C_RMSE_F": float(
            np.sqrt(np.mean(ec**2))
        ),
        "C_MAE_F": float(
            np.mean(np.abs(ec))
        ),
        "C_Bias_F": float(
            np.mean(ec)
        ),
        "G_RMSE_S": float(
            np.sqrt(np.mean(eg**2))
        ),
        "G_MAE_S": float(
            np.mean(np.abs(eg))
        )
    }


# ============================================================
# 7. GROUPED CROSS VALIDATION
# ============================================================

rows = []
param_rows = []

splitter = GroupShuffleSplit(
    n_splits=N_FOLDS,
    test_size=TEST_SIZE,
    random_state=SEED
)

for fold, (train, test) in enumerate(
    splitter.split(np.arange(N), groups=GROUPS),
    start=1
):
    for mode in MODELS:
        x = fit_model(train, mode)

        train_score = evaluate(train, x, mode)
        test_score = evaluate(test, x, mode)

        zs, yp = physical_params(x, mode)

        rows.append({
            "Fold": fold,
            "Model": mode,
            "Train_C_RMSE_F":
                train_score["C_RMSE_F"],
            **test_score
        })

        param_rows.append({
            "Fold": fold,
            "Model": mode,
            "Zs_real_Ohm": float(zs.real),
            "Zs_imag_Ohm": float(zs.imag),
            "Yp_real_S": float(yp.real),
            "Yp_imag_S": float(yp.imag)
        })

        print(
            f"Fold {fold} | {mode:14} | "
            f"C RMSE={test_score['C_RMSE_F']:.6e} F | "
            f"G RMSE={test_score['G_RMSE_S']:.6e} S"
        )


# ============================================================
# 8. SUMMARY
# ============================================================

scores = pd.DataFrame(rows)
params = pd.DataFrame(param_rows)

summary = (
    scores.groupby("Model")
    .agg(
        C_RMSE_mean_F=("C_RMSE_F", "mean"),
        C_RMSE_std_F=("C_RMSE_F", "std"),
        C_MAE_mean_F=("C_MAE_F", "mean"),
        G_RMSE_mean_S=("G_RMSE_S", "mean"),
        G_RMSE_std_S=("G_RMSE_S", "std"),
        Train_C_RMSE_mean_F=("Train_C_RMSE_F", "mean")
    )
    .sort_values("C_RMSE_mean_F")
)

print("\n=== CALIBRATION COMPARISON ===")
print(summary.to_string(float_format=lambda x: f"{x:.6e}"))

print("\n=== PARAMETER STABILITY ===")
print(
    params.groupby("Model")[
        [
            "Zs_real_Ohm",
            "Zs_imag_Ohm",
            "Yp_real_S",
            "Yp_imag_S"
        ]
    ].agg(["mean", "std"])
    .to_string(float_format=lambda x: f"{x:.6e}")
)


# ============================================================
# 9. FINAL FIT ON ALL ROWS
# ============================================================

final_models = {}

all_idx = np.arange(N)

for mode in MODELS:
    x = fit_model(all_idx, mode)

    zs, yp = physical_params(x, mode)

    final_models[mode] = {
        "x": x,
        "Zs_Ohm": zs,
        "Yp_S": yp
    }

    print(f"\n{mode}")
    print(f"  Zs [Ohm] = {zs:.8e}")
    print(f"  Yp [S]   = {yp:.8e}")


# ============================================================
# 10. PREDICT KEYSIGHT-EQUIVALENT C/G
#
# Inputs:
#   C_v93k [F]
#   G_v93k [S]
#
# Outputs:
#   C_corrected [F]
#   G_corrected [S]
# ============================================================

def predict_keysight(
    c_v93k,
    g_v93k,
    mode="Series+Shunt"
):
    x = final_models[mode]["x"]

    c = np.asarray(c_v93k, dtype=float)
    g = np.asarray(g_v93k, dtype=float)

    yv = g + 1j*OMEGA*c

    yc = correct_y(yv, x, mode)

    return yc.imag / OMEGA, yc.real


# ============================================================
# 11. SAVE
# ============================================================

scores.to_csv(
    "dut_calibration_folds_SI.csv",
    index=False
)

params.to_csv(
    "dut_calibration_parameters_SI.csv",
    index=False
)

summary.to_csv(
    "dut_calibration_summary_SI.csv"
)

import joblib

joblib.dump(
    {
        "frequency_hz": FREQ_HZ,
        "models": final_models,
        "Z_ref_Ohm": Z_REF_OHM,
        "Yp_ref_S": YP_REF_S
    },
    "dut_calibration_SI.joblib"
)
