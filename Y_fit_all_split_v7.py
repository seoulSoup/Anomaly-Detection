
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit
import joblib

# ==========================================================
# 0. REQUIREMENTS FROM PREVIOUS CODE
#
# df, X, pair_c, IMP_MODES, PAIR_MODES
#
# X fields:
#   H, L, counts, h_seen, l_seen, config, signature
#
# pair_c(X, idx, cross_pf, pair_mode)
# returns empirical pair capacitance in pF.
# ==========================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

# Conversion: pF -> susceptance in nS
B_PER_PF = OMEGA * 1e-3

# IMPORTANT: set the actual unit of df["G"]
G_UNIT = "S"   # "S", "uS", or "nS"
G_TO_NS = {"S": 1e9, "uS": 1e3, "nS": 1.0}

N_FOLDS = 2
TEST_SIZE = 0.25
SEED = 42

# Loss scales. Adjust after inspecting data noise.
C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

# Individual-pin regularization:
# effective RMS tolerances for pin differences.
PIN_G_SIGMA_NS = 1.0
PIN_B_SIGMA_NS = 20.0
PIN_REG_WEIGHT = 1.0

# Optional regularization for common parameters.
# Larger sigma => weaker prior.
COMMON_G_SIGMA_NS = 10.0
COMMON_B_SIGMA_NS = 100.0

IMP_MODES = ["Parallel", "Mean", "SharedPath"]
PAIR_MODES = ["Sum", "MaxPerPin", "Sqrt", "PartialSum"]

# ----------------------------------------------------------
# Use original df to construct measured complex admittance.
# ----------------------------------------------------------

C_TRUE_PF = pd.to_numeric(
    df["C"]
).to_numpy(dtype=float) * 1e12

G_TRUE_NS = pd.to_numeric(
    df["G"]
).to_numpy(dtype=float) * G_TO_NS[G_UNIT]

if len(C_TRUE_PF) != len(X["H"]):
    raise ValueError("df and X have different row counts")

if not (
    np.isfinite(C_TRUE_PF).all()
    and np.isfinite(G_TRUE_NS).all()
):
    raise ValueError("Non-finite C/G values")

Y_TRUE_NS = (
    G_TRUE_NS + 1j * B_PER_PF * C_TRUE_PF
)

print("C range [pF]:", C_TRUE_PF.min(), C_TRUE_PF.max())
print("G range [nS]:", G_TRUE_NS.min(), G_TRUE_NS.max())

# ==========================================================
# 1. COMPLEX ADMITTANCE AGGREGATION
#
# Complex pin admittance in nS.
#
# SharedPath:
#  Z_shared is in MOhm
#  Y is in nS
#  Z(MOhm)*Y(nS) = 1e-3 dimensionless
# ==========================================================

def aggregate_complex(mask, y_pin, mode, z_shared=None):
    total = mask @ y_pin

    if mode == "Parallel":
        return total

    if mode == "Mean":
        count = mask.sum(axis=1)
        return np.divide(
            total, count,
            out=np.zeros_like(total),
            where=count > 0
        )

    if mode == "SharedPath":
        denominator = (
            1.0 + 1e-3*z_shared*total
        )
        return total / denominator

    raise ValueError(mode)


def line_admittance(idx, y_pin, mode, z_shared=None):
    yh = aggregate_complex(
        X["H"][idx], y_pin, mode, z_shared
    )
    yl = aggregate_complex(
        X["L"][idx], y_pin, mode, z_shared
    )

    denominator = yh + yl

    out = np.zeros_like(yh, dtype=complex)
    valid = np.abs(denominator) > 1e-12

    out[valid] = (
        yh[valid]*yl[valid]/denominator[valid]
    )

    return out


def predict_complex(
    idx, y_pin, cross_pf, imp_mode,
    pair_mode, z_shared=None
):
    y_line = line_admittance(
        idx, y_pin, imp_mode, z_shared
    )

    c_pair = pair_c(
        X, idx, cross_pf, pair_mode
    )

    return y_line + 1j*B_PER_PF*c_pair


# ==========================================================
# 2. DATA ERROR IN C AND G
# ==========================================================

def data_residual(pred, idx):
    pred_g = pred.real
    pred_c = pred.imag / B_PER_PF

    return np.r_[
        (pred_c - C_TRUE_PF[idx]) / C_SCALE_PF,
        (pred_g - G_TRUE_NS[idx]) / G_SCALE_NS
    ]


# ==========================================================
# 3. STAGE 1: COMMON COMPLEX PIN ADMITTANCE
#
# Parameters:
#  g0, b0   [nS]
#  cross_pf [pF]
#  SharedPath only: zs_re, zs_im [MOhm]
#
# Both real and imaginary parts are fitted.
# ==========================================================

def decode_common(t, mode):
    g0, b0, cross_pf = t[:3]

    z_shared = None
    if mode == "SharedPath":
        z_shared = complex(t[3], t[4])

    return complex(g0, b0), cross_pf, z_shared


def predict_common(idx, t, imp_mode, pair_mode):
    y0, cross_pf, z_shared = decode_common(
        t, imp_mode
    )

    y_pin = np.full(44, y0, dtype=complex)

    return predict_complex(
        idx, y_pin, cross_pf,
        imp_mode, pair_mode, z_shared
    )


def fit_common(train, imp_mode, pair_mode):
    # Pin susceptance initial value: equivalent 0.5 pF.
    initial = [
        0.1, 0.5*B_PER_PF, 0.01
    ]

    lower = [-1000.0, -1000.0, 0.0]
    upper = [1000.0, 1000.0, 2.0]

    if imp_mode == "SharedPath":
        # Complex shared impedance in MOhm
        initial += [1.0, 0.0]
        lower += [-1000.0, -1000.0]
        upper += [1000.0, 1000.0]

    def residual(t):
        pred = predict_common(
            train, t, imp_mode, pair_mode
        )

        # Weak preference for modest common G/B.
        reg = [
            t[0] / COMMON_G_SIGMA_NS,
            t[1] / COMMON_B_SIGMA_NS
        ]

        if imp_mode == "SharedPath":
            # Prevent near-singular shared paths.
            reg.extend([
                t[3] / 300.0,
                t[4] / 300.0
            ])

        return np.r_[
            data_residual(pred, train),
            reg
        ]

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=500,
        x_scale="jac"
    )

    if not result.success:
        print("Common fit warning:", result.message)

    return result.x


# ==========================================================
# 4. STAGE 2: INDIVIDUAL COMPLEX PIN ADMITTANCES
#
# Stage 1 parameters are fixed.
#
# y_i = y0 + dG_i + j*dB_i
#
# We center deviations to preserve common level.
# ==========================================================

def decode_individual(t_common, v, mode):
    y0, cross_pf, z_shared = decode_common(
        t_common, mode
    )

    dg = v[:44] - np.mean(v[:44])
    db = v[44:] - np.mean(v[44:])

    y_pin = y0 + dg + 1j*db

    return y_pin, cross_pf, z_shared


def predict_individual(
    idx, t_common, v, imp_mode, pair_mode
):
    y_pin, cross_pf, z_shared = decode_individual(
        t_common, v, imp_mode
    )

    return predict_complex(
        idx, y_pin, cross_pf,
        imp_mode, pair_mode, z_shared
    )


def fit_individual(train, t_common, imp_mode, pair_mode):
    def residual(v):
        pred = predict_individual(
            train, t_common, v,
            imp_mode, pair_mode
        )

        dg = v[:44] - np.mean(v[:44])
        db = v[44:] - np.mean(v[44:])

        reg = PIN_REG_WEIGHT * np.r_[
            dg / PIN_G_SIGMA_NS,
            db / PIN_B_SIGMA_NS
        ]

        return np.r_[
            data_residual(pred, train),
            reg
        ]

    result = least_squares(
        residual,
        np.zeros(88),
        bounds=(
            np.r_[
                np.full(44, -100.0),
                np.full(44, -300.0)
            ],
            np.r_[
                np.full(44, 100.0),
                np.full(44, 300.0)
            ]
        ),
        max_nfev=300,
        x_scale="jac"
    )

    if not result.success:
        print("Individual fit warning:", result.message)

    return result.x


# ==========================================================
# 5. METRICS
# ==========================================================

def evaluate(y_pred, idx):
    pred_c = y_pred.imag / B_PER_PF
    pred_g = y_pred.real

    ec_ff = (
        pred_c - C_TRUE_PF[idx]
    ) * 1000

    eg_ns = pred_g - G_TRUE_NS[idx]

    return {
        "C_RMSE_fF": np.sqrt(np.mean(ec_ff**2)),
        "C_MAE_fF": np.mean(np.abs(ec_ff)),
        "G_RMSE_nS": np.sqrt(np.mean(eg_ns**2)),
        "G_MAE_nS": np.mean(np.abs(eg_ns)),
        "Combined": np.sqrt(np.mean(
            (ec_ff / (1000*C_SCALE_PF))**2
            + (eg_ns / G_SCALE_NS)**2
        ))
    }


# ==========================================================
# 6. ALL 12 COMBINATIONS
# ==========================================================

rows = []

for split_name, groups in [
    ("RandomConfig", X["config"]),
    ("UnseenSignature", X["signature"])
]:
    cv = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        cv.split(X["H"], groups=groups),
        start=1
    ):
        for imp_mode in IMP_MODES:
            for pair_mode in PAIR_MODES:

                # Common complex pin model
                t = fit_common(
                    train, imp_mode, pair_mode
                )

                pred0 = predict_common(
                    test, t, imp_mode, pair_mode
                )

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Impedance": imp_mode,
                    "Pair": pair_mode,
                    "PinFit": "Common",
                    **evaluate(pred0, test)
                })

                # Individual complex pin model
                v = fit_individual(
                    train, t, imp_mode, pair_mode
                )

                pred1 = predict_individual(
                    test, t, v, imp_mode, pair_mode
                )

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Impedance": imp_mode,
                    "Pair": pair_mode,
                    "PinFit": "Individual",
                    **evaluate(pred1, test)
                })

                print(
                    split_name, fold,
                    imp_mode, pair_mode,
                    "C RMSE:",
                    round(evaluate(pred1, test)["C_RMSE_fF"], 1),
                    "G RMSE:",
                    round(evaluate(pred1, test)["G_RMSE_nS"], 3)
                )


# ==========================================================
# 7. REPORT
# ==========================================================

complex_scores = pd.DataFrame(rows)

complex_summary = (
    complex_scores.groupby(
        ["Split", "Impedance", "Pair", "PinFit"]
    )
    .agg(
        C_RMSE_mean=("C_RMSE_fF", "mean"),
        C_RMSE_std=("C_RMSE_fF", "std"),
        G_RMSE_mean=("G_RMSE_nS", "mean"),
        G_RMSE_std=("G_RMSE_nS", "std"),
        Combined_mean=("Combined", "mean")
    )
    .reset_index()
)

print("\n=== COMPLEX ADMITTANCE MODEL COMPARISON ===")
print(
    complex_summary.round(3).to_string(index=False)
)

print("\n=== TOP MODELS: UNSEEN SIGNATURE ===")

ranked = (
    complex_summary[
        complex_summary["Split"] == "UnseenSignature"
    ]
    .sort_values("Combined_mean")
)

print(
    ranked.head(12).round(3).to_string(index=False)
)

complex_scores.to_csv(
    "complex_admittance_cv.csv",
    index=False
)

complex_summary.to_csv(
    "complex_admittance_summary.csv",
    index=False
)


# ==========================================================
# 8. FINAL MODEL FIT
# ==========================================================

best = ranked.iloc[0]

BEST_IMP = best["Impedance"]
BEST_PAIR = best["Pair"]
BEST_PINFIT = best["PinFit"]

all_idx = np.arange(len(C_TRUE_PF))

final_t = fit_common(
    all_idx, BEST_IMP, BEST_PAIR
)

final_v = (
    fit_individual(
        all_idx, final_t, BEST_IMP, BEST_PAIR
    )
    if BEST_PINFIT == "Individual"
    else np.zeros(88)
)

y_pin, cross_pf, z_shared = decode_individual(
    final_t, final_v, BEST_IMP
)

# Convert effective pin admittance to impedance.
# y_pin in nS => impedance in Ohms.
z_pin_ohm = 1.0 / (y_pin * 1e-9)

pin_report = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "G_pin_nS": y_pin.real,
    "B_pin_nS": y_pin.imag,
    "R_Z_MOhm": z_pin_ohm.real / 1e6,
    "X_Z_MOhm": z_pin_ohm.imag / 1e6
})

print("\n=== SELECTED MODEL ===")
print(BEST_IMP, BEST_PAIR, BEST_PINFIT)
print("Cross coefficient [pF]:", cross_pf)
print("Shared Z [MOhm]:", z_shared)

print("\n=== PIN COMPLEX ADMITTANCES ===")
print(pin_report.round(4).to_string(index=False))

joblib.dump({
    "impedance_mode": BEST_IMP,
    "pair_mode": BEST_PAIR,
    "pinfit": BEST_PINFIT,
    "common_params": final_t,
    "individual_deviations": final_v,
    "frequency_hz": FREQ
}, "complex_admittance_model.joblib")
