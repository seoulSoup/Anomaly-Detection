
import numpy as np
import pandas as pd
import joblib

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ==========================================================
# 0. SETTINGS
# ==========================================================

IMP_MODE = "SharedPath"
PAIR_MODES = ["Sum", "Sqrt"]

# Start with 2 folds to check results.
# Increase to 5 for final validation.
N_FOLDS = 2
TEST_SIZE = 0.25
SEED = 42

FREQ = 10000.0
B_PER_PF = 2 * np.pi * FREQ * 1e-3

C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

# Identical Pin / SharedPath priors for all models.
PIN_G_SIGMA = 3.0       # nS
PIN_B_SIGMA = 30.0      # nS
ZS_SIGMA = 100.0        # MOhm
ZS_REG_WEIGHT = 0.05

# Pair coefficient priors [pF]
PAIR_SIGMA = np.array([0.5, 0.5, 0.2])

# Pair coefficient upper bound [pF]
PAIR_MAX = np.array([5.0, 5.0, 3.0])

MAX_NFEV = 200
N = len(C_TRUE)

assert H.shape == (N, 44)
assert L.shape == (N, 44)


# ==========================================================
# 1. BUILD PAIR FEATURES
#
# 0: Same-SMU adjacent, larger pin even
# 1: Same-SMU adjacent, larger pin odd
# 2: Same-SMU distance 2
# 3: No pair correction
# ==========================================================

smu_id = np.zeros(44, dtype=int)

for g, pins in enumerate(SMUS):
    for p in pins:
        smu_id[p - 1] = g


def pair_zone(a, b):
    if smu_id[a] != smu_id[b]:
        return 3

    d = abs(a - b)

    if d == 1:
        return 0 if max(a, b) % 2 == 1 else 1

    if d == 2:
        return 2

    return 3


# Precompute pair count for Sum and padded zone
# matrices for weighted Sqrt aggregation.

high_lists = [
    np.flatnonzero(row > 0)
    for row in H
]

low_lists = [
    np.flatnonzero(row > 0)
    for row in L
]

max_h = max(len(x) for x in high_lists)
max_l = max(len(x) for x in low_lists)

# Index 3 means zero contribution.
PAIR_INDEX = np.full(
    (N, max_h, max_l), 3, dtype=np.int8
)

PAIR_COUNTS = np.zeros((N, 3), dtype=float)

for k, (hh, ll) in enumerate(
    zip(high_lists, low_lists)
):
    for i, a in enumerate(hh):
        for j, b in enumerate(ll):
            z = pair_zone(a, b)
            PAIR_INDEX[k, i, j] = z

            if z < 3:
                PAIR_COUNTS[k, z] += 1


# ==========================================================
# 2. PIN + SHARED PATH
#
# theta = [
#   g0, b0, z_shared_real, z_shared_imag,
#   dg[44], db[44]
# ]
#
# Effective pin admittance is in nS.
# Shared impedance is in MOhm.
# ==========================================================

BASE_DIM = 4 + 88

def unpack_base(theta):
    t = theta[:4]
    v = theta[4:BASE_DIM]

    y0 = complex(t[0], t[1])
    zs = complex(t[2], t[3])

    dg = v[:44] - v[:44].mean()
    db = v[44:] - v[44:].mean()

    yp = y0 + dg + 1j*db

    return yp, zs, dg, db


def safe_divide(a, den):
    return (
        a * np.conj(den)
        / (np.abs(den)**2 + 1e-12)
    )


def predict_base(indices, theta):
    yp, zs, _, _ = unpack_base(theta)

    yh_raw = H[indices] @ yp
    yl_raw = L[indices] @ yp

    yh = safe_divide(
        yh_raw, 1 + 1e-3*zs*yh_raw
    )

    yl = safe_divide(
        yl_raw, 1 + 1e-3*zs*yl_raw
    )

    return safe_divide(
        yh * yl, yh + yl
    )


# ==========================================================
# 3. PAIR C AGGREGATION
#
# Sum:
#   signed linear pair residual
#
# Sqrt:
#   weighted full-pair matrix,
#   nonnegative coefficients,
#   sqrt(Sum * MaxPerPin)
# ==========================================================

def pair_correction(indices, alpha, mode):
    if mode == "Sum":
        return PAIR_COUNTS[indices] @ alpha

    if mode == "Sqrt":
        # Padding/category 3 contributes zero.
        coeff = np.r_[alpha, 0.0]

        W = coeff[PAIR_INDEX[indices]]

        S = W.sum(axis=(1, 2))

        max_h = W.max(axis=2).sum(axis=1)
        max_l = W.max(axis=1).sum(axis=1)

        M = 0.5 * (max_h + max_l)

        return np.sqrt(
            np.maximum(S*M, 0.0)
        )

    raise ValueError(mode)


def predict_joint(indices, theta, mode):
    y_base = predict_base(
        indices, theta[:BASE_DIM]
    )

    alpha = theta[BASE_DIM:]

    cp = pair_correction(
        indices, alpha, mode
    )

    return y_base + 1j*B_PER_PF*cp


# ==========================================================
# 4. SAME OBJECTIVE FOR BASE AND JOINT
# ==========================================================

def data_residual(indices, yp):
    cp = yp.imag / B_PER_PF
    gp = yp.real

    return np.r_[
        (cp-C_TRUE[indices]) / C_SCALE_PF,
        (gp-G_TRUE[indices]) / G_SCALE_NS
    ]


def base_regularization(theta):
    yp, zs, dg, db = unpack_base(theta)

    v = theta[4:BASE_DIM]

    return np.r_[
        dg / PIN_G_SIGMA,
        db / PIN_B_SIGMA,
        ZS_REG_WEIGHT * zs.real / ZS_SIGMA,
        ZS_REG_WEIGHT * zs.imag / ZS_SIGMA,
        # Fix redundant mean of the deviations.
        v[:44].mean(),
        v[44:].mean()
    ]


def base_objective(indices, theta):
    return np.r_[
        data_residual(
            indices,
            predict_base(indices, theta)
        ),
        base_regularization(theta)
    ]


def joint_objective(indices, theta, mode):
    alpha = theta[BASE_DIM:]

    return np.r_[
        data_residual(
            indices,
            predict_joint(indices, theta, mode)
        ),
        base_regularization(theta[:BASE_DIM]),
        alpha / PAIR_SIGMA
    ]


# ==========================================================
# 5. FIT BASE WITH ALL PIN PARAMETERS UNFROZEN
#
# Old Common -> Individual fitting is used
# only for initialization.
#
# Then all Base parameters are optimized jointly.
# ==========================================================

BASE_LB = np.r_[
    [-100, -1000, -100, -100],
    np.full(44, -100),
    np.full(44, -300)
]

BASE_UB = np.r_[
    [100, 1000, 100, 100],
    np.full(44, 100),
    np.full(44, 300)
]


def fit_base_joint(train):
    t0 = fit_common(train, IMP_MODE)
    v0 = fit_individual(
        train, t0, IMP_MODE
    )

    x0 = np.r_[t0, v0]

    x0 = np.clip(
        x0, BASE_LB + 1e-8,
        BASE_UB - 1e-8
    )

    result = least_squares(
        lambda x: base_objective(train, x),
        x0,
        bounds=(BASE_LB, BASE_UB),
        max_nfev=MAX_NFEV,
        x_scale="jac"
    )

    return result.x, result


# ==========================================================
# 6. JOINT FIT
#
# Every Pin Y and SharedPath parameter is released.
#
# Base optimum supplies initialization.
# Pair starts close to zero.
# ==========================================================

def fit_joint(train, base_theta, mode):

    if mode == "Sum":
        starts = [
            np.zeros(3)
        ]
        pair_lb = -PAIR_MAX

    else:
        # Sqrt is nonnegative and can have
        # a nondifferentiable origin, so use
        # two small positive initializations.
        starts = [
            np.array([1e-5, 1e-5, 1e-5]),
            np.array([0.05, 0.03, 0.01])
        ]
        pair_lb = np.zeros(3)

    lower = np.r_[BASE_LB, pair_lb]
    upper = np.r_[BASE_UB, PAIR_MAX]

    results = []

    for alpha0 in starts:
        x0 = np.r_[base_theta, alpha0]

        result = least_squares(
            lambda x: joint_objective(
                train, x, mode
            ),
            x0,
            bounds=(lower, upper),
            max_nfev=MAX_NFEV,
            x_scale="jac"
        )

        results.append(result)

    # Select on TRAIN objective only.
    best = min(
        results,
        key=lambda r: np.sum(r.fun**2)
    )

    return best.x, best


# ==========================================================
# 7. METRICS
# ==========================================================

def evaluate(indices, prediction):
    actual = C_TRUE[indices]
    pred_c = prediction.imag / B_PER_PF

    ec = (pred_c - actual) * 1000

    pred_g = prediction.real
    eg = pred_g - G_TRUE[indices]

    return {
        "C_RMSE_fF": float(np.sqrt(np.mean(ec**2))),
        "C_MAE_fF": float(np.mean(np.abs(ec))),
        "C_P95_fF": float(np.percentile(np.abs(ec), 95)),
        "G_RMSE_nS": float(np.sqrt(np.mean(eg**2)))
    }


# ==========================================================
# 8. MATCHED GROUPED SPLITS
# ==========================================================

rows = []
coef_rows = []
prediction_rows = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:

    cv = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        cv.split(H, groups=groups), 1
    ):

        base_theta, base_opt = fit_base_joint(train)

        model_parameters = {
            "Base": base_theta
        }

        model_predictions = {
            "Base": predict_base(test, base_theta)
        }

        for mode in PAIR_MODES:
            joint_theta, opt = fit_joint(
                train, base_theta, mode
            )

            key = f"Joint-{mode}"

            model_parameters[key] = joint_theta

            model_predictions[key] = predict_joint(
                test, joint_theta, mode
            )

            alpha = joint_theta[BASE_DIM:]

            correction = pair_correction(
                test, alpha, mode
            )

            coef_rows.append({
                "Split": split_name,
                "Fold": fold,
                "Model": key,
                "Even_pF": alpha[0],
                "Odd_pF": alpha[1],
                "Dist2_pF": alpha[2],
                "Correction_MeanAbs_fF": float(
                    np.mean(np.abs(correction))*1000
                ),
                "Correction_MaxAbs_fF": float(
                    np.max(np.abs(correction))*1000
                ),
                "Converged": bool(opt.success)
            })

        for key, pred in model_predictions.items():
            m = evaluate(test, pred)

            rows.append({
                "Split": split_name,
                "Fold": fold,
                "Model": key,
                **m
            })

            for k, y in zip(test, pred):
                prediction_rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Row": int(k),
                    "Model": key,
                    "Actual_C_pF": float(C_TRUE[k]),
                    "Pred_C_pF": float(
                        y.imag / B_PER_PF
                    ),
                    "Error_fF": float(
                        (y.imag/B_PER_PF-C_TRUE[k])*1000
                    )
                })

        print(
            f"\n{split_name} / Fold {fold}"
        )

        for key, pred in model_predictions.items():
            m = evaluate(test, pred)

            print(
                f"  {key:12} "
                f"C RMSE={m['C_RMSE_fF']:.2f} fF, "
                f"G RMSE={m['G_RMSE_nS']:.4f} nS"
            )


# ==========================================================
# 9. RESULTS
# ==========================================================

score_df = pd.DataFrame(rows)
coeff_df = pd.DataFrame(coef_rows)
oof_df = pd.DataFrame(prediction_rows)

summary = (
    score_df.groupby(["Split", "Model"])
    .agg(
        RMSE_Mean=("C_RMSE_fF", "mean"),
        RMSE_Std=("C_RMSE_fF", "std"),
        MAE_Mean=("C_MAE_fF", "mean"),
        P95_Mean=("C_P95_fF", "mean"),
        G_RMSE_Mean=("G_RMSE_nS", "mean")
    )
    .reset_index()
)

print("\n=== JOINT ABLATION ===")
print(
    summary.round(4).to_string(index=False)
)

print("\n=== PAIR COEFFICIENTS ===")
print(
    coeff_df.round(5).to_string(index=False)
)


# ==========================================================
# 10. PAIRED RMSE IMPROVEMENT
# ==========================================================

wide = score_df.pivot_table(
    index=["Split", "Fold"],
    columns="Model",
    values="C_RMSE_fF"
).reset_index()

for mode in PAIR_MODES:
    wide[f"{mode}_Benefit_fF"] = (
        wide["Base"] - wide[f"Joint-{mode}"]
    )

print("\n=== JOINT BENEFIT ===")
print(
    wide.round(4).to_string(index=False)
)

benefit_cols = [
    f"{mode}_Benefit_fF"
    for mode in PAIR_MODES
]

print("\n=== BENEFIT MEAN / STD ===")
print(
    wide.groupby("Split")[benefit_cols]
    .agg(["mean", "std"])
    .round(4).to_string()
)


# ==========================================================
# 11. SAVE REPORTS
# ==========================================================

score_df.to_csv(
    "joint_pair_scores.csv", index=False
)

coeff_df.to_csv(
    "joint_pair_coefficients.csv", index=False
)

oof_df.to_csv(
    "joint_pair_predictions.csv", index=False
)

# Full-data model refit should be done
# only after selecting the model from CV.
