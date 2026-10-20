
import ast
import re
import numpy as np
import pandas as pd
import joblib

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 0. SETTINGS
# ============================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

MODELS = ["Parallel", "Mean", "SharedPath"]

N_FOLDS = 5
TEST_SIZE = 0.25
SEED = 42

# Fixed empirical local increments [pF]
LOCAL_COEFF = np.array([1.544, 0.605, 0.175])

# Regularization in pF residual scale.
# Penalize within-SMU pin variation, not overall scale.
PIN_SIGMA = 0.35
SMU_SIGMA = 0.70
REG_WEIGHT = 0.15

# Bounds on effective pin coefficients [pF]
PIN_MIN = 0.005
PIN_MAX = 100.0

# Shared-path effective capacitance [pF]
SHARED_MIN = 0.005
SHARED_MAX = 500.0

SMU_GROUPS = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g
    for g, group in enumerate(SMU_GROUPS)
    for p in group
}

SMU_IDX = [
    np.array(list(group)) - 1
    for group in SMU_GROUPS
]


# ============================================================
# 1. PIN PARSING
# ============================================================

def parse_pins(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        pins = [int(x) for x in value]

    elif isinstance(value, str):
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, (list, tuple, set)):
                pins = [int(x) for x in obj]
            else:
                pins = [
                    int(x) for x in re.findall(r"\d+", value)
                ]
        except (ValueError, SyntaxError):
            pins = [
                int(x) for x in re.findall(r"\d+", value)
            ]
    else:
        raise ValueError(f"Invalid pin list: {value}")

    if not pins or len(pins) != len(set(pins)):
        raise ValueError(f"Empty/duplicate pins: {value}")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError(f"Pin out of range: {value}")

    return tuple(sorted(pins))


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(high, low):
    high = parse_pins(high)
    low = parse_pins(low)

    if set(high) & set(low):
        raise ValueError("High and Low overlap")

    h = np.zeros(44)
    l = np.zeros(44)

    h[np.array(high) - 1] = 1.0
    l[np.array(low) - 1] = 1.0

    hc = np.array([
        sum(p in high for p in group)
        for group in SMU_GROUPS
    ])

    lc = np.array([
        sum(p in low for p in group)
        for group in SMU_GROUPS
    ])

    # Fixed local increment counts:
    # Only H-L pairs within the same SMU.
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

    local_pf = np.dot(
        LOCAL_COEFF,
        [ne, no, n2]
    )

    signature = tuple(
        int(v) for v in np.r_[hc, lc]
    )

    configuration = (high, low)

    return h, l, local_pf, signature, configuration


# ============================================================
# 3. PREPARE DATA
# ============================================================

data = df.copy().reset_index(drop=True)

for col in ["high_pins", "low_pins", "C"]:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

records = [
    extract_features(h, l)
    for h, l in zip(
        data["high_pins"],
        data["low_pins"]
    )
]

H = np.vstack([r[0] for r in records])
L = np.vstack([r[1] for r in records])

LOCAL = np.array([r[2] for r in records])

SIGNATURE = np.array(
    [str(r[3]) for r in records]
)

CONFIG = np.array(
    [str(r[4]) for r in records]
)

# Input C in Farads -> pF
C_TRUE = pd.to_numeric(
    data["C"]
).to_numpy(dtype=float) * 1e12

if not np.isfinite(C_TRUE).all():
    raise ValueError("Non-finite C values")

C_BASE = C_TRUE - LOCAL

print("Measurements:", len(data))
print("Configurations:", len(set(CONFIG)))
print("Signatures:", len(set(SIGNATURE)))
print("C range (pF):", C_TRUE.min(), C_TRUE.max())


# ============================================================
# 4. AGGREGATION MODELS
#
# Parameters:
#
# Parallel:
#   theta = log(c_pin[44])
#
# Mean:
#   theta = log(c_pin[44])
#
# SharedPath:
#   theta = log(c_pin[44]) + log(c_shared)
#
# All capacitances in pF.
# ============================================================

def decode_theta(theta, model):
    c_pin = np.exp(theta[:44])

    c_shared = None
    if model == "SharedPath":
        c_shared = np.exp(theta[44])

    return c_pin, c_shared


def aggregate_c(h, c_pin, model, c_shared):
    c_sum = h @ c_pin

    if model == "Parallel":
        return c_sum

    elif model == "Mean":
        counts = h.sum(axis=1)

        return np.divide(
            c_sum,
            counts,
            out=np.zeros_like(c_sum),
            where=counts > 0
        )

    elif model == "SharedPath":
        # Capacitive equivalent of a shared series path.
        #
        # C_eff = C_sum * C_shared /
        #         (C_sum + C_shared)
        #
        return (
            c_sum * c_shared
            / (c_sum + c_shared + 1e-30)
        )

    else:
        raise ValueError(model)


def predict_c(theta, model, h, l, local):
    c_pin, c_shared = decode_theta(theta, model)

    ch = aggregate_c(
        h, c_pin, model, c_shared
    )

    cl = aggregate_c(
        l, c_pin, model, c_shared
    )

    c_line = np.divide(
        ch * cl,
        ch + cl,
        out=np.zeros_like(ch),
        where=(ch + cl) > 0
    )

    return c_line + local


# ============================================================
# 5. REGULARIZATION
#
# Important:
# Model B (Mean) has a different natural coefficient
# scale from A (Parallel).
#
# Therefore penalize relative/log deviations,
# not absolute capacitance magnitudes.
# ============================================================

def regularization(theta):
    q = theta[:44]

    group_means = np.array([
        np.mean(q[idx])
        for idx in SMU_IDX
    ])

    within = []

    for g, idx in enumerate(SMU_IDX):
        within.extend(
            (q[idx] - group_means[g])
            / PIN_SIGMA
        )

    between = (
        group_means - group_means.mean()
    ) / SMU_SIGMA

    return REG_WEIGHT * np.r_[
        within, between
    ]


# ============================================================
# 6. INITIALIZATION
#
# Estimate a reasonable starting scale separately
# for each aggregation.
# ============================================================

def initial_parameters(model, train):
    h = H[train]
    l = L[train]

    target = C_BASE[train]

    # Uniform pin coefficients of 1 pF
    uniform = np.ones(44)

    if model == "SharedPath":
        c_shared0 = 8.0
    else:
        c_shared0 = None

    ch = aggregate_c(
        h, uniform, model, c_shared0
    )

    cl = aggregate_c(
        l, uniform, model, c_shared0
    )

    unit_line = np.divide(
        ch*cl, ch+cl,
        out=np.zeros_like(ch),
        where=(ch+cl) > 0
    )

    valid = (
        (unit_line > 1e-9)
        & (target > 0)
    )

    if np.any(valid):
        ratio = np.median(
            target[valid] / unit_line[valid]
        )
    else:
        ratio = 1.0

    initial_c = np.clip(
        ratio, 0.05, 20.0
    )

    if model == "SharedPath":
        # Also scale the shared capacitance so the
        # starting model has a similar magnitude.
        theta0 = np.r_[
            np.full(44, np.log(initial_c)),
            np.log(
                np.clip(
                    c_shared0 * initial_c,
                    SHARED_MIN,
                    SHARED_MAX
                )
            )
        ]
    else:
        theta0 = np.full(
            44, np.log(initial_c)
        )

    return theta0


# ============================================================
# 7. FIT MODEL
# ============================================================

def fit_model(model, train):
    theta0 = initial_parameters(
        model, train
    )

    lower = np.full(
        len(theta0), np.log(PIN_MIN)
    )

    upper = np.full(
        len(theta0), np.log(PIN_MAX)
    )

    if model == "SharedPath":
        lower[-1] = np.log(SHARED_MIN)
        upper[-1] = np.log(SHARED_MAX)

    # SharedPath has an extra parameter, but its
    # pin regularization is identical to A/B.
    def residual(theta):
        pred = predict_c(
            theta,
            model,
            H[train],
            L[train],
            LOCAL[train]
        )

        data_error = pred - C_TRUE[train]

        return np.r_[
            data_error,
            regularization(theta)
        ]

    result = least_squares(
        residual,
        theta0,
        bounds=(lower, upper),
        max_nfev=1200,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9
    )

    if not result.success:
        print(
            f"Warning ({model}):",
            result.message
        )

    return result.x


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    error_fF = (predicted - actual) * 1000

    return {
        "RMSE_fF": np.sqrt(
            np.mean(error_fF**2)
        ),
        "MAE_fF": np.mean(
            np.abs(error_fF)
        ),
        "MedianAE_fF": np.median(
            np.abs(error_fF)
        ),
        "P95AE_fF": np.percentile(
            np.abs(error_fF), 95
        ),
        "Bias_fF": np.mean(error_fF)
    }


# ============================================================
# 9. GROUPED CROSS VALIDATION
#
# RandomConfig:
#   exact H/L configurations stay together.
#
# UnseenSignature:
#   all measurements of one SMU occupancy
#   signature stay together.
#
# Every aggregation is evaluated on precisely
# the same train/test indices.
# ============================================================

scores = []
parameter_records = []

split_results = {}

for split_type, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    splitter = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        splitter.split(H, groups=groups),
        start=1
    ):
        for model in MODELS:
            theta = fit_model(model, train)

            pred_train = predict_c(
                theta,
                model,
                H[train],
                L[train],
                LOCAL[train]
            )

            pred_test = predict_c(
                theta,
                model,
                H[test],
                L[test],
                LOCAL[test]
            )

            train_stats = calculate_metrics(
                C_TRUE[train], pred_train
            )

            test_stats = calculate_metrics(
                C_TRUE[test], pred_test
            )

            scores.append({
                "Split": split_type,
                "Fold": fold,
                "Model": model,
                "Train_RMSE_fF": train_stats["RMSE_fF"],
                **test_stats
            })

            c_pin, c_shared = decode_theta(
                theta, model
            )

            rec = {
                "Split": split_type,
                "Fold": fold,
                "Model": model,
                "PinMin_pF": c_pin.min(),
                "PinMedian_pF": np.median(c_pin),
                "PinMax_pF": c_pin.max(),
                "PinMaxMinRatio": (
                    c_pin.max() / c_pin.min()
                ),
                "Shared_pF": (
                    c_shared if c_shared is not None
                    else np.nan
                )
            }

            parameter_records.append(rec)

            split_results[
                (split_type, fold, model)
            ] = {
                "test_idx": test.copy(),
                "actual_pF": C_TRUE[test].copy(),
                "predicted_pF": pred_test.copy()
            }

            print(
                f"{split_type:16} "
                f"fold={fold} "
                f"{model:11} "
                f"RMSE={test_stats['RMSE_fF']:.1f} fF"
            )


# ============================================================
# 10. RESULTS SUMMARY
# ============================================================

score_df = pd.DataFrame(scores)
param_df = pd.DataFrame(parameter_records)

summary = (
    score_df.groupby(["Split", "Model"])
    .agg(
        Train_RMSE_mean=("Train_RMSE_fF", "mean"),
        Test_RMSE_mean=("RMSE_fF", "mean"),
        Test_RMSE_std=("RMSE_fF", "std"),
        Test_MAE_mean=("MAE_fF", "mean"),
        Test_MedianAE_mean=("MedianAE_fF", "mean"),
        Test_P95_mean=("P95AE_fF", "mean"),
        Test_Bias_mean=("Bias_fF", "mean")
    )
    .reset_index()
)

print("\n" + "="*85)
print("AGGREGATION MODEL COMPARISON")
print("="*85)
print(
    summary.round(2).to_string(index=False)
)

print("\n" + "="*85)
print("PARAMETER DIAGNOSTICS")
print("="*85)

param_summary = (
    param_df.groupby(["Split", "Model"])
    .agg(
        PinRatio_mean=("PinMaxMinRatio", "mean"),
        PinRatio_std=("PinMaxMinRatio", "std"),
        Shared_mean=("Shared_pF", "mean"),
        Shared_std=("Shared_pF", "std")
    )
    .reset_index()
)

print(
    param_summary.round(3).to_string(index=False)
)


# ============================================================
# 11. BEST MODEL SELECTION
#
# Select based on mean UnseenSignature RMSE.
# Then refit on the full measurement dataset.
# ============================================================

unseen_summary = summary[
    summary["Split"] == "UnseenSignature"
]

best_name = (
    unseen_summary.sort_values(
        "Test_RMSE_mean"
    ).iloc[0]["Model"]
)

print("\nBEST UNSEEN-SIGNATURE MODEL:", best_name)

full_train = np.arange(len(data))

final_models = {}

for model in MODELS:
    final_models[model] = fit_model(
        model, full_train
    )

# ============================================================
# 12. PIN PARAMETER REPORT
# ============================================================

pin_report = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "SMU": [
        PIN_SMU[p] + 1 for p in range(1, 45)
    ]
})

for model in MODELS:
    c_pin, c_shared = decode_theta(
        final_models[model],
        model
    )

    pin_report[f"{model}_C_pF"] = c_pin

    # Effective capacitive impedance magnitude
    # at 10kHz: |Z| = 1 / (omega * C)
    pin_report[f"{model}_Z_MOhm"] = (
        1 / (OMEGA * c_pin * 1e-12) / 1e6
    )

print("\n" + "="*85)
print("PIN PARAMETER REPORT")
print("="*85)
print(
    pin_report.round(3).to_string(index=False)
)

# Shared-path parameter
shared_c = np.exp(
    final_models["SharedPath"][44]
)

print(
    "\nSharedPath effective C_shared:",
    round(shared_c, 5), "pF"
)


# ============================================================
# 13. INDEPENDENT ABSOLUTE PREDICTION
#
# No measured baseline.
# No comparison combination.
# ============================================================

def predict_capacitance(high, low, model=None):
    if model is None:
        model = best_name

    if model not in final_models:
        raise ValueError(f"Unknown model: {model}")

    h, l, local_pf, _, _ = (
        extract_features(high, low)
    )

    pred_pf = predict_c(
        final_models[model],
        model,
        h[None, :],
        l[None, :],
        np.array([local_pf])
    )[0]

    y_imag_S = OMEGA * pred_pf * 1e-12

    return {
        "Model": model,
        "C_pF": float(pred_pf),
        "ImagY_S": float(y_imag_S)
    }


# ============================================================
# 14. EXAMPLE: ALL THREE MODELS
# ============================================================

example_high = list(range(15, 45))
example_low = list(range(7, 14))

print("\nINDEPENDENT PREDICTION COMPARISON")

for model in MODELS:
    result = predict_capacitance(
        example_high,
        example_low,
        model=model
    )

    print(
        f"{model:11} "
        f"C={result['C_pF']:.5f} pF, "
        f"ImagY={result['ImagY_S']:.6e} S"
    )


# ============================================================
# 15. SAVE
# ============================================================

joblib.dump(
    {
        "models": final_models,
        "best_model": best_name,
        "frequency_hz": FREQ,
        "fixed_local_pf": LOCAL_COEFF
    },
    "aggregation_comparison.joblib"
)

score_df.to_csv(
    "aggregation_cv_results.csv",
    index=False
)

param_df.to_csv(
    "aggregation_parameter_diagnostics.csv",
    index=False
)

pin_report.to_csv(
    "aggregation_pin_parameters.csv",
    index=False
)
