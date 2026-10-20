
import ast
import re
import numpy as np
import pandas as pd
import joblib

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# =========================================================
# 0. CONFIG
# =========================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

N_FOLDS = 5
TEST_SIZE = 0.25
SEED = 42

AGGREGATIONS = ["Sum", "MaxPerPin", "Sqrt", "PartialSum"]

# Fixed empirical zone coefficients [pF].
# [same-SMU adjacent even, adjacent odd,
#  same-SMU distance 2, cross-SMU]
LOCAL_COEFF_PF = np.array([1.544, 0.605, 0.175])

# New cross-SMU coefficient fitted in Stage 1.
CROSS_BOUNDS_PF = (0.0, 1.0)

# Common effective pin capacitance [pF].
COMMON_BOUNDS_PF = (0.01, 50.0)

# Pin deviations fitted only in Stage 3.
PIN_LOG_SIGMA = 0.20
PIN_REG_PF = 0.35

SMU_RANGES = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g for g, grp in enumerate(SMU_RANGES)
    for p in grp
}

# =========================================================
# 1. PARSE PINS
# =========================================================

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
                    int(x) for x in re.findall(r"\d+", value)
                ]
        except (ValueError, SyntaxError):
            pins = [
                int(x) for x in re.findall(r"\d+", value)
            ]
    else:
        raise ValueError(f"Invalid pin list: {value}")

    if not pins or len(set(pins)) != len(pins):
        raise ValueError(f"Empty/duplicate pins: {value}")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError(f"Invalid pin: {value}")

    return tuple(sorted(pins))


# =========================================================
# 2. BUILD ZONE MATRICES
#
# Zone 0: same SMU, adjacent, larger pin even
# Zone 1: same SMU, adjacent, larger pin odd
# Zone 2: same SMU, distance 2
# Zone 3: different SMU
#
# Same-SMU distance >= 3: zero local increment.
#
# Symmetric zone matrices.
# =========================================================

ZONE = np.zeros((4, 44, 44), dtype=float)

for a in range(1, 45):
    for b in range(1, 45):
        if a == b:
            continue

        if PIN_SMU[a] != PIN_SMU[b]:
            k = 3

        elif abs(a - b) == 1:
            k = 0 if max(a, b) % 2 == 0 else 1

        elif abs(a - b) == 2:
            k = 2

        else:
            continue

        ZONE[k, a-1, b-1] = 1.0


# =========================================================
# 3. DATA PREPARATION
# =========================================================

data = df.copy().reset_index(drop=True)

for col in ["high_pins", "low_pins", "C"]:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

high_keys = data["high_pins"].apply(parse_pins)
low_keys = data["low_pins"].apply(parse_pins)

H = np.zeros((len(data), 44))
L = np.zeros((len(data), 44))

for k, (high, low) in enumerate(zip(high_keys, low_keys)):
    if set(high) & set(low):
        raise ValueError(f"High/Low overlap in row {k}")

    H[k, np.array(high) - 1] = 1.0
    L[k, np.array(low) - 1] = 1.0

# Measurements: F -> pF
C_TRUE = pd.to_numeric(
    data["C"]
).to_numpy(dtype=float) * 1e12

if not np.isfinite(C_TRUE).all():
    raise ValueError("Non-finite C values")

NH = H.sum(axis=1)
NL = L.sum(axis=1)

# Counts by SMU
SMU_MATRIX = np.zeros((44, 6))
for p in range(1, 45):
    SMU_MATRIX[p-1, PIN_SMU[p]] = 1

HC = H @ SMU_MATRIX
LC = L @ SMU_MATRIX

SIGNATURE = np.array([
    str(tuple(np.r_[HC[k], LC[k]].astype(int)))
    for k in range(len(data))
])

CONFIG = np.array([
    str((high_keys.iloc[k], low_keys.iloc[k]))
    for k in range(len(data))
])

# =========================================================
# 4. PRECOMPUTE PAIR ZONE COUNTS
#
# h_zone[n,i,k] = number of Low pins of
# zone k connected to High pin i.
#
# l_zone[n,j,k] = number of High pins of
# zone k connected to Low pin j.
#
# Both are zero for unselected pins.
# =========================================================

h_zone = np.zeros((len(data), 44, 4))
l_zone = np.zeros((len(data), 44, 4))

for k in range(4):
    mask = ZONE[k]

    h_zone[:, :, k] = (
        H * (L @ mask.T)
    )

    l_zone[:, :, k] = (
        L * (H @ mask)
    )

# Total number of H-L pairs in each zone
zone_counts = h_zone.sum(axis=1)

print("Measurements:", len(data))
print("Configurations:", len(set(CONFIG)))
print("Signatures:", len(set(SIGNATURE)))
print("Total zone counts:", zone_counts.sum(axis=0))

# =========================================================
# 5. PAIR AGGREGATION
#
# Single pair always contributes its zone coefficient.
#
# Sum:
#   All pair contributions.
#
# MaxPerPin:
#   For each H pin, maximum coefficient among
#   its connected L pins.
#   Same for L pins.
#   Average both totals.
#
# Sqrt:
#   sqrt(Sum * MaxPerPin)
#
# PartialSum:
#   MaxPerPin + 0.25*(Sum-MaxPerPin)
# =========================================================

def pair_aggregation(cross_pf, indices, mode):
    coeff = np.r_[LOCAL_COEFF_PF, cross_pf]

    counts = zone_counts[indices]
    sum_contrib = counts @ coeff

    # Per-pin maximum category coefficient.
    # A Pin with no opposite-polarity neighbor
    # in a nonzero zone contributes zero.
    h_scores = np.where(
        h_zone[indices] > 0,
        coeff[None, None, :],
        0.0
    )

    l_scores = np.where(
        l_zone[indices] > 0,
        coeff[None, None, :],
        0.0
    )

    max_h = h_scores.max(axis=2).sum(axis=1)
    max_l = l_scores.max(axis=2).sum(axis=1)

    max_contrib = 0.5 * (max_h + max_l)

    if mode == "Sum":
        return sum_contrib

    if mode == "MaxPerPin":
        return max_contrib

    if mode == "Sqrt":
        return np.sqrt(
            np.maximum(
                sum_contrib * max_contrib, 0
            )
        )

    if mode == "PartialSum":
        return (
            max_contrib
            + 0.25 * (sum_contrib - max_contrib)
        )

    raise ValueError(mode)


# =========================================================
# 6. PIN PATH
#
# Stage 1:
#   All pins have c_common.
#
# Stage 3:
#   c_pin[i] = c_common * exp(v_i - mean(v))
#
# Effective H and L capacitances summed separately,
# then combined in series.
# =========================================================

def pin_path(indices, c_pin):
    ch = H[indices] @ c_pin
    cl = L[indices] @ c_pin

    return np.divide(
        ch * cl,
        ch + cl,
        out=np.zeros_like(ch),
        where=(ch + cl) > 0
    )


def predict_stage1(params, indices, mode):
    c_common, cross_pf = params

    c_pin = np.full(44, c_common)

    return (
        pin_path(indices, c_pin)
        + pair_aggregation(cross_pf, indices, mode)
    )


# =========================================================
# 7. STAGE 1 FIT
#
# Only TWO fitted parameters:
#   1. Common pin c
#   2. Cross-SMU pair coefficient
#
# The three local coefficients are fixed.
# =========================================================

def fit_stage1(train, mode):
    initial = np.array([0.6, 0.02])

    def residual(params):
        return (
            predict_stage1(params, train, mode)
            - C_TRUE[train]
        )

    result = least_squares(
        residual,
        initial,
        bounds=(
            [COMMON_BOUNDS_PF[0], CROSS_BOUNDS_PF[0]],
            [COMMON_BOUNDS_PF[1], CROSS_BOUNDS_PF[1]]
        ),
        max_nfev=1500,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10
    )

    return result.x


# =========================================================
# 8. STAGE 3: FIT PIN DEVIATIONS LAST
#
# c_common and cross_pf are FIXED.
# Only pin-specific deviations v_i are optimized.
#
# Centering v keeps the geometric mean of pin
# coefficients equal to c_common.
# =========================================================

def predict_stage3(v, base_params, indices, mode):
    c_common, cross_pf = base_params

    v_centered = v - v.mean()
    c_pin = c_common * np.exp(v_centered)

    return (
        pin_path(indices, c_pin)
        + pair_aggregation(cross_pf, indices, mode)
    )


def fit_stage3(train, mode, base_params):
    def residual(v):
        data_error = (
            predict_stage3(
                v, base_params, train, mode
            )
            - C_TRUE[train]
        )

        # Regularize pin deviations.
        # All coefficients remain relative to the
        # previously fitted common pin value.
        centered = v - v.mean()

        reg = (
            PIN_REG_PF
            * centered / PIN_LOG_SIGMA
        )

        return np.r_[data_error, reg]

    result = least_squares(
        residual,
        np.zeros(44),
        bounds=(
            np.full(44, -2.0),
            np.full(44, 2.0)
        ),
        max_nfev=500,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9
    )

    return result.x - result.x.mean()


# =========================================================
# 9. METRICS
# =========================================================

def metrics(actual, predicted):
    e = (predicted - actual) * 1000

    return {
        "RMSE_fF": np.sqrt(np.mean(e**2)),
        "MAE_fF": np.mean(np.abs(e)),
        "P95_fF": np.percentile(np.abs(e), 95),
        "Bias_fF": np.mean(e)
    }


# =========================================================
# 10. CROSS VALIDATION: STAGE 1
#
# Both methods:
#  - grouped by exact H/L configuration
#  - grouped by unseen occupancy signature
#
# Every aggregation uses identical folds.
# =========================================================

folds = {}
rows = []
param_rows = []

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
        folds[(split_type, fold)] = (train, test)

        for mode in AGGREGATIONS:
            params = fit_stage1(train, mode)

            pred_train = predict_stage1(
                params, train, mode
            )
            pred_test = predict_stage1(
                params, test, mode
            )

            train_m = metrics(
                C_TRUE[train], pred_train
            )
            test_m = metrics(
                C_TRUE[test], pred_test
            )

            rows.append({
                "Split": split_type,
                "Fold": fold,
                "Model": mode,
                "Train_RMSE_fF": train_m["RMSE_fF"],
                **test_m
            })

            param_rows.append({
                "Split": split_type,
                "Fold": fold,
                "Model": mode,
                "Common_C_pF": params[0],
                "Cross_C_pF": params[1]
            })


score_df = pd.DataFrame(rows)
param_df = pd.DataFrame(param_rows)

summary = (
    score_df.groupby(["Split", "Model"])
    .agg(
        RMSE_mean=("RMSE_fF", "mean"),
        RMSE_std=("RMSE_fF", "std"),
        MAE_mean=("MAE_fF", "mean"),
        P95_mean=("P95_fF", "mean"),
        Bias_mean=("Bias_fF", "mean")
    )
    .reset_index()
)

print("\n=== STAGE 1 AGGREGATION COMPARISON ===")
print(summary.round(2).to_string(index=False))

print("\n=== STAGE 1 PARAMETER STABILITY ===")
print(
    param_df.groupby(["Split", "Model"])[
        ["Common_C_pF", "Cross_C_pF"]
    ].agg(["mean", "std"]).round(5).to_string()
)


# =========================================================
# 11. SELECT AGGREGATION
#
# Lowest average UnseenSignature RMSE.
# Do not use test data during individual fits.
# =========================================================

unseen = summary[
    summary["Split"] == "UnseenSignature"
]

best_mode = (
    unseen.sort_values("RMSE_mean")
    .iloc[0]["Model"]
)

print("\nSelected aggregation:", best_mode)

# =========================================================
# 12. STAGE 3 VALIDATION
#
# Evaluate selected aggregation with vs without
# pin-specific deviations on SAME folds.
#
# Selection using Unseen results makes this comparison
# exploratory, not a fully nested unbiased CV estimate.
# =========================================================

pin_rows = []
pin_params = []

for (split_type, fold), (train, test) in folds.items():
    base_params = fit_stage1(train, best_mode)

    v = fit_stage3(
        train, best_mode, base_params
    )

    pred_base = predict_stage1(
        base_params, test, best_mode
    )

    pred_pin = predict_stage3(
        v, base_params, test, best_mode
    )

    for label, pred in [
        ("CommonPin", pred_base),
        ("IndividualPin", pred_pin)
    ]:
        pin_rows.append({
            "Split": split_type,
            "Fold": fold,
            "Model": label,
            **metrics(C_TRUE[test], pred)
        })

    c_pin = base_params[0] * np.exp(v)

    pin_params.append({
        "Split": split_type,
        "Fold": fold,
        "Min_C_pF": c_pin.min(),
        "Median_C_pF": np.median(c_pin),
        "Max_C_pF": c_pin.max(),
        "MaxMinRatio": c_pin.max() / c_pin.min(),
        "Pin7_C_pF": c_pin[6]
    })


pin_score_df = pd.DataFrame(pin_rows)
pin_param_df = pd.DataFrame(pin_params)

pin_summary = (
    pin_score_df.groupby(["Split", "Model"])
    .agg(
        RMSE_mean=("RMSE_fF", "mean"),
        RMSE_std=("RMSE_fF", "std"),
        P95_mean=("P95_fF", "mean")
    )
    .reset_index()
)

print("\n=== STAGE 3 PIN-DEVIATION VALIDATION ===")
print(pin_summary.round(2).to_string(index=False))

print("\n=== PIN STABILITY ===")
print(
    pin_param_df.groupby("Split")[
        ["MaxMinRatio", "Pin7_C_pF"]
    ].agg(["mean", "std"]).round(4).to_string()
)


# =========================================================
# 13. FINAL FIT
# =========================================================

all_idx = np.arange(len(data))

final_base = fit_stage1(all_idx, best_mode)
final_v = fit_stage3(
    all_idx, best_mode, final_base
)

final_c_pin = (
    final_base[0] * np.exp(final_v)
)

pin_table = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "SMU": [
        PIN_SMU[p] + 1
        for p in range(1, 45)
    ],
    "Effective_C_pF": final_c_pin,
    "AbsZ_MOhm": (
        1.0 / (
            OMEGA * final_c_pin * 1e-12
        ) / 1e6
    )
})

print("\n=== FINAL FIT ===")
print("Aggregation:", best_mode)
print("Common Pin C [pF]:", final_base[0])
print("Cross-SMU coefficient [pF]:", final_base[1])

print("\n=== PIN EFFECTIVE IMPEDANCES ===")
print(pin_table.round(4).to_string(index=False))


# =========================================================
# 14. INDEPENDENT PREDICTION
#
# No reference measurement required.
# Choose use_pin_deviation=False for Stage 1.
# =========================================================

def predict_new(high, low, use_pin_deviation=False):
    high = parse_pins(high)
    low = parse_pins(low)

    if set(high) & set(low):
        raise ValueError("High/Low overlap")

    h = np.zeros(44)
    l = np.zeros(44)

    h[np.array(high)-1] = 1
    l[np.array(low)-1] = 1

    # Single-sample zone count matrices.
    hz = np.zeros((1, 44, 4))
    lz = np.zeros((1, 44, 4))

    for k in range(4):
        hz[0, :, k] = h * (ZONE[k] @ l)
        lz[0, :, k] = l * (ZONE[k] @ h)

    coeff = np.r_[LOCAL_COEFF_PF, final_base[1]]

    sum_value = np.sum(hz * coeff[None, None, :])

    max_h = np.where(
        hz > 0, coeff[None, None, :], 0
    ).max(axis=2).sum()

    max_l = np.where(
        lz > 0, coeff[None, None, :], 0
    ).max(axis=2).sum()

    max_value = 0.5 * (max_h + max_l)

    if best_mode == "Sum":
        c_pair = sum_value
    elif best_mode == "MaxPerPin":
        c_pair = max_value
    elif best_mode == "Sqrt":
        c_pair = np.sqrt(sum_value * max_value)
    else:
        c_pair = max_value + 0.25 * (
            sum_value - max_value
        )

    if use_pin_deviation:
        c_pin = final_c_pin
    else:
        c_pin = np.full(44, final_base[0])

    ch = h @ c_pin
    cl = l @ c_pin

    c_line = ch * cl / (ch + cl)

    c_total = c_line + c_pair

    return {
        "C_pF": float(c_total),
        "C_line_pF": float(c_line),
        "C_pair_pF": float(c_pair),
        "ImagY_S": float(
            OMEGA * c_total * 1e-12
        )
    }


# =========================================================
# 15. EXAMPLE
# =========================================================

high = list(range(15, 45))
low = list(range(7, 14))

print("\nCOMMON PIN PREDICTION")
print(predict_new(high, low, False))

print("\nINDIVIDUAL PIN PREDICTION")
print(predict_new(high, low, True))


# =========================================================
# 16. SAVE
# =========================================================

joblib.dump({
    "mode": best_mode,
    "base_params": final_base,
    "pin_deviations": final_v,
    "local_coefficients_pf": LOCAL_COEFF_PF,
    "frequency_hz": FREQ
}, "pair_aggregation_staged.joblib")

score_df.to_csv("pair_aggregation_stage1_cv.csv", index=False)
param_df.to_csv("pair_aggregation_stage1_params.csv", index=False)
pin_score_df.to_csv("pair_aggregation_stage3_cv.csv", index=False)
pin_table.to_csv("pair_aggregation_pin_impedance.csv", index=False)
