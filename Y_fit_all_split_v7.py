
import ast
import re
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit
import joblib

# ============================================================
# 0. SETTINGS
# ============================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

IMP_MODES = ["Parallel", "Mean", "SharedPath"]
PAIR_MODES = ["Sum", "MaxPerPin", "Sqrt", "PartialSum"]

N_FOLDS = 2
TEST_SIZE = 0.25
SEED = 42

# Fixed zone coefficients, pF
# Adjacent even / adjacent odd / distance 2
LOCAL = np.array([1.544, 0.605, 0.175])

# Cross-SMU pair coefficient to be fitted
CROSS_RANGE = (0.0, 2.0)

# Common effective pin capacitance, pF
C_RANGE = (0.001, 100.0)

# Shared-path effective capacitance, pF
SHARED_RANGE = (0.01, 1000.0)

# Stage 2 relative pin deviation regularization
PIN_SIGMA = 0.25
PIN_REG = 0.30

SMUS = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g for g, grp in enumerate(SMUS) for p in grp
}

# ============================================================
# 1. PARSING
# ============================================================

def parse_pins(v):
    if isinstance(v, (tuple, list, np.ndarray)):
        pins = [int(x) for x in v]
    elif isinstance(v, str):
        try:
            obj = ast.literal_eval(v)
            pins = list(obj) if isinstance(
                obj, (list, tuple, set)
            ) else re.findall(r"\d+", v)
            pins = [int(x) for x in pins]
        except (ValueError, SyntaxError):
            pins = [int(x) for x in re.findall(r"\d+", v)]
    else:
        raise ValueError(f"Unsupported pins: {v}")

    if not pins or len(pins) != len(set(pins)):
        raise ValueError(f"Empty or duplicate pins: {v}")
    if not all(1 <= p <= 44 for p in pins):
        raise ValueError(f"Pin out of range: {v}")
    return tuple(sorted(pins))


# ============================================================
# 2. FOUR ZONES
#
# 0: Same SMU, adjacent, larger pin even
# 1: Same SMU, adjacent, larger pin odd
# 2: Same SMU, distance 2
# 3: Different SMU
#
# Same SMU, distance >= 3 has zero pair increment.
# ============================================================

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


# ============================================================
# 3. CONSTRUCT FEATURES
#
# Usable for training and arbitrary independent predictions.
# ============================================================

def build_features(high_list, low_list):
    n = len(high_list)

    h = np.zeros((n, 44))
    l = np.zeros((n, 44))
    configurations = []
    signatures = []

    for k, (hv, lv) in enumerate(zip(high_list, low_list)):
        hp = parse_pins(hv)
        lp = parse_pins(lv)

        if set(hp) & set(lp):
            raise ValueError(f"Overlapping H/L: row {k}")

        h[k, np.asarray(hp)-1] = 1
        l[k, np.asarray(lp)-1] = 1

        hc = [
            sum(p in hp for p in group)
            for group in SMUS
        ]
        lc = [
            sum(p in lp for p in group)
            for group in SMUS
        ]

        configurations.append(str((hp, lp)))
        signatures.append(str(tuple(hc + lc)))

    # Per-pin presence of each crossing pair category.
    # Shape: [measurements, 44 pins, 4 zones]
    h_seen = np.stack([
        (h > 0) & ((l @ ZONE[z].T) > 0)
        for z in range(4)
    ], axis=2)

    l_seen = np.stack([
        (l > 0) & ((h @ ZONE[z]) > 0)
        for z in range(4)
    ], axis=2)

    # Total number of crossing pairs in each category.
    counts = np.stack([
        np.sum(h * (l @ ZONE[z].T), axis=1)
        for z in range(4)
    ], axis=1)

    return {
        "H": h,
        "L": l,
        "h_seen": h_seen,
        "l_seen": l_seen,
        "counts": counts,
        "config": np.array(configurations),
        "signature": np.array(signatures)
    }


data = df.copy().reset_index(drop=True)

for col in ["high_pins", "low_pins", "C"]:
    if col not in data:
        raise ValueError(f"Missing {col}")

X = build_features(
    data["high_pins"],
    data["low_pins"]
)

# Measurements: F -> pF
Y = pd.to_numeric(
    data["C"]
).to_numpy(dtype=float) * 1e12

if not np.isfinite(Y).all():
    raise ValueError("Invalid capacitance data")

print("Measurements:", len(Y))
print("Unique configs:", len(set(X["config"])))
print("Unique signatures:", len(set(X["signature"])))
print("Zone pair counts:", X["counts"].sum(axis=0))


# ============================================================
# 4. PIN IMPEDANCE AGGREGATION
#
# Pin i:
#    y_i = j*omega*c_i
#
# For pure capacitive fitting, each mode is
# expressed directly as an effective capacitance.
#
# Parallel: sum(c_i)
# Mean:     mean(c_i)
# Shared:   sum(c_i)*Cs/(sum(c_i)+Cs)
#
# After H and L aggregation:
# Cimp = Ch*Cl/(Ch+Cl)
# ============================================================

def aggregate_side(mask, c_pin, imp_mode, cs):
    s = mask @ c_pin

    if imp_mode == "Parallel":
        return s

    if imp_mode == "Mean":
        n = mask.sum(axis=1)
        return np.divide(
            s, n,
            out=np.zeros_like(s),
            where=n > 0
        )

    if imp_mode == "SharedPath":
        return s * cs / (s + cs)

    raise ValueError(imp_mode)


def impedance_c(features, idx, c_pin, imp_mode, cs):
    ch = aggregate_side(
        features["H"][idx], c_pin, imp_mode, cs
    )
    cl = aggregate_side(
        features["L"][idx], c_pin, imp_mode, cs
    )

    return np.divide(
        ch * cl, ch + cl,
        out=np.zeros_like(ch),
        where=(ch + cl) > 0
    )


# ============================================================
# 5. PAIR INCREMENT AGGREGATION
#
# Four candidate functions.
#
# Sum:
#   Sum all crossing pair contributions.
#
# MaxPerPin:
#   Maximum coefficient seen by each selected H pin,
#   plus maximum for each selected L pin,
#   averaged to retain H/L symmetry.
#
# Sqrt:
#   Geometric mean of Sum and MaxPerPin.
#
# PartialSum:
#   MaxPerPin + 0.25*(Sum-MaxPerPin).
# ============================================================

def pair_c(features, idx, cross_pf, pair_mode):
    coef = np.r_[LOCAL, cross_pf]

    s = features["counts"][idx] @ coef

    hs = np.where(
        features["h_seen"][idx],
        coef[None, None, :],
        0.0
    )
    ls = np.where(
        features["l_seen"][idx],
        coef[None, None, :],
        0.0
    )

    m = 0.5 * (
        hs.max(axis=2).sum(axis=1)
        + ls.max(axis=2).sum(axis=1)
    )

    if pair_mode == "Sum":
        return s
    if pair_mode == "MaxPerPin":
        return m
    if pair_mode == "Sqrt":
        return np.sqrt(np.maximum(s*m, 0))
    if pair_mode == "PartialSum":
        return m + 0.25*(s-m)

    raise ValueError(pair_mode)


# ============================================================
# 6. STAGE 1: COMMON PIN IMPEDANCE
#
# Fitted parameters:
#   c_common
#   cross_pair_coefficient
#   c_shared (only for SharedPath)
#
# All individual pin coefficients are identical.
# ============================================================

def decode_common(t, imp_mode):
    common_c = np.exp(t[0])
    cross_pf = t[1]
    shared = (
        np.exp(t[2]) if imp_mode == "SharedPath"
        else None
    )
    return common_c, cross_pf, shared


def predict_common(features, idx, t, imp_mode, pair_mode):
    common_c, cross_pf, shared = decode_common(
        t, imp_mode
    )

    c_pin = np.full(44, common_c)

    return (
        impedance_c(features, idx, c_pin, imp_mode, shared)
        + pair_c(features, idx, cross_pf, pair_mode)
    )


def fit_common(train, imp_mode, pair_mode):
    init = [np.log(0.6), 0.01]
    lb = [np.log(C_RANGE[0]), CROSS_RANGE[0]]
    ub = [np.log(C_RANGE[1]), CROSS_RANGE[1]]

    if imp_mode == "SharedPath":
        init.append(np.log(10.0))
        lb.append(np.log(SHARED_RANGE[0]))
        ub.append(np.log(SHARED_RANGE[1]))

    def residual(t):
        return (
            predict_common(
                X, train, t, imp_mode, pair_mode
            ) - Y[train]
        )

    result = least_squares(
        residual,
        init,
        bounds=(lb, ub),
        max_nfev=600,
        x_scale="jac"
    )

    return result.x


# ============================================================
# 7. STAGE 2: INDIVIDUAL PIN IMPEDANCES
#
# All Stage 1 parameters remain FIXED.
#
# c_i = c_common * exp(v_i - mean(v))
#
# Thus only relative pin deviations are fitted.
# ============================================================

def pin_coefficients(t_common, v, imp_mode):
    common_c, _, _ = decode_common(
        t_common, imp_mode
    )

    centered = v - np.mean(v)

    return common_c * np.exp(centered)


def predict_individual(
    features, idx, t_common, v, imp_mode, pair_mode
):
    _, cross_pf, shared = decode_common(
        t_common, imp_mode
    )

    c_pin = pin_coefficients(
        t_common, v, imp_mode
    )

    return (
        impedance_c(features, idx, c_pin, imp_mode, shared)
        + pair_c(features, idx, cross_pf, pair_mode)
    )


def fit_individual(train, t_common, imp_mode, pair_mode):
    def residual(v):
        pred = predict_individual(
            X, train, t_common, v,
            imp_mode, pair_mode
        )

        centered = v - np.mean(v)

        reg = (
            PIN_REG / PIN_SIGMA
        ) * centered

        return np.r_[
            pred - Y[train],
            reg
        ]

    result = least_squares(
        residual,
        np.zeros(44),
        bounds=(-2*np.ones(44), 2*np.ones(44)),
        max_nfev=350,
        x_scale="jac"
    )

    return result.x - np.mean(result.x)


# ============================================================
# 8. METRICS
# ============================================================

def metrics(y, p):
    e = (p-y)*1000  # pF -> fF
    return {
        "RMSE_fF": float(np.sqrt(np.mean(e**2))),
        "MAE_fF": float(np.mean(np.abs(e))),
        "P95_fF": float(np.percentile(np.abs(e), 95)),
        "Bias_fF": float(np.mean(e))
    }


# ============================================================
# 9. CROSS VALIDATION: ALL 24 VARIANTS
#
# 12 aggregation pairs * 2 pin modes.
#
# Two validation methods:
# - RandomConfig: never leak exact configs.
# - UnseenSignature: new SMU occupancy patterns.
# ============================================================

rows = []
param_rows = []

for split_name, group in [
    ("RandomConfig", X["config"]),
    ("UnseenSignature", X["signature"])
]:

    cv = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        cv.split(X["H"], groups=group), start=1
    ):

        for imp_mode in IMP_MODES:
            for pair_mode in PAIR_MODES:

                # ---- Stage 1 ----
                t = fit_common(
                    train, imp_mode, pair_mode
                )

                p0 = predict_common(
                    X, test, t, imp_mode, pair_mode
                )

                m0 = metrics(Y[test], p0)

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Impedance": imp_mode,
                    "Pair": pair_mode,
                    "PinFit": "Common",
                    **m0
                })

                # ---- Stage 2 ----
                v = fit_individual(
                    train, t, imp_mode, pair_mode
                )

                p1 = predict_individual(
                    X, test, t, v, imp_mode, pair_mode
                )

                m1 = metrics(Y[test], p1)

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Impedance": imp_mode,
                    "Pair": pair_mode,
                    "PinFit": "Individual",
                    **m1
                })

                common_c, cross_pf, shared = (
                    decode_common(t, imp_mode)
                )

                c_pin = pin_coefficients(
                    t, v, imp_mode
                )

                param_rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Impedance": imp_mode,
                    "Pair": pair_mode,
                    "CommonC_pF": common_c,
                    "CrossC_pF": cross_pf,
                    "SharedC_pF": shared,
                    "PinMin_pF": c_pin.min(),
                    "PinMax_pF": c_pin.max(),
                    "PinRatio": c_pin.max()/c_pin.min(),
                    "Pin7_pF": c_pin[6]
                })

                print(
                    f"{split_name:<17} "
                    f"F{fold} "
                    f"{imp_mode:<11} "
                    f"{pair_mode:<11} "
                    f"Common={m0['RMSE_fF']:7.1f} "
                    f"Individual={m1['RMSE_fF']:7.1f}"
                )


# ============================================================
# 10. RESULTS SUMMARY
# ============================================================

score_df = pd.DataFrame(rows)
param_df = pd.DataFrame(param_rows)

summary = (
    score_df.groupby(
        ["Split", "Impedance", "Pair", "PinFit"]
    )
    .agg(
        RMSE_mean=("RMSE_fF", "mean"),
        RMSE_std=("RMSE_fF", "std"),
        MAE_mean=("MAE_fF", "mean"),
        P95_mean=("P95_fF", "mean"),
        Bias_mean=("Bias_fF", "mean")
    )
    .reset_index()
)

print("\n" + "="*90)
print("ALL AGGREGATION COMBINATIONS")
print("="*90)

print(
    summary.round(2).to_string(index=False)
)


# ============================================================
# 11. PERFORMANCE PIVOT TABLES
# ============================================================

for split_name in ["RandomConfig", "UnseenSignature"]:
    for pinfit in ["Common", "Individual"]:
        sub = summary[
            (summary["Split"] == split_name)
            & (summary["PinFit"] == pinfit)
        ]

        pivot = sub.pivot(
            index="Impedance",
            columns="Pair",
            values="RMSE_mean"
        )

        print(
            f"\n{split_name} | {pinfit} | RMSE fF"
        )
        print(pivot.round(1).to_string())


# ============================================================
# 12. PIN-FITTING BENEFIT
# ============================================================

gain_table = summary.pivot_table(
    index=["Split", "Impedance", "Pair"],
    columns="PinFit",
    values="RMSE_mean"
).reset_index()

gain_table["Gain_fF"] = (
    gain_table["Common"]
    - gain_table["Individual"]
)

print("\n=== PIN FITTING BENEFIT ===")
print(
    gain_table.sort_values(
        ["Split", "Gain_fF"],
        ascending=[True, False]
    ).round(2).to_string(index=False)
)

print("\n=== PARAMETER STABILITY ===")
print(
    param_df.groupby(["Split", "Impedance", "Pair"])[
        ["CommonC_pF", "CrossC_pF", "SharedC_pF",
         "PinRatio", "Pin7_pF"]
    ].agg(["mean", "std"]).round(3).to_string()
)


# ============================================================
# 13. SELECT MODEL
#
# Best UnseenSignature mean RMSE.
# Selection on these folds is exploratory.
# ============================================================

unseen = summary[
    summary["Split"] == "UnseenSignature"
]

best = unseen.sort_values("RMSE_mean").iloc[0]

BEST_IMP = best["Impedance"]
BEST_PAIR = best["Pair"]
BEST_PINFIT = best["PinFit"]

print("\n=== SELECTED MODEL ===")
print("Impedance :", BEST_IMP)
print("Pair      :", BEST_PAIR)
print("PinFit    :", BEST_PINFIT)
print("RMSE      :", best["RMSE_mean"], "fF")


# ============================================================
# 14. FINAL FIT ON ALL MEASUREMENTS
# ============================================================

all_idx = np.arange(len(Y))
final_common = fit_common(
    all_idx, BEST_IMP, BEST_PAIR
)

final_v = (
    fit_individual(
        all_idx, final_common,
        BEST_IMP, BEST_PAIR
    )
    if BEST_PINFIT == "Individual"
    else np.zeros(44)
)

final_c_pin = pin_coefficients(
    final_common, final_v, BEST_IMP
)

pin_table = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "SMU": [PIN_SMU[p]+1 for p in range(1, 45)],
    "EffectiveC_pF": final_c_pin,
    "AbsZ_MOhm": (
        1/(OMEGA*final_c_pin*1e-12)/1e6
    )
})

print("\n=== FINAL PIN PARAMETERS ===")
print(pin_table.round(4).to_string(index=False))


# ============================================================
# 15. INDEPENDENT ABSOLUTE PREDICTION
# ============================================================

def predict_new(high, low,
                impedance=None,
                pair=None,
                pinfit=None):

    impedance = impedance or BEST_IMP
    pair = pair or BEST_PAIR
    pinfit = pinfit or BEST_PINFIT

    if (impedance, pair) != (BEST_IMP, BEST_PAIR):
        raise ValueError(
            "Only selected model was refitted for deployment."
        )

    features = build_features([high], [low])
    idx = np.array([0])

    if pinfit == "Individual":
        c_pred = predict_individual(
            features, idx, final_common,
            final_v, impedance, pair
        )[0]
    elif pinfit == "Common":
        c_pred = predict_common(
            features, idx, final_common,
            impedance, pair
        )[0]
    else:
        raise ValueError(pinfit)

    return {
        "C_pF": float(c_pred),
        "ImagY_S": float(OMEGA*c_pred*1e-12)
    }


print("\n=== EXAMPLE ===")
print(
    predict_new(
        high=list(range(15, 45)),
        low=list(range(7, 14))
    )
)

# ============================================================
# 16. SAVE
# ============================================================

joblib.dump({
    "impedance": BEST_IMP,
    "pair": BEST_PAIR,
    "pinfit": BEST_PINFIT,
    "common_params": final_common,
    "pin_deviations": final_v,
    "local_coefficients_pf": LOCAL,
    "frequency_hz": FREQ
}, "all_aggregation_comparison.joblib")

score_df.to_csv(
    "all_aggregation_scores.csv", index=False
)
param_df.to_csv(
    "all_aggregation_parameters.csv", index=False
)
summary.to_csv(
    "all_aggregation_summary.csv", index=False
)
pin_table.to_csv(
    "all_aggregation_pin_table.csv", index=False
)
