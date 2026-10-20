
import ast
import re
import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ==================================================
# SETTINGS
# ==================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

N_FOLDS = 5
TEST_SIZE = 0.25
SEED = 42

# All C values handled in pF.
# Input df["C"] is in Farads.
PAIR_NAMES = ["Even", "Odd", "SameOther", "Different"]

MODES = ["Parallel", "Mean", "SharedPath"]

SMU_RANGES = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g
    for g, pins in enumerate(SMU_RANGES)
    for p in pins
}

# Regularization scales (pF).
# The residual is divided by these scales.
REG_MULTI = 0.20
REG_AGG = 0.10

# Parameter bounds
MIN_C = 0.001
MAX_C = 100.0

# ==================================================
# 1. PIN PARSING
# ==================================================

def parse_pins(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        pins = [int(v) for v in value]
    elif isinstance(value, str):
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, (list, tuple, set)):
                pins = [int(v) for v in obj]
            else:
                pins = [
                    int(v) for v in re.findall(r"\d+", value)
                ]
        except (ValueError, SyntaxError):
            pins = [
                int(v) for v in re.findall(r"\d+", value)
            ]
    else:
        raise ValueError(value)

    if not pins or len(set(pins)) != len(pins):
        raise ValueError(f"Invalid pin list: {value}")
    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pin outside 1..44")

    return tuple(sorted(pins))


# ==================================================
# 2. FOUR SINGLE-PAIR CATEGORIES
#
# Even / Odd apply only when distance == 1.
# ==================================================

def pair_category(a, b):
    if PIN_SMU[a] != PIN_SMU[b]:
        return 3  # Different SMU

    if abs(a - b) == 1:
        return 0 if max(a, b) % 2 == 0 else 1

    return 2  # Other within the same SMU


def make_features(high, low):
    high = parse_pins(high)
    low = parse_pins(low)

    if set(high) & set(low):
        raise ValueError("Overlapping High/Low pins")

    nh, nl = len(high), len(low)
    counts = np.zeros(4)

    for a in high:
        for b in low:
            counts[pair_category(a, b)] += 1

    # Pair category composition. Sums to 1.
    pair_fraction = counts / (nh * nl)

    hc = np.array([
        sum(p in high for p in group)
        for group in SMU_RANGES
    ])
    lc = np.array([
        sum(p in low for p in group)
        for group in SMU_RANGES
    ])

    signature = tuple(
        int(v) for v in np.r_[hc, lc]
    )

    configuration = (high, low)

    # Multi-pin features vanish for 1H / 1L.
    #
    # nH/nL growth, joint growth,
    # and distribution across multiple SMUs.
    multi = np.array([
        np.log(nh),
        np.log(nl),
        np.log(nh) * np.log(nl),
        (np.count_nonzero(hc) - 1)
        * (np.count_nonzero(lc) - 1)
    ], dtype=float)

    return (
        pair_fraction, nh, nl, multi,
        signature, configuration
    )


# ==================================================
# 3. PREPARE DATA
# ==================================================

data = df.copy().reset_index(drop=True)

required = ["high_pins", "low_pins", "C"]
for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing: {col}")

records = [
    make_features(h, l)
    for h, l in zip(data["high_pins"], data["low_pins"])
]

P = np.vstack([r[0] for r in records])
NH = np.array([r[1] for r in records], dtype=float)
NL = np.array([r[2] for r in records], dtype=float)
MULTI = np.vstack([r[3] for r in records])

SIG = np.array([str(r[4]) for r in records])
CONFIG = np.array([str(r[5]) for r in records])

Y = pd.to_numeric(data["C"]).to_numpy(dtype=float) * 1e12

if not np.isfinite(Y).all():
    raise ValueError("Non-finite C data")

single_mask = (NH == 1) & (NL == 1)

print("Measurements:", len(Y))
print("Unique configurations:", len(set(CONFIG)))
print("Unique signatures:", len(set(SIG)))
print("Single-pin measurements:", single_mask.sum())

print("\nSingle-pair category coverage:")
for k, name in enumerate(PAIR_NAMES):
    mask = single_mask & (P[:, k] > 0.5)
    print(
        f"{name:<12}: n={mask.sum():4d}, "
        f"mean={Y[mask].mean():.4f} pF"
        if np.any(mask) else
        f"{name:<12}: n=0 (not directly anchored)"
    )


# ==================================================
# 4. AGGREGATION
#
# All pins have identical effective c.
#
# SharedPath:
#   per-side Ce = (n*c)*Cs / (n*c+Cs)
#
# Two sides are combined in series.
#
# For each model subtract the single-pair value.
# ==================================================

def aggregate_side(n, c, cs, mode):
    if mode == "Parallel":
        return n * c

    if mode == "Mean":
        return np.full_like(n, c, dtype=float)

    if mode == "SharedPath":
        total = n * c
        return total * cs / (total + cs)

    raise ValueError(mode)


def aggregation_delta(nh, nl, c, cs, mode):
    ch = aggregate_side(nh, c, cs, mode)
    cl = aggregate_side(nl, c, cs, mode)

    full = ch * cl / (ch + cl)

    one = np.array([1.0])
    ch1 = aggregate_side(one, c, cs, mode)[0]
    cl1 = aggregate_side(one, c, cs, mode)[0]

    single = ch1 * cl1 / (ch1 + cl1)

    return full - single


# ==================================================
# 5. MODEL PARAMETERS
#
# theta:
#   [0:4]    pair category values (pF)
#   [4]      log(c), common pin coefficient
#   [5]      log(Cs), SharedPath only
#   [last 4] multi-pin correction coefficients
#
# Mean aggregation has zero delta by construction,
# so c is fixed and not fitted.
# ==================================================

def decode(theta, mode, add_multi):
    pair_c = theta[:4]

    pos = 4
    if mode == "Mean":
        c = 1.0
        cs = None
    else:
        c = np.exp(theta[pos])
        pos += 1

        if mode == "SharedPath":
            cs = np.exp(theta[pos])
            pos += 1
        else:
            cs = None

    if add_multi:
        beta = theta[pos:pos+4]
    else:
        beta = np.zeros(4)

    return pair_c, c, cs, beta


def make_initial(mode, add_multi, train):
    # Initial pair coefficients from single-pair measurements.
    # Unobserved categories start at global data median.
    initial_pairs = np.full(4, np.median(Y[train]))

    for k in range(4):
        idx = train[
            single_mask[train] & (P[train, k] > 0.5)
        ]
        if len(idx):
            initial_pairs[k] = np.median(Y[idx])

    parts = [initial_pairs]

    if mode != "Mean":
        parts.append(np.array([np.log(1.0)]))

        if mode == "SharedPath":
            parts.append(np.array([np.log(10.0)]))

    if add_multi:
        parts.append(np.zeros(4))

    return np.concatenate(parts)


def predict(theta, mode, add_multi, indices=None):
    if indices is None:
        indices = np.arange(len(Y))

    pair_c, c, cs, beta = decode(
        theta, mode, add_multi
    )

    pair_base = P[indices] @ pair_c

    delta_agg = aggregation_delta(
        NH[indices], NL[indices],
        c, cs, mode
    )

    delta_multi = MULTI[indices] @ beta

    return pair_base + delta_agg + delta_multi


# ==================================================
# 6. FIT
# ==================================================

def fit_model(mode, add_multi, train):
    theta0 = make_initial(mode, add_multi, train)

    lower = np.full(len(theta0), -np.inf)
    upper = np.full(len(theta0), np.inf)

    # Effective pair values constrained positive
    lower[:4] = 0
    upper[:4] = MAX_C

    pos = 4
    if mode != "Mean":
        lower[pos] = np.log(MIN_C)
        upper[pos] = np.log(MAX_C)
        pos += 1

        if mode == "SharedPath":
            lower[pos] = np.log(MIN_C)
            upper[pos] = np.log(1000.0)
            pos += 1

    # Fit all observations, but balance the training
    # objective so overrepresented configurations
    # do not dominate the fit.
    cfg = CONFIG[train]
    _, inverse, counts = np.unique(
        cfg, return_inverse=True, return_counts=True
    )
    weights = 1.0 / np.sqrt(counts[inverse])
    weights *= np.sqrt(len(weights) / np.sum(weights**2))

    def residual(theta):
        pred = predict(theta, mode, add_multi, train)
        data_error = (pred - Y[train]) * weights

        reg = []

        # Optional multi-pin correction regularization.
        if add_multi:
            _, _, _, beta = decode(
                theta, mode, add_multi
            )
            reg.extend(REG_MULTI * beta)

        # Mild preference against unbounded c.
        # Not applied to Mean because it cancels.
        if mode != "Mean":
            _, c, _, _ = decode(
                theta, mode, add_multi
            )
            reg.append(
                REG_AGG * np.log(c)
            )

        return np.r_[data_error, reg]

    result = least_squares(
        residual,
        theta0,
        bounds=(lower, upper),
        max_nfev=2500,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10
    )

    if not result.success:
        print(
            "Fit warning:",
            mode, add_multi, result.message
        )

    return result.x


# ==================================================
# 7. CROSS VALIDATION
# ==================================================

variants = [
    (mode, add_multi)
    for mode in MODES
    for add_multi in [False, True]
]

def metric(actual, pred):
    e = (pred - actual) * 1000
    return {
        "RMSE_fF": np.sqrt(np.mean(e**2)),
        "MAE_fF": np.mean(np.abs(e)),
        "P95_fF": np.percentile(np.abs(e), 95),
        "Bias_fF": np.mean(e)
    }


scores = []
parameters = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIG)
]:
    cv = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        cv.split(P, groups=groups),
        start=1
    ):
        for mode, add_multi in variants:
            theta = fit_model(mode, add_multi, train)

            prediction = predict(
                theta, mode, add_multi, test
            )

            stats = metric(Y[test], prediction)

            name = mode + ("+Multi" if add_multi else "")

            scores.append({
                "Split": split_name,
                "Fold": fold,
                "Model": name,
                **stats
            })

            pair_c, c, cs, beta = decode(
                theta, mode, add_multi
            )

            parameters.append({
                "Split": split_name,
                "Fold": fold,
                "Model": name,
                "C_E": pair_c[0],
                "C_O": pair_c[1],
                "C_S": pair_c[2],
                "C_D": pair_c[3],
                "C_pin": c,
                "C_shared": cs,
                "beta_1": beta[0],
                "beta_2": beta[1],
                "beta_3": beta[2],
                "beta_4": beta[3]
            })

            print(
                f"{split_name:16} "
                f"fold={fold} "
                f"{name:18} "
                f"RMSE={stats['RMSE_fF']:.1f} fF"
            )


# ==================================================
# 8. RESULTS
# ==================================================

score_df = pd.DataFrame(scores)
param_df = pd.DataFrame(parameters)

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

print("\n=== MODEL COMPARISON ===")
print(summary.round(2).to_string(index=False))

print("\n=== PARAMETER STABILITY ===")

parameter_summary = (
    param_df.groupby(["Split", "Model"])
    .agg({
        "C_E": ["mean", "std"],
        "C_O": ["mean", "std"],
        "C_S": ["mean", "std"],
        "C_D": ["mean", "std"],
        "C_pin": ["mean", "std"],
        "C_shared": ["mean", "std"]
    })
)

print(parameter_summary.round(4).to_string())


# ==================================================
# 9. FIT FINAL MODELS
# ==================================================

all_idx = np.arange(len(Y))
final_models = {}

for mode, add_multi in variants:
    name = mode + ("+Multi" if add_multi else "")
    final_models[name] = fit_model(
        mode, add_multi, all_idx
    )

print("\n=== FULL DATA PAIR COEFFICIENTS ===")

for name, theta in final_models.items():
    mode = name.split("+")[0]
    add_multi = "+Multi" in name

    pair_c, c, cs, beta = decode(
        theta, mode, add_multi
    )

    print(
        name,
        "Pair=", np.round(pair_c, 4),
        "Pin=", round(c, 4),
        "Shared=", (
            round(cs, 4) if cs is not None else None
        ),
        "Multi=", np.round(beta, 4)
    )


# ==================================================
# 10. INDEPENDENT PREDICTION
# ==================================================

def predict_new(high, low, model_name):
    # Temporarily construct one-row feature representation
    # without requiring any reference measurement.
    record = make_features(high, low)

    pair_fraction, nh, nl, multi, _, _ = record

    theta = final_models[model_name]
    mode = model_name.split("+")[0]
    add_multi = "+Multi" in model_name

    pair_c, c, cs, beta = decode(
        theta, mode, add_multi
    )

    pair_base = pair_fraction @ pair_c

    delta_agg = aggregation_delta(
        np.array([float(nh)]),
        np.array([float(nl)]),
        c, cs, mode
    )[0]

    delta_multi = multi @ beta

    c_pf = pair_base + delta_agg + delta_multi

    return {
        "C_pF": float(c_pf),
        "Pair_pF": float(pair_base),
        "Aggregation_pF": float(delta_agg),
        "Multi_pF": float(delta_multi),
        "ImagY_S": float(OMEGA * c_pf * 1e-12)
    }


print("\n=== INDEPENDENT EXAMPLE ===")

for name in final_models:
    out = predict_new(
        high=list(range(15, 45)),
        low=list(range(7, 14)),
        model_name=name
    )
    print(
        f"{name:20} "
        f"C={out['C_pF']:.4f} pF  "
        f"Pair={out['Pair_pF']:.3f} "
        f"Agg={out['Aggregation_pF']:.3f} "
        f"Multi={out['Multi_pF']:.3f}"
    )

score_df.to_csv("pair_aggregation_scores.csv", index=False)
param_df.to_csv("pair_aggregation_parameters.csv", index=False)
