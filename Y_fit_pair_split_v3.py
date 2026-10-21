
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
B_PER_PF = 2 * np.pi * FREQ * 1e-3

# Measured C in F, G unit configurable.
G_UNIT = "S"
G_FACTOR = {"S": 1e9, "uS": 1e3, "nS": 1.0}

C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

N_FOLDS = 3
TEST_SIZE = 0.25
RANDOM_STATE = 42

STRUCTURES = ["A", "B"]
AGG_MODES = ["Sum", "Mean", "MaxPerPin", "Sqrt"]

ZONE_NAMES = [
    "Even",
    "Odd",
    "Dist2",
    "FarSameSMU",
    "CrossSMU"
]

SMUS = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g
    for g, pins in enumerate(SMUS)
    for p in pins
}

# Bounds for effective pair admittance [nS].
# Real part constrained nonnegative; imaginary signed.
G_MAX_NS = 100.0
B_MAX_NS = 100.0

# Shared path impedance [MOhm].
# 1 MOhm * 1 nS = 1e-3 dimensionless.
ZS_REAL_MAX = 50000.0
ZS_IMAG_MAX = 50000.0

# Weak regularization: adjustable after first experiment.
ZS_REG_SCALE = 5000.0
ZS_REG_WEIGHT = 0.03

# Reduce large group model overfitting.
B_GROUP_REG_WEIGHT = 0.1


# ============================================================
# 1. DATA
# ============================================================

def parse_pins(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = re.findall(r"\d+", v)

    pins = tuple(sorted(int(x) for x in v))

    if not pins:
        raise ValueError("Empty pin selection")

    if len(pins) != len(set(pins)):
        raise ValueError("Duplicate pins")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pin outside 1..44")

    return pins


data = df.copy().reset_index(drop=True)

HIGH = [parse_pins(x) for x in data["high_pins"]]
LOW = [parse_pins(x) for x in data["low_pins"]]

for h, l in zip(HIGH, LOW):
    if set(h) & set(l):
        raise ValueError("High and Low overlap")

C_TRUE = data["C"].astype(float).to_numpy() * 1e12
G_TRUE = (
    data["G"].astype(float).to_numpy()
    * G_FACTOR[G_UNIT]
)

N = len(data)

CONFIG = np.array([
    str((h, l)) for h, l in zip(HIGH, LOW)
])

def signature(h, l):
    hc = [sum(p in h for p in s) for s in SMUS]
    lc = [sum(p in l for p in s) for s in SMUS]
    return str(tuple(hc + lc))

SIGNATURE = np.array([
    signature(h, l) for h, l in zip(HIGH, LOW)
])

assert np.isfinite(C_TRUE).all()
assert np.isfinite(G_TRUE).all()


# ============================================================
# 2. CLASSIFY EVERY H-L PAIR
# ============================================================

def pair_zone(h, l):
    if PIN_SMU[h] != PIN_SMU[l]:
        return 4

    d = abs(h - l)

    if d == 1:
        return 0 if max(h, l) % 2 == 0 else 1

    if d == 2:
        return 2

    return 3


# Every observation has A: 1 group,
# and B: up to 6 groups by High-side SMU.
#
# Store compact pair zone matrices for fast reuse.

SAMPLES = []

for h, l in zip(HIGH, LOW):

    all_zone = np.array([
        [pair_zone(hp, lp) for lp in l]
        for hp in h
    ], dtype=np.int8)

    group_zones = []

    for g in range(6):
        rows = [
            k for k, p in enumerate(h)
            if PIN_SMU[p] == g
        ]

        if rows:
            group_zones.append(all_zone[rows, :])
        else:
            group_zones.append(None)

    SAMPLES.append({
        "A": [all_zone],
        "B": group_zones
    })


# ============================================================
# 3. COMPLEX PAIR AGGREGATION
#
# y_zone: five complex admittances [nS]
#
# MaxPerPin:
# Select the highest |Y| on each active
# High and Low pin.
#
# Sqrt:
# Geometric mean of Sum and MaxPerPin magnitudes,
# retaining Sum phase.
# ============================================================

def aggregate_pair(zones, y_zone, mode):
    W = y_zone[zones]  # complex H x L

    S = W.sum()

    if mode == "Sum":
        return S

    if mode == "Mean":
        return S / W.size

    # Every row/column represents a selected pin.
    h_idx = np.argmax(np.abs(W), axis=1)
    l_idx = np.argmax(np.abs(W), axis=0)

    max_h = W[np.arange(W.shape[0]), h_idx].sum()
    max_l = W[l_idx, np.arange(W.shape[1])].sum()

    M = 0.5 * (max_h + max_l)

    if mode == "MaxPerPin":
        return M

    if mode == "Sqrt":
        magnitude = np.sqrt(np.abs(S) * np.abs(M))

        if np.abs(S) > 1e-14:
            phase = S / np.abs(S)
        elif np.abs(M) > 1e-14:
            phase = M / np.abs(M)
        else:
            return 0j

        return magnitude * phase

    raise ValueError(mode)


# ============================================================
# 4. PARAMETERS
#
# t[0:5]  = Pair G [nS]
# t[5:10] = Pair B [nS]
#
# A: one complex shared Z
# B: six complex shared Z
#
# t[10:10+ng] = shared R [MOhm]
# t[10+ng:]    = shared X [MOhm]
# ============================================================

def decode(t, structure):
    ng = 1 if structure == "A" else 6

    y_zone = t[:5] + 1j * t[5:10]

    zr = t[10:10+ng]
    zi = t[10+ng:10+2*ng]

    zs = zr + 1j * zi

    return y_zone, zs


def initial_parameters(structure):
    ng = 1 if structure == "A" else 6

    # Start from a rough per-pair capacitive admittance.
    # One pF at 10 kHz ~ 0.0628 nS.
    g0 = np.full(5, 0.002)
    b0 = np.array([
        0.06, 0.03, 0.015, 0.008, 0.003
    ])

    zr = np.full(ng, 100.0)
    zi = np.zeros(ng)

    return np.r_[g0, b0, zr, zi]


def parameter_bounds(structure):
    ng = 1 if structure == "A" else 6

    lb = np.r_[
        np.zeros(5),
        np.full(5, -B_MAX_NS),
        np.zeros(ng),
        np.full(ng, -ZS_IMAG_MAX)
    ]

    ub = np.r_[
        np.full(5, G_MAX_NS),
        np.full(5, B_MAX_NS),
        np.full(ng, ZS_REAL_MAX),
        np.full(ng, ZS_IMAG_MAX)
    ]

    return lb, ub


# ============================================================
# 5. PREDICT COMPLEX OPEN ADMITTANCE
#
# A: aggregate all pairs, then one SharedPath.
#
# B: aggregate pairs by High-side SMU,
#    apply separate SharedPath, sum contributions.
# ============================================================

def predict_one(sample, y_zone, zs, structure, mode):
    result = 0j

    for g, zones in enumerate(sample[structure]):
        if zones is None:
            continue

        y_agg = aggregate_pair(
            zones, y_zone, mode
        )

        z = zs[0] if structure == "A" else zs[g]

        den = 1.0 + 1e-3 * z * y_agg

        # Smooth denominator regularization:
        # avoids numerical explosions at poles.
        den_abs2 = np.abs(den)**2

        y_path = (
            y_agg * np.conj(den)
            / (den_abs2 + 1e-12)
        )

        result += y_path

    return result


def predict_rows(indices, t, structure, mode):
    yz, zs = decode(t, structure)

    return np.array([
        predict_one(
            SAMPLES[k], yz, zs,
            structure, mode
        )
        for k in indices
    ])


# ============================================================
# 6. LOSS: BOTH C AND G
# ============================================================

def data_residual(indices, t, structure, mode):
    y = predict_rows(indices, t, structure, mode)

    cp = y.imag / B_PER_PF
    gp = y.real

    return np.r_[
        (cp - C_TRUE[indices]) / C_SCALE_PF,
        (gp - G_TRUE[indices]) / G_SCALE_NS
    ]


def regularization(t, structure):
    yz, zs = decode(t, structure)

    # Prefer finite, moderate shared impedance.
    reg_z = (
        ZS_REG_WEIGHT
        * np.r_[zs.real, zs.imag]
        / ZS_REG_SCALE
    )

    if structure == "B":
        # Weakly encourage SMU impedances to remain
        # near their mean unless data supports differences.
        deviation = zs - zs.mean()

        reg_group = (
            B_GROUP_REG_WEIGHT
            * np.r_[
                deviation.real,
                deviation.imag
            ] / ZS_REG_SCALE
        )

        return np.r_[reg_z, reg_group]

    return reg_z


def fit_model(indices, structure, mode, x0=None):
    if x0 is None:
        x0 = initial_parameters(structure)

    lb, ub = parameter_bounds(structure)

    def residual(t):
        return np.r_[
            data_residual(
                indices, t, structure, mode
            ),
            regularization(t, structure)
        ]

    opt = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        max_nfev=350,
        x_scale="jac",
        ftol=1e-7,
        xtol=1e-7,
        gtol=1e-7
    )

    return opt.x, opt


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate(indices, t, structure, mode):
    y = predict_rows(indices, t, structure, mode)

    cp = y.imag / B_PER_PF
    gp = y.real

    ec = (cp - C_TRUE[indices]) * 1000
    eg = gp - G_TRUE[indices]

    return {
        "C_RMSE_fF": np.sqrt(np.mean(ec**2)),
        "C_MAE_fF": np.mean(np.abs(ec)),
        "C_P95_fF": np.percentile(np.abs(ec), 95),
        "C_Bias_fF": np.mean(ec),
        "G_RMSE_nS": np.sqrt(np.mean(eg**2)),
        "Combined": np.sqrt(np.mean(
            ((cp - C_TRUE[indices]) / C_SCALE_PF)**2
            + ((gp - G_TRUE[indices]) / G_SCALE_NS)**2
        ))
    }


# ============================================================
# 8. ALL 8 MODELS, MATCHED GROUPED SPLITS
# ============================================================

rows = []
parameter_rows = []
model_cache = {}

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    splitter = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    for fold, (tr, te) in enumerate(
        splitter.split(np.arange(N), groups=groups),
        start=1
    ):

        for structure in STRUCTURES:
            for mode in AGG_MODES:

                t, opt = fit_model(
                    tr, structure, mode
                )

                train_m = evaluate(
                    tr, t, structure, mode
                )
                test_m = evaluate(
                    te, t, structure, mode
                )

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Structure": structure,
                    "Aggregation": mode,
                    "Model": f"{structure}-{mode}",
                    "Train_C_RMSE_fF":
                        train_m["C_RMSE_fF"],
                    **test_m
                })

                yz, zs = decode(t, structure)

                parameter_rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Model": f"{structure}-{mode}",
                    "Success": bool(opt.success),
                    "Nfev": opt.nfev,
                    "Shared_abs_mean_MOhm":
                        np.abs(zs).mean(),
                    "Shared_abs_max_MOhm":
                        np.abs(zs).max(),
                    "Pair_G_mean_nS":
                        yz.real.mean(),
                    "Pair_B_mean_nS":
                        yz.imag.mean(),
                    "Params": t.copy()
                })

                model_cache[
                    (split_name, fold, structure, mode)
                ] = t.copy()

                print(
                    f"{split_name:16} F{fold} "
                    f"{structure}-{mode:10} "
                    f"Train={train_m['C_RMSE_fF']:8.1f} "
                    f"Test={test_m['C_RMSE_fF']:8.1f} fF "
                    f"G={test_m['G_RMSE_nS']:.3f} nS"
                )


# ============================================================
# 9. RESULTS
# ============================================================

scores = pd.DataFrame(rows)
param_df = pd.DataFrame(parameter_rows)

summary = (
    scores.groupby(["Split", "Model"])
    .agg(
        C_RMSE_mean=("C_RMSE_fF", "mean"),
        C_RMSE_std=("C_RMSE_fF", "std"),
        C_MAE_mean=("C_MAE_fF", "mean"),
        C_P95_mean=("C_P95_fF", "mean"),
        G_RMSE_mean=("G_RMSE_nS", "mean"),
        Combined_mean=("Combined", "mean"),
        Train_RMSE_mean=("Train_C_RMSE_fF", "mean")
    )
    .reset_index()
)

print("\n=== 8-MODEL COMPARISON ===")
print(
    summary.sort_values(
        ["Split", "C_RMSE_mean"]
    ).round(3).to_string(index=False)
)

print("\n=== UNSEEN SIGNATURE RANKING ===")

ranked = (
    summary[summary["Split"] == "UnseenSignature"]
    .sort_values("C_RMSE_mean")
)

print(ranked.round(3).to_string(index=False))

print("\n=== PARAMETER STABILITY ===")
print(
    param_df.groupby("Model")[
        [
            "Shared_abs_mean_MOhm",
            "Shared_abs_max_MOhm",
            "Pair_G_mean_nS",
            "Pair_B_mean_nS"
        ]
    ].agg(["mean", "std"])
    .round(4).to_string()
)


# ============================================================
# 10. FULL-DATA FIT OF ALL 8 MODELS
# ============================================================

all_idx = np.arange(N)
final_models = {}

for structure in STRUCTURES:
    for mode in AGG_MODES:
        t, opt = fit_model(
            all_idx, structure, mode
        )

        key = f"{structure}-{mode}"

        yz, zs = decode(t, structure)

        final_models[key] = {
            "structure": structure,
            "aggregation": mode,
            "parameters": t,
            "pair_admittance_nS": yz,
            "shared_impedance_MOhm": zs
        }

        print(
            key,
            "Full-data:",
            evaluate(
                all_idx, t, structure, mode
            )["C_RMSE_fF"]
        )


# ============================================================
# 11. SAVE FITTED PARAMETERS AND CV RESULTS
# ============================================================

joblib.dump(
    {
        "frequency_hz": FREQ,
        "models": final_models,
        "zone_names": ZONE_NAMES,
        "smu_groups": [
            list(s) for s in SMUS
        ]
    },
    "pair_sharedpath_8models.joblib"
)

summary.to_csv(
    "pair_sharedpath_8models_summary.csv",
    index=False
)

scores.to_csv(
    "pair_sharedpath_8models_folds.csv",
    index=False
)

# Save parameters without the array column as CSV.
param_df.drop(columns=["Params"]).to_csv(
    "pair_sharedpath_parameter_stability.csv",
    index=False
)
