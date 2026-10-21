
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

# Measured C: farads.
# Measured G: select actual unit.
G_UNIT = "S"
G_FACTOR = {"S": 1e9, "uS": 1e3, "nS": 1.0}

# Conversion:
# pF -> nS of capacitive susceptance.
B_PER_PF = OMEGA * 1e-3

STRUCTURES = ["M1", "M2"]
AGGREGATIONS = ["Sum", "Mean", "MaxPerPin", "Sqrt"]

N_FOLDS = 3
TEST_SIZE = 0.25
SEED = 42

# Objective scales
C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

# Exploratory bounds, not physical calibration limits.
# Total R, L, G, C per pair category.
R_MAX_MOHM = 100.0
L_MAX_H = 100.0
G_MAX_NS = 100.0
C_MAX_PF = 30.0

# Common shared impedance
ZS_MAX_MOHM = 100.0

# Weak priors for poorly identifiable R/L.
REG_R = 0.05
REG_L = 0.05
REG_SHARED = 0.05

MAX_NFEV_STAGE1 = 150
MAX_NFEV_STAGE2 = 250

SMUS = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    pin: g
    for g, group in enumerate(SMUS)
    for pin in group
}

ZONE_NAMES = [
    "Even",
    "Odd",
    "Dist2",
    "FarSameSMU",
    "CrossSMU"
]


# ============================================================
# 1. INPUT DATA
# ============================================================

def parse_pins(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = re.findall(r"\d+", value)

    pins = tuple(sorted(int(p) for p in value))

    if not pins:
        raise ValueError("Empty pin selection")

    if len(pins) != len(set(pins)):
        raise ValueError("Duplicate pins")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pin outside range 1..44")

    return pins


data = df.copy().reset_index(drop=True)

HIGH = [parse_pins(x) for x in data["high_pins"]]
LOW = [parse_pins(x) for x in data["low_pins"]]

for h, l in zip(HIGH, LOW):
    if set(h) & set(l):
        raise ValueError("High/Low overlap")

C_TRUE = (
    pd.to_numeric(data["C"]).to_numpy(dtype=float)
    * 1e12
)

G_TRUE = (
    pd.to_numeric(data["G"]).to_numpy(dtype=float)
    * G_FACTOR[G_UNIT]
)

assert np.isfinite(C_TRUE).all()
assert np.isfinite(G_TRUE).all()

N = len(data)

CONFIG = np.array([
    str((h, l))
    for h, l in zip(HIGH, LOW)
])

def make_signature(h, l):
    hc = [sum(p in h for p in group) for group in SMUS]
    lc = [sum(p in l for p in group) for group in SMUS]
    return str(tuple(hc + lc))

SIGNATURE = np.array([
    make_signature(h, l)
    for h, l in zip(HIGH, LOW)
])


# ============================================================
# 2. PAIR CLASSIFICATION
# ============================================================

def pair_zone(h, l):
    if PIN_SMU[h] != PIN_SMU[l]:
        return 4

    d = abs(h-l)

    if d == 1:
        return 0 if max(h, l) % 2 == 0 else 1

    if d == 2:
        return 2

    return 3


# Each sample stores its H x L zone-index matrix.
PAIR_MATRICES = []

for h, l in zip(HIGH, LOW):
    zones = np.array([
        [pair_zone(hp, lp) for lp in l]
        for hp in h
    ], dtype=np.int8)

    PAIR_MATRICES.append(zones)


# ============================================================
# 3. PARAMETER REPRESENTATION
#
# t[0:5]   = R [MOhm]
# t[5:10]  = L [H]
# t[10:15] = G [nS]
# t[15:20] = C [pF]
# t[20]    = Shared R [MOhm]
# t[21]    = Shared X [MOhm]
#
# Note: parameters are TOTAL path-equivalent RLGC,
# not per-unit-length values.
# ============================================================

def decode(t):
    R = t[0:5]
    L = t[5:10]
    G = t[10:15]
    C = t[15:20]

    # Series Z in MOhm
    Z = R + 1j * OMEGA * L / 1e6

    # Shunt Y in nS
    Y = G + 1j * B_PER_PF * C

    Zshared = complex(t[20], t[21])

    return Z, Y, Zshared


def initial_params():
    R = np.full(5, 1.0)
    L = np.full(5, 0.01)

    G = np.full(5, 0.002)
    C = np.array([
        1.0, 0.6, 0.2, 0.1, 0.05
    ])

    return np.r_[
        R, L, G, C,
        1.0, 0.0
    ]


def bounds():
    lb = np.r_[
        np.full(5, 1e-6),   # R > 0
        np.zeros(5),         # L >= 0
        np.zeros(5),         # G >= 0
        np.zeros(5),         # C >= 0
        0.0,                # Shared R
        -ZS_MAX_MOHM        # Shared X
    ]

    ub = np.r_[
        np.full(5, R_MAX_MOHM),
        np.full(5, L_MAX_H),
        np.full(5, G_MAX_NS),
        np.full(5, C_MAX_PF),
        ZS_MAX_MOHM,
        ZS_MAX_MOHM
    ]

    return lb, ub


# ============================================================
# 4. OPEN TRANSMISSION-LINE RESPONSE
#
# z [MOhm], y [nS]
#
# Dimensionless:
#    x^2 = Z * Y = 1e-3*z*y
#
# Yin[nS] = y*tanh(x)/x
#
# Small x expansion:
#    tanh(x)/x = 1 - x^2/3 + 2*x^4/15 + ...
# ============================================================

def open_y(z, y):
    x2 = 1e-3 * z * y

    if abs(x2) < 1e-7:
        factor = (
            1.0
            - x2 / 3.0
            + 2.0 * x2*x2 / 15.0
        )
        return y * factor

    x = np.sqrt(x2 + 0j)

    return y * np.tanh(x) / x


# ============================================================
# 5. COMPLEX PAIR AGGREGATION
# ============================================================

def aggregate_complex_pair(W, mode):
    S = np.sum(W)

    if mode == "Sum":
        return S

    if mode == "Mean":
        return S / W.size

    # Strongest pair for each selected pin.
    ih = np.argmax(np.abs(W), axis=1)
    il = np.argmax(np.abs(W), axis=0)

    mh = W[np.arange(W.shape[0]), ih].sum()
    ml = W[il, np.arange(W.shape[1])].sum()

    M = 0.5 * (mh + ml)

    if mode == "MaxPerPin":
        return M

    if mode == "Sqrt":
        amplitude = np.sqrt(
            np.abs(S) * np.abs(M)
        )

        if abs(S) > 1e-14:
            phase = S / abs(S)
        elif abs(M) > 1e-14:
            phase = M / abs(M)
        else:
            return 0j

        return amplitude * phase

    raise ValueError(mode)


# ============================================================
# 6. M1: AGGREGATE RLGC FIRST
#
# Z series paths: parallel equivalent.
# Y shunt paths : requested aggregation.
#
# Then calculate ONE open-line response.
# ============================================================

def predict_M1(zones, Z, Y, aggregation):
    z_pairs = Z[zones]
    y_pairs = Y[zones]

    z_agg = 1.0 / np.sum(
        1.0 / z_pairs
    )

    y_agg = aggregate_complex_pair(
        y_pairs, aggregation
    )

    return open_y(z_agg, y_agg), z_agg, y_agg


# ============================================================
# 7. M2: CALCULATE INDIVIDUAL PAIR OPEN Y FIRST
#
# Then aggregate all pair responses.
# ============================================================

def predict_M2(zones, Z, Y, aggregation):
    # Only five zone responses needed.
    y_pair_open = np.array([
        open_y(Z[k], Y[k])
        for k in range(5)
    ])

    W = y_pair_open[zones]

    y_agg = aggregate_complex_pair(
        W, aggregation
    )

    return y_agg, None, None


# ============================================================
# 8. SHARED PATH
#
# Uses one common complex series impedance.
# ============================================================

def apply_shared_path(y, z_shared):
    den = 1.0 + 1e-3*z_shared*y

    # Smooth regularization near a singularity.
    return (
        y * np.conj(den)
        / (abs(den)**2 + 1e-12)
    )


# ============================================================
# 9. UNIFIED PREDICTION
# ============================================================

def predict_one(zones, t, structure, aggregation):
    Z, Y, Zshared = decode(t)

    if structure == "M1":
        yo, _, _ = predict_M1(
            zones, Z, Y, aggregation
        )

    elif structure == "M2":
        yo, _, _ = predict_M2(
            zones, Z, Y, aggregation
        )

    else:
        raise ValueError(structure)

    return apply_shared_path(yo, Zshared)


def predict_rows(indices, t, structure, aggregation):
    # M2 category responses are constant for a given t.
    Z, Y, Zshared = decode(t)

    if structure == "M2":
        y_pair = np.array([
            open_y(Z[k], Y[k])
            for k in range(5)
        ])

        return np.array([
            apply_shared_path(
                aggregate_complex_pair(
                    y_pair[PAIR_MATRICES[k]],
                    aggregation
                ),
                Zshared
            )
            for k in indices
        ])

    return np.array([
        predict_one(
            PAIR_MATRICES[k],
            t, structure, aggregation
        )
        for k in indices
    ])


# ============================================================
# 10. METRICS & OBJECTIVE
# ============================================================

def metrics(indices, y_pred):
    cp = y_pred.imag / B_PER_PF
    gp = y_pred.real

    err_c = (cp - C_TRUE[indices]) * 1000
    err_g = gp - G_TRUE[indices]

    return {
        "C_RMSE_fF": float(
            np.sqrt(np.mean(err_c**2))
        ),
        "C_MAE_fF": float(
            np.mean(np.abs(err_c))
        ),
        "G_RMSE_nS": float(
            np.sqrt(np.mean(err_g**2))
        ),
        "Combined": float(np.sqrt(np.mean(
            ((cp-C_TRUE[indices])/C_SCALE_PF)**2
            + ((gp-G_TRUE[indices])/G_SCALE_NS)**2
        )))
    }


def residual_data(indices, t, structure, aggregation):
    y_pred = predict_rows(
        indices, t, structure, aggregation
    )

    cp = y_pred.imag / B_PER_PF
    gp = y_pred.real

    return np.r_[
        (cp - C_TRUE[indices]) / C_SCALE_PF,
        (gp - G_TRUE[indices]) / G_SCALE_NS
    ]


def regularization(t):
    # Weak priors keep underdetermined R/L
    # from going to extreme values too easily.
    return np.r_[
        REG_R * t[:5] / 10.0,
        REG_L * t[5:10] / 10.0,
        REG_SHARED * t[20:22] / 10.0
    ]


# ============================================================
# 11. TWO-STAGE FIT
#
# Stage 1:
#   Optimize only G/C, keeping R/L and Zshared fixed.
#
# Stage 2:
#   Release all RLGC + SharedPath parameters.
# ============================================================

def fit_model(indices, structure, aggregation):
    x0 = initial_params()
    lb, ub = bounds()

    # -------- Stage 1: G/C only --------
    def res_stage1(gc):
        t = x0.copy()
        t[10:20] = gc

        return residual_data(
            indices, t, structure, aggregation
        )

    opt1 = least_squares(
        res_stage1,
        x0[10:20],
        bounds=(lb[10:20], ub[10:20]),
        x_scale="jac",
        max_nfev=MAX_NFEV_STAGE1
    )

    x1 = x0.copy()
    x1[10:20] = opt1.x

    # -------- Stage 2: all parameters --------
    def res_stage2(t):
        return np.r_[
            residual_data(
                indices, t, structure, aggregation
            ),
            regularization(t)
        ]

    opt2 = least_squares(
        res_stage2,
        x1,
        bounds=(lb, ub),
        x_scale="jac",
        max_nfev=MAX_NFEV_STAGE2,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8
    )

    return opt2.x, opt1, opt2


# ============================================================
# 12. IDENTIFIABILITY DIAGNOSTICS
#
# Effective electrical length:
#    gamma*l = sqrt(Zagg * Yagg)
#
# M1: aggregated electrical length.
# M2: individual pair electrical lengths.
# ============================================================

def electrical_length_stats(indices, t, structure, aggregation):
    Z, Y, Zshared = decode(t)

    values = []

    if structure == "M1":
        for k in indices:
            _, zagg, yagg = predict_M1(
                PAIR_MATRICES[k],
                Z, Y, aggregation
            )
            values.append(
                abs(np.sqrt(1e-3*zagg*yagg))
            )

    else:
        for z, y in zip(Z, Y):
            values.append(
                abs(np.sqrt(1e-3*z*y))
            )

    values = np.array(values)

    return {
        "GammaL_mean": float(np.mean(values)),
        "GammaL_max": float(np.max(values)),
        "GammaL_min": float(np.min(values))
    }


def bound_diagnostics(t):
    lb, ub = bounds()

    normalized = (
        (t - lb) / (ub - lb)
    )

    return {
        "NearLower": int(np.sum(normalized < 1e-3)),
        "NearUpper": int(np.sum(normalized > 1-1e-3))
    }


# ============================================================
# 13. MATCHED GROUPED CV FOR ALL 8 MODELS
# ============================================================

rows = []
params = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    cv = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (tr, te) in enumerate(
        cv.split(np.arange(N), groups=groups),
        start=1
    ):
        for structure in STRUCTURES:
            for agg in AGGREGATIONS:

                t, opt1, opt2 = fit_model(
                    tr, structure, agg
                )

                y_train = predict_rows(
                    tr, t, structure, agg
                )
                y_test = predict_rows(
                    te, t, structure, agg
                )

                train_m = metrics(tr, y_train)
                test_m = metrics(te, y_test)

                diag = electrical_length_stats(
                    te, t, structure, agg
                )

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Model": f"{structure}-{agg}",
                    "Structure": structure,
                    "Aggregation": agg,
                    "Train_C_RMSE_fF":
                        train_m["C_RMSE_fF"],
                    **test_m,
                    **diag,
                    **bound_diagnostics(t),
                    "Success": bool(opt2.success)
                })

                params.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Model": f"{structure}-{agg}",
                    "Parameters": t.copy()
                })

                print(
                    f"{split_name:16} F{fold} "
                    f"{structure}-{agg:10} "
                    f"Train={train_m['C_RMSE_fF']:8.1f} "
                    f"Test={test_m['C_RMSE_fF']:8.1f} fF "
                    f"|gamma*l|={diag['GammaL_mean']:.5f}"
                )


# ============================================================
# 14. SUMMARY
# ============================================================

scores = pd.DataFrame(rows)

summary = (
    scores.groupby(["Split", "Model"])
    .agg(
        C_RMSE_mean=("C_RMSE_fF", "mean"),
        C_RMSE_std=("C_RMSE_fF", "std"),
        G_RMSE_mean=("G_RMSE_nS", "mean"),
        Combined_mean=("Combined", "mean"),
        Train_RMSE_mean=("Train_C_RMSE_fF", "mean"),
        GammaL_mean=("GammaL_mean", "mean"),
        GammaL_max=("GammaL_max", "max"),
        BoundLower_mean=("NearLower", "mean"),
        BoundUpper_mean=("NearUpper", "mean")
    )
    .reset_index()
)

print("\n=== RLGC AGGREGATION COMPARISON ===")
print(
    summary.sort_values(
        ["Split", "C_RMSE_mean"]
    ).round(5).to_string(index=False)
)

print("\n=== UNSEEN SIGNATURE RANKING ===")
print(
    summary[
        summary["Split"] == "UnseenSignature"
    ].sort_values(
        "C_RMSE_mean"
    ).round(5).to_string(index=False)
)


# ============================================================
# 15. ALL-DATA REFIT
# ============================================================

all_idx = np.arange(N)
final_models = {}

for structure in STRUCTURES:
    for agg in AGGREGATIONS:
        key = f"{structure}-{agg}"

        t, _, opt = fit_model(
            all_idx, structure, agg
        )

        Z, Y, Zshared = decode(t)

        final_models[key] = {
            "structure": structure,
            "aggregation": agg,
            "parameters": t,
            "pair_z_MOhm": Z,
            "pair_y_nS": Y,
            "z_shared_MOhm": Zshared
        }

        result = metrics(
            all_idx,
            predict_rows(
                all_idx, t, structure, agg
            )
        )

        print(
            f"{key:16} Full-data C RMSE "
            f"{result['C_RMSE_fF']:.2f} fF"
        )


# ============================================================
# 16. PARAMETER REPORT
# ============================================================

best_row = (
    summary[
        summary["Split"] == "UnseenSignature"
    ]
    .sort_values("C_RMSE_mean")
    .iloc[0]
)

BEST_MODEL = best_row["Model"]
best_t = final_models[BEST_MODEL]["parameters"]

best_report = pd.DataFrame({
    "Zone": ZONE_NAMES,
    "R_MOhm": best_t[:5],
    "L_H": best_t[5:10],
    "G_nS": best_t[10:15],
    "C_pF": best_t[15:20]
})

print("\n=== BEST MODEL ===")
print(BEST_MODEL)
print(best_report.round(8).to_string(index=False))

print("\nShared path Z [MOhm]:")
print(complex(best_t[20], best_t[21]))


# ============================================================
# 17. SAVE
# ============================================================

joblib.dump({
    "frequency_hz": FREQ,
    "models": final_models,
    "zone_names": ZONE_NAMES,
    "smu_groups": [
        list(group) for group in SMUS
    ]
}, "rlgc_aggregation_models.joblib")

summary.to_csv(
    "rlgc_aggregation_summary.csv",
    index=False
)

scores.to_csv(
    "rlgc_aggregation_folds.csv",
    index=False
)

best_report.to_csv(
    "rlgc_best_parameters.csv",
    index=False
)
