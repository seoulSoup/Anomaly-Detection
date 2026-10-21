
import ast
import re
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 0. SETTINGS
# ============================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

C_UNIT = "F"
G_UNIT = "S"          # "S", "uS", "nS"

G_FACTOR = {"S": 1e9, "uS": 1e3, "nS": 1.0}
B_PER_PF = OMEGA * 1e-3  # pF -> nS

MODES = ["Parallel", "Mean", "SharedPath"]

N_FOLDS = 3
TEST_SIZE = 0.25
SEED = 42

# Residual scaling
C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

# Individual pin regularization
REG_G_NS = 3.0
REG_B_NS = 30.0
REG_WEIGHT = 1.0

SMUS = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g for g, group in enumerate(SMUS)
    for p in group
}


# ============================================================
# 1. INPUT PARSING
# ============================================================

def parse_pins(value):
    if isinstance(value, str):
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, (list, tuple, set)):
                value = obj
            else:
                value = re.findall(r"\d+", value)
        except (ValueError, SyntaxError):
            value = re.findall(r"\d+", value)

    pins = tuple(sorted(int(p) for p in value))

    if not pins:
        raise ValueError("Empty pin list")
    if len(set(pins)) != len(pins):
        raise ValueError("Duplicate pins")
    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pin out of range")

    return pins


def make_features(high_values, low_values):
    n = len(high_values)

    H = np.zeros((n, 44))
    L = np.zeros((n, 44))

    configs = []
    signatures = []

    for k, (hv, lv) in enumerate(
        zip(high_values, low_values)
    ):
        high = parse_pins(hv)
        low = parse_pins(lv)

        if set(high) & set(low):
            raise ValueError(
                f"Overlapping High/Low at row {k}"
            )

        H[k, np.array(high)-1] = 1.0
        L[k, np.array(low)-1] = 1.0

        hc = [
            sum(p in high for p in smu)
            for smu in SMUS
        ]
        lc = [
            sum(p in low for p in smu)
            for smu in SMUS
        ]

        configs.append(str((high, low)))
        signatures.append(str(tuple(hc + lc)))

    return H, L, np.array(configs), np.array(signatures)


data = df.copy().reset_index(drop=True)

H, L, CONFIG, SIGNATURE = make_features(
    data["high_pins"],
    data["low_pins"]
)

C_TRUE = pd.to_numeric(
    data["C"]
).to_numpy(dtype=float) * 1e12

G_TRUE = pd.to_numeric(
    data["G"]
).to_numpy(dtype=float) * G_FACTOR[G_UNIT]

if not (np.isfinite(C_TRUE).all()
        and np.isfinite(G_TRUE).all()):
    raise ValueError("Invalid C/G measurements")

print("Measurements:", len(C_TRUE))
print("Configurations:", len(set(CONFIG)))
print("Signatures:", len(set(SIGNATURE)))


# ============================================================
# 2. COMPLEX AGGREGATION
#
# Pin admittance: nS
# Shared impedance: MOhm
# ============================================================

def aggregate_side(mask, y_pin, mode, z_shared=None):
    y_sum = mask @ y_pin

    if mode == "Parallel":
        return y_sum

    if mode == "Mean":
        count = mask.sum(axis=1)
        return np.divide(
            y_sum, count,
            out=np.zeros_like(y_sum),
            where=count > 0
        )

    if mode == "SharedPath":
        denominator = 1.0 + 1e-3*z_shared*y_sum

        # Avoid undefined near-singular states.
        if np.any(np.abs(denominator) < 1e-9):
            raise FloatingPointError(
                "SharedPath singularity"
            )

        return y_sum / denominator

    raise ValueError(mode)


def predict_y(h, l, y_pin, mode, z_shared=None):
    yh = aggregate_side(h, y_pin, mode, z_shared)
    yl = aggregate_side(l, y_pin, mode, z_shared)

    den = yh + yl

    # Use small regularized denominator to keep
    # optimization finite near invalid regions.
    safe_den = np.where(
        np.abs(den) > 1e-9, den, 1e-9+0j
    )

    return yh * yl / safe_den


def predict_cg(h, l, y_pin, mode, z_shared=None):
    y = predict_y(h, l, y_pin, mode, z_shared)

    return (
        y.imag / B_PER_PF,
        y.real
    )


# ============================================================
# 3. LOSS
# ============================================================

def measurement_residual(c_pred, g_pred, idx):
    return np.r_[
        (c_pred - C_TRUE[idx]) / C_SCALE_PF,
        (g_pred - G_TRUE[idx]) / G_SCALE_NS
    ]


# ============================================================
# 4. COMMON PIN INITIAL FIT
#
# y_pin = g0 + j*b0
#
# SharedPath adds shared impedance
# z_shared = zr + j*zi [MOhm].
# ============================================================

def unpack_common(t, mode):
    y0 = t[0] + 1j*t[1]

    z_shared = None
    if mode == "SharedPath":
        z_shared = t[2] + 1j*t[3]

    return y0, z_shared


def fit_common(train, mode):
    # Start from a modest capacitive admittance
    # and a small conductance.
    x0 = [0.1, 0.5 * B_PER_PF]

    lb = [-100.0, -1000.0]
    ub = [100.0, 1000.0]

    if mode == "SharedPath":
        x0 += [0.1, 0.0]
        lb += [-100.0, -100.0]
        ub += [100.0, 100.0]

    def residual(t):
        y0, zs = unpack_common(t, mode)

        y_pin = np.full(44, y0, dtype=complex)

        try:
            c, g = predict_cg(
                H[train], L[train],
                y_pin, mode, zs
            )
            res = measurement_residual(c, g, train)
        except FloatingPointError:
            res = np.full(2*len(train), 1e6)

        # Mild shared impedance prior
        if mode == "SharedPath":
            res = np.r_[
                res,
                0.05*t[2],
                0.05*t[3]
            ]

        return res

    result = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        max_nfev=1200,
        x_scale="jac"
    )

    return result.x


# ============================================================
# 5. INDIVIDUAL COMPLEX PIN FIT
#
# y_i = y0 + dg_i + j*db_i
#
# The common component is fixed.
# ============================================================

def unpack_individual(common_t, v, mode):
    y0, zs = unpack_common(common_t, mode)

    dg = v[:44] - v[:44].mean()
    db = v[44:] - v[44:].mean()

    y_pin = y0 + dg + 1j*db

    return y_pin, zs


def fit_individual(train, common_t, mode):

    def residual(v):
        y_pin, zs = unpack_individual(
            common_t, v, mode
        )

        try:
            c, g = predict_cg(
                H[train], L[train],
                y_pin, mode, zs
            )
            res = measurement_residual(c, g, train)
        except FloatingPointError:
            res = np.full(2*len(train), 1e6)

        dg = v[:44] - v[:44].mean()
        db = v[44:] - v[44:].mean()

        reg = REG_WEIGHT * np.r_[
            dg / REG_G_NS,
            db / REG_B_NS
        ]

        return np.r_[res, reg]

    result = least_squares(
        residual,
        np.zeros(88),
        bounds=(
            np.r_[np.full(44, -100.0),
                  np.full(44, -300.0)],
            np.r_[np.full(44, 100.0),
                  np.full(44, 300.0)]
        ),
        max_nfev=500,
        x_scale="jac"
    )

    return result.x


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate(c_pred, g_pred, idx):
    ec = (c_pred - C_TRUE[idx]) * 1000
    eg = g_pred - G_TRUE[idx]

    return {
        "C_RMSE_fF": float(
            np.sqrt(np.mean(ec**2))
        ),
        "C_MAE_fF": float(np.mean(np.abs(ec))),
        "C_P95_fF": float(
            np.percentile(np.abs(ec), 95)
        ),
        "G_RMSE_nS": float(
            np.sqrt(np.mean(eg**2))
        )
    }


# ============================================================
# 7. GROUPED HOLDOUT COMPARISON
# ============================================================

rows = []
parameter_rows = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    splitter = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (tr, te) in enumerate(
        splitter.split(H, groups=groups),
        start=1
    ):
        for mode in MODES:

            common = fit_common(tr, mode)
            individual = fit_individual(
                tr, common, mode
            )

            for fit_name in ["Common", "Individual"]:
                if fit_name == "Common":
                    y0, zs = unpack_common(common, mode)
                    y_pin = np.full(44, y0, dtype=complex)
                else:
                    y_pin, zs = unpack_individual(
                        common, individual, mode
                    )

                c_pred, g_pred = predict_cg(
                    H[te], L[te], y_pin, mode, zs
                )

                m = evaluate(c_pred, g_pred, te)

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Aggregation": mode,
                    "PinFit": fit_name,
                    **m
                })

                print(
                    f"{split_name:16} F{fold} "
                    f"{mode:11} {fit_name:10} "
                    f"C={m['C_RMSE_fF']:8.1f} fF "
                    f"G={m['G_RMSE_nS']:.3f} nS"
                )

            y_pin, zs = unpack_individual(
                common, individual, mode
            )

            parameter_rows.append({
                "Split": split_name,
                "Fold": fold,
                "Aggregation": mode,
                "G_Mean_nS": y_pin.real.mean(),
                "B_Mean_nS": y_pin.imag.mean(),
                "G_Std_nS": y_pin.real.std(),
                "B_Std_nS": y_pin.imag.std(),
                "Zshared_Re_MOhm": (
                    zs.real if zs is not None else np.nan
                ),
                "Zshared_Im_MOhm": (
                    zs.imag if zs is not None else np.nan
                )
            })


# ============================================================
# 8. SUMMARY
# ============================================================

scores = pd.DataFrame(rows)
parameters = pd.DataFrame(parameter_rows)

summary = (
    scores.groupby(["Split", "Aggregation", "PinFit"])
    .agg(
        C_RMSE_mean=("C_RMSE_fF", "mean"),
        C_RMSE_std=("C_RMSE_fF", "std"),
        C_MAE_mean=("C_MAE_fF", "mean"),
        C_P95_mean=("C_P95_fF", "mean"),
        G_RMSE_mean=("G_RMSE_nS", "mean")
    )
    .reset_index()
)

print("\n=== PIN IMPEDANCE ONLY MODEL ===")
print(summary.round(3).to_string(index=False))

print("\n=== PARAMETER STABILITY ===")
print(
    parameters.groupby(["Split", "Aggregation"])[[
        "G_Mean_nS", "B_Mean_nS",
        "G_Std_nS", "B_Std_nS",
        "Zshared_Re_MOhm", "Zshared_Im_MOhm"
    ]].agg(["mean", "std"]).round(3).to_string()
)


# ============================================================
# 9. FINAL MODEL
#
# Choose one aggregation after reviewing CV.
# Can change this manually.
# ============================================================

BEST_MODE = "Parallel"

all_idx = np.arange(len(data))

final_common = fit_common(all_idx, BEST_MODE)
final_individual = fit_individual(
    all_idx, final_common, BEST_MODE
)

final_y_pin, final_zs = unpack_individual(
    final_common, final_individual, BEST_MODE
)


# ============================================================
# 10. FULL-DATA ACTUAL VS PREDICTED
# ============================================================

full_c, full_g = predict_cg(
    H, L,
    final_y_pin,
    BEST_MODE,
    final_zs
)

full_stats = evaluate(
    full_c, full_g, all_idx
)

print("\n=== FULL DATA FIT ===")
print(full_stats)

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    C_TRUE, full_c,
    alpha=0.6
)

lo = min(C_TRUE.min(), full_c.min())
hi = max(C_TRUE.max(), full_c.max())

ax.plot([lo, hi], [lo, hi], "--")
ax.set_xlabel("Measured Open C (pF)")
ax.set_ylabel("Predicted Open C (pF)")
ax.set_title(
    f"{BEST_MODE} | Pin Impedance Only"
)
ax.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 11. PIN COMPLEX IMPEDANCE REPORT
# ============================================================

z_pin_ohm = 1.0 / (final_y_pin * 1e-9)

pin_table = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "SMU": [
        PIN_SMU[p] + 1 for p in range(1, 45)
    ],
    "G_pin_nS": final_y_pin.real,
    "B_pin_nS": final_y_pin.imag,
    "Z_real_MOhm": z_pin_ohm.real / 1e6,
    "Z_imag_MOhm": z_pin_ohm.imag / 1e6
})

print("\n=== INDIVIDUAL PIN IMPEDANCES ===")
print(pin_table.round(4).to_string(index=False))


# ============================================================
# 12. INDEPENDENT PREDICTION
# ============================================================

def predict_open_cap(high, low):
    h, l, _, _ = make_features([high], [low])

    c, g = predict_cg(
        h, l, final_y_pin,
        BEST_MODE, final_zs
    )

    return {
        "C_pF": float(c[0]),
        "G_nS": float(g[0]),
        "Y_S": complex(
            (g[0] + 1j*B_PER_PF*c[0]) * 1e-9
        )
    }


print("\n=== EXAMPLE ===")
print(
    predict_open_cap(
        high=list(range(15, 45)),
        low=list(range(7, 14))
    )
)

# ============================================================
# 13. SAVE
# ============================================================

joblib.dump({
    "aggregation": BEST_MODE,
    "y_pin_nS": final_y_pin,
    "z_shared_MOhm": final_zs,
    "frequency_hz": FREQ,
    "pair_correction": False
}, "pin_impedance_only.joblib")

summary.to_csv(
    "pin_impedance_only_summary.csv",
    index=False
)

pin_table.to_csv(
    "pin_impedance_only_parameters.csv",
    index=False
)
