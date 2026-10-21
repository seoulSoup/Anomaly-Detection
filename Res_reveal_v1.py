
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 0. SETTINGS
# ============================================================

FREQ = 10000.0
OMEGA = 2.0 * np.pi * FREQ

# df["C"] assumed in farads
C_TO_PF = 1e12

# Set according to the actual df["G"] unit
G_UNIT = "S"
G_TO_NS = {
    "S": 1e9,
    "uS": 1e3,
    "nS": 1.0
}

# pF -> nS capacitive susceptance
B_PER_PF = OMEGA * 1e-3

# Fit C and G jointly
C_SCALE_PF = 0.5
G_SCALE_NS = 1.0

# Individual pin regularization
PIN_G_SIGMA_NS = 3.0
PIN_B_SIGMA_NS = 30.0

# SharedPath regularization
SHARED_Z_SIGMA_MOHM = 100.0
SHARED_REG_WEIGHT = 0.05

# Optimization
MAX_NFEV_COMMON = 400
MAX_NFEV_INDIVIDUAL = 300

# Optional true holdout analysis
RUN_HOLDOUT = False
HOLDOUT_SPLIT = "UnseenSignature"
HOLDOUT_FOLDS = 3
HOLDOUT_SIZE = 0.25
SEED = 42

# Outlier selection
OUTLIER_QUANTILE = 0.95
MIN_OUTLIER_FF = 300.0
TOP_N = 30

OUTPUT_DIR = Path("sharedpath_outlier_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMUS = [
    list(range(1, 7)),
    list(range(7, 15)),
    list(range(15, 23)),
    list(range(23, 31)),
    list(range(31, 39)),
    list(range(39, 45))
]

PIN_SMU = {
    p: g + 1
    for g, group in enumerate(SMUS)
    for p in group
}


# ============================================================
# 1. DATA PREPARATION
# ============================================================

def parse_pins(value):
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = re.findall(r"\d+", value)

    if isinstance(value, (int, np.integer)):
        value = [value]

    pins = tuple(sorted(int(x) for x in value))

    if not pins:
        raise ValueError("Empty pin selection")

    if len(pins) != len(set(pins)):
        raise ValueError("Duplicate pins")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pins must be in 1..44")

    return pins


data = df.copy().reset_index(drop=True)

required = {"high_pins", "low_pins", "C", "G"}

if not required.issubset(data.columns):
    raise ValueError(
        f"Missing columns: {required - set(data.columns)}"
    )

HIGH = [
    parse_pins(x) for x in data["high_pins"]
]

LOW = [
    parse_pins(x) for x in data["low_pins"]
]

N = len(data)

H = np.zeros((N, 44), dtype=float)
L = np.zeros((N, 44), dtype=float)

CONFIG = []
SIGNATURE = []

for k, (high, low) in enumerate(zip(HIGH, LOW)):
    if set(high) & set(low):
        raise ValueError(f"High/Low overlap at row {k}")

    H[k, np.asarray(high)-1] = 1.0
    L[k, np.asarray(low)-1] = 1.0

    CONFIG.append(str((high, low)))

    high_counts = [
        sum(p in high for p in smu)
        for smu in SMUS
    ]

    low_counts = [
        sum(p in low for p in smu)
        for smu in SMUS
    ]

    SIGNATURE.append(
        str(tuple(high_counts + low_counts))
    )

CONFIG = np.asarray(CONFIG)
SIGNATURE = np.asarray(SIGNATURE)

C_TRUE = (
    pd.to_numeric(data["C"]).to_numpy(dtype=float)
    * C_TO_PF
)

G_TRUE = (
    pd.to_numeric(data["G"]).to_numpy(dtype=float)
    * G_TO_NS[G_UNIT]
)

if not (
    np.isfinite(C_TRUE).all()
    and np.isfinite(G_TRUE).all()
):
    raise ValueError("C/G contains nonfinite values")

print("Rows:", N)
print("Unique configurations:", len(np.unique(CONFIG)))
print("Unique SMU signatures:", len(np.unique(SIGNATURE)))


# ============================================================
# 2. SHARED PATH + INDIVIDUAL COMPLEX Y MODEL
#
# y_pin = G_pin + j B_pin  [nS]
#
# Shared impedance Zs [MOhm]
#
# YH = sum(Y_pin) / (1 + 1e-3 Zs sum(Y_pin))
# YL = same for Low
# Yopen = YH*YL/(YH+YL)
# ============================================================

def predict_complex(idx, y_pin, z_shared):
    yh_raw = H[idx] @ y_pin
    yl_raw = L[idx] @ y_pin

    den_h = 1.0 + 1e-3*z_shared*yh_raw
    den_l = 1.0 + 1e-3*z_shared*yl_raw

    with np.errstate(divide="ignore", invalid="ignore"):
        yh = yh_raw / den_h
        yl = yl_raw / den_l
        y_pred = yh * yl / (yh + yl)

    if not np.isfinite(y_pred).all():
        return np.full(len(idx), 1e12 + 1e12j)

    return y_pred


def data_residual(idx, y_pred):
    c_pred = y_pred.imag / B_PER_PF
    g_pred = y_pred.real

    return np.r_[
        (c_pred - C_TRUE[idx]) / C_SCALE_PF,
        (g_pred - G_TRUE[idx]) / G_SCALE_NS
    ]


# ============================================================
# 3. STAGE 1: COMMON COMPLEX Y
# ============================================================

def fit_common(train):
    # g0, b0, shared_R, shared_X
    x0 = np.array([
        0.1,
        0.5 * B_PER_PF,
        1.0,
        0.0
    ])

    lb = [-100, -1000, -100, -100]
    ub = [100, 1000, 100, 100]

    def residual(t):
        y0 = complex(t[0], t[1])
        zs = complex(t[2], t[3])

        yp = np.full(44, y0, dtype=complex)

        pred = predict_complex(train, yp, zs)

        reg = SHARED_REG_WEIGHT * np.array([
            t[2] / SHARED_Z_SIGMA_MOHM,
            t[3] / SHARED_Z_SIGMA_MOHM
        ])

        return np.r_[
            data_residual(train, pred),
            reg
        ]

    result = least_squares(
        residual,
        x0,
        bounds=(lb, ub),
        max_nfev=MAX_NFEV_COMMON,
        x_scale="jac"
    )

    return result.x


# ============================================================
# 4. STAGE 2: INDIVIDUAL COMPLEX Y
#
# Fit pin deviations from common values.
# ============================================================

def unpack_individual(common, v):
    y0 = complex(common[0], common[1])
    zs = complex(common[2], common[3])

    dg = v[:44] - np.mean(v[:44])
    db = v[44:] - np.mean(v[44:])

    yp = y0 + dg + 1j*db

    return yp, zs


def fit_individual(train, common):
    def residual(v):
        yp, zs = unpack_individual(common, v)

        pred = predict_complex(train, yp, zs)

        dg = v[:44] - np.mean(v[:44])
        db = v[44:] - np.mean(v[44:])

        return np.r_[
            data_residual(train, pred),
            dg / PIN_G_SIGMA_NS,
            db / PIN_B_SIGMA_NS
        ]

    lb = np.r_[
        np.full(44, -100.0),
        np.full(44, -300.0)
    ]

    ub = np.r_[
        np.full(44, 100.0),
        np.full(44, 300.0)
    ]

    result = least_squares(
        residual,
        np.zeros(88),
        bounds=(lb, ub),
        max_nfev=MAX_NFEV_INDIVIDUAL,
        x_scale="jac"
    )

    return result.x


def train_model(train):
    common = fit_common(train)

    individual = fit_individual(train, common)

    y_pin, z_shared = unpack_individual(
        common, individual
    )

    return {
        "y_pin": y_pin,
        "z_shared": z_shared,
        "common": common,
        "individual": individual
    }


# ============================================================
# 5. PREDICTION
# ============================================================

def predict_model(model, idx):
    return predict_complex(
        idx,
        model["y_pin"],
        model["z_shared"]
    )


# ============================================================
# 6. FULL DATA FIT
# ============================================================

all_idx = np.arange(N)

model_full = train_model(all_idx)

y_full = predict_model(model_full, all_idx)

c_full = y_full.imag / B_PER_PF
g_full = y_full.real

err_full = (c_full - C_TRUE) * 1000

full_rmse = np.sqrt(np.mean(err_full**2))

print("\n=== FULL DATA FIT ===")
print(f"C RMSE: {full_rmse:.2f} fF")
print(f"Shared Z: {model_full['z_shared']}")
print("Pin Y mean [nS]:", np.mean(model_full["y_pin"]))


# ============================================================
# 7. OPTIONAL GROUPED HOLDOUT
#
# CV predictions are recorded only for test rows.
# A row may appear in more than one fold.
# ============================================================

holdout_records = []

if RUN_HOLDOUT:
    groups = (
        SIGNATURE
        if HOLDOUT_SPLIT == "UnseenSignature"
        else CONFIG
    )

    splitter = GroupShuffleSplit(
        n_splits=HOLDOUT_FOLDS,
        test_size=HOLDOUT_SIZE,
        random_state=SEED
    )

    for fold, (train, test) in enumerate(
        splitter.split(H, groups=groups),
        start=1
    ):
        model = train_model(train)
        pred = predict_model(model, test)

        cp = pred.imag / B_PER_PF
        gp = pred.real

        fold_rmse = np.sqrt(
            np.mean((cp - C_TRUE[test])**2)
        ) * 1000

        print(
            f"Holdout fold {fold}: "
            f"C RMSE = {fold_rmse:.2f} fF"
        )

        for k, c, g in zip(test, cp, gp):
            holdout_records.append({
                "Row": int(k),
                "Fold": fold,
                "C_pred_pF": float(c),
                "G_pred_nS": float(g)
            })


# ============================================================
# 8. BUILD ANALYSIS TABLE
#
# Default: full-data fitting residuals.
# Holdout option: test-only predictions.
# ============================================================

if RUN_HOLDOUT:
    holdout_df = pd.DataFrame(holdout_records)

    analysis = holdout_df.copy()

    row_id = analysis["Row"].to_numpy(dtype=int)

    analysis["C_actual_pF"] = C_TRUE[row_id]
    analysis["G_actual_nS"] = G_TRUE[row_id]

else:
    analysis = pd.DataFrame({
        "Row": np.arange(N),
        "Fold": 0,
        "C_actual_pF": C_TRUE,
        "G_actual_nS": G_TRUE,
        "C_pred_pF": c_full,
        "G_pred_nS": g_full
    })

analysis["Error_fF"] = (
    analysis["C_pred_pF"] - analysis["C_actual_pF"]
) * 1000

analysis["AbsError_fF"] = np.abs(
    analysis["Error_fF"]
)

analysis["High"] = [
    HIGH[k] for k in analysis["Row"]
]

analysis["Low"] = [
    LOW[k] for k in analysis["Row"]
]

analysis["Config"] = [
    CONFIG[k] for k in analysis["Row"]
]

analysis["Signature"] = [
    SIGNATURE[k] for k in analysis["Row"]
]

analysis["N_High"] = [
    len(HIGH[k]) for k in analysis["Row"]
]

analysis["N_Low"] = [
    len(LOW[k]) for k in analysis["Row"]
]


# ============================================================
# 9. OUTLIER THRESHOLD
#
# Choose at least MIN_OUTLIER_FF and
# the upper OUTLIER_QUANTILE of |error|.
# ============================================================

threshold = max(
    MIN_OUTLIER_FF,
    np.quantile(
        analysis["AbsError_fF"],
        OUTLIER_QUANTILE
    )
)

analysis["Outlier"] = (
    analysis["AbsError_fF"] >= threshold
)

outliers = (
    analysis[analysis["Outlier"]]
    .sort_values("AbsError_fF", ascending=False)
    .copy()
)

print("\n=== OUTLIER SUMMARY ===")
print("Threshold [fF]:", round(threshold, 2))
print("Outliers:", len(outliers), "/", len(analysis))

print("\n=== TOP OUTLIER CONFIGURATIONS ===")
print(
    outliers[[
        "Row", "Fold", "High", "Low",
        "N_High", "N_Low",
        "C_actual_pF", "C_pred_pF",
        "Error_fF"
    ]].head(TOP_N).round(4).to_string(index=False)
)


# ============================================================
# 10. SCATTER + RESIDUAL PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

normal = ~analysis["Outlier"]
out = analysis["Outlier"]

ax = axes[0]

ax.scatter(
    analysis.loc[normal, "C_actual_pF"],
    analysis.loc[normal, "C_pred_pF"],
    s=20,
    alpha=0.4,
    label="Normal"
)

ax.scatter(
    analysis.loc[out, "C_actual_pF"],
    analysis.loc[out, "C_pred_pF"],
    s=50,
    marker="x",
    label="Outlier"
)

lo = min(
    analysis["C_actual_pF"].min(),
    analysis["C_pred_pF"].min()
)

hi = max(
    analysis["C_actual_pF"].max(),
    analysis["C_pred_pF"].max()
)

ax.plot([lo, hi], [lo, hi], "k--")

for _, r in outliers.head(15).iterrows():
    label = str(int(r["Row"]))

    if RUN_HOLDOUT:
        label += f"/F{int(r['Fold'])}"

    ax.annotate(
        label,
        (r["C_actual_pF"], r["C_pred_pF"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel("Actual C [pF]")
ax.set_ylabel("Predicted C [pF]")
ax.set_title("SharedPath Individual: Actual vs Predicted")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]

ax.scatter(
    analysis.loc[normal, "C_actual_pF"],
    analysis.loc[normal, "Error_fF"],
    s=20,
    alpha=0.4
)

ax.scatter(
    analysis.loc[out, "C_actual_pF"],
    analysis.loc[out, "Error_fF"],
    s=50,
    marker="x"
)

ax.axhline(0, color="black", linestyle="--")
ax.axhline(threshold, color="red", linestyle=":")
ax.axhline(-threshold, color="red", linestyle=":")

ax.set_xlabel("Actual C [pF]")
ax.set_ylabel("Error [fF]")
ax.set_title("Prediction Residual")
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=180
)

plt.show()


# ============================================================
# 11. PIN ASSOCIATION
#
# Compare:
# P(Outlier | selected pin)
# P(Outlier | pin not selected)
#
# An enrichment is an association, not proof of
# an independently defective pin.
# ============================================================

analysis_rows = analysis["Row"].to_numpy(dtype=int)
out_flag = analysis["Outlier"].to_numpy(dtype=bool)
abs_err = analysis["AbsError_fF"].to_numpy()

pin_rows = []

for polarity, matrix in [
    ("High", H),
    ("Low", L)
]:
    for pin in range(1, 45):
        selected = (
            matrix[analysis_rows, pin-1] > 0
        )

        n_sel = selected.sum()
        n_other = (~selected).sum()

        if n_sel == 0 or n_other == 0:
            continue

        rate_sel = out_flag[selected].mean()
        rate_other = out_flag[~selected].mean()

        pin_rows.append({
            "Polarity": polarity,
            "Pin": pin,
            "SMU": PIN_SMU[pin],
            "SelectedCount": int(n_sel),
            "OutlierCount": int(
                np.sum(out_flag & selected)
            ),
            "OutlierRate": rate_sel,
            "OtherRate": rate_other,
            "Enrichment": rate_sel - rate_other,
            "MeanAbsError_fF": abs_err[selected].mean()
        })

pin_stats = pd.DataFrame(pin_rows)

print("\n=== HIGH-RISK PIN ASSOCIATIONS ===")
print(
    pin_stats[
        pin_stats["SelectedCount"] >= 5
    ].sort_values(
        ["Enrichment", "OutlierCount"],
        ascending=False
    ).head(20).round(4).to_string(index=False)
)


# ============================================================
# 12. H-L PAIR ASSOCIATION
#
# Each observation contains multiple H-L pairs.
# Check which pairs are overrepresented in outliers.
# ============================================================

pair_records = []

for row, outlier, error in zip(
    analysis_rows,
    out_flag,
    abs_err
):
    for hp in HIGH[row]:
        for lp in LOW[row]:
            pair_records.append({
                "Row": int(row),
                "HighPin": hp,
                "LowPin": lp,
                "HighSMU": PIN_SMU[hp],
                "LowSMU": PIN_SMU[lp],
                "Outlier": bool(outlier),
                "AbsError_fF": float(error)
            })

pair_data = pd.DataFrame(pair_records)

pair_stats = (
    pair_data.groupby(["HighPin", "LowPin"])
    .agg(
        Count=("Outlier", "size"),
        OutlierCount=("Outlier", "sum"),
        OutlierRate=("Outlier", "mean"),
        MeanAbsError_fF=("AbsError_fF", "mean")
    )
    .reset_index()
)

pair_stats["GlobalOutlierRate"] = out_flag.mean()

pair_stats["RateExcess"] = (
    pair_stats["OutlierRate"]
    - pair_stats["GlobalOutlierRate"]
)

print("\n=== H-L PAIRS ASSOCIATED WITH OUTLIERS ===")
print(
    pair_stats[
        pair_stats["Count"] >= 5
    ].sort_values(
        ["RateExcess", "OutlierCount"],
        ascending=False
    ).head(25).round(4).to_string(index=False)
)


# ============================================================
# 13. SMU COMBINATION ASSOCIATION
# ============================================================

smu_rows = []

for row, outlier, error in zip(
    analysis_rows,
    out_flag,
    abs_err
):
    active_h = sorted({
        PIN_SMU[p] for p in HIGH[row]
    })

    active_l = sorted({
        PIN_SMU[p] for p in LOW[row]
    })

    for sh in active_h:
        for sl in active_l:
            smu_rows.append({
                "HighSMU": sh,
                "LowSMU": sl,
                "Outlier": bool(outlier),
                "AbsError_fF": float(error)
            })

smu_data = pd.DataFrame(smu_rows)

smu_stats = (
    smu_data.groupby(["HighSMU", "LowSMU"])
    .agg(
        Count=("Outlier", "size"),
        OutlierCount=("Outlier", "sum"),
        OutlierRate=("Outlier", "mean"),
        MeanAbsError_fF=("AbsError_fF", "mean")
    )
    .reset_index()
)

smu_stats["RateExcess"] = (
    smu_stats["OutlierRate"] - out_flag.mean()
)

print("\n=== SMU PAIR ASSOCIATIONS ===")
print(
    smu_stats[
        smu_stats["Count"] >= 5
    ].sort_values(
        "RateExcess", ascending=False
    ).round(4).to_string(index=False)
)


# ============================================================
# 14. REPEATED CONFIGURATION ANALYSIS
#
# Uses original measured data, not duplicated
# holdout fold records.
# ============================================================

measurement_df = pd.DataFrame({
    "Config": CONFIG,
    "High": HIGH,
    "Low": LOW,
    "C_actual_pF": C_TRUE
})

repeat_stats = (
    measurement_df.groupby("Config")
    .agg(
        N=("C_actual_pF", "size"),
        MeanC_pF=("C_actual_pF", "mean"),
        StdC_fF=(
            "C_actual_pF",
            lambda x: x.std(ddof=1) * 1000
        )
    )
    .reset_index()
)

# Average residual per configuration, using
# whichever prediction set is being analyzed.
config_error = (
    analysis.groupby("Config")
    .agg(
        MeanError_fF=("Error_fF", "mean"),
        MeanAbsError_fF=("AbsError_fF", "mean"),
        MaxAbsError_fF=("AbsError_fF", "max")
    )
    .reset_index()
)

repeat_stats = repeat_stats.merge(
    config_error, on="Config", how="left"
)

print("\n=== REPEATED CONFIGURATION DIAGNOSTICS ===")
print(
    repeat_stats[
        repeat_stats["N"] >= 2
    ].sort_values(
        "MeanAbsError_fF", ascending=False
    ).head(20).round(3).to_string(index=False)
)


# ============================================================
# 15. EXPORT
# ============================================================

analysis.to_csv(
    OUTPUT_DIR / "all_predictions.csv",
    index=False
)

outliers.to_csv(
    OUTPUT_DIR / "outlier_pin_combinations.csv",
    index=False
)

pin_stats.to_csv(
    OUTPUT_DIR / "pin_associations.csv",
    index=False
)

pair_stats.to_csv(
    OUTPUT_DIR / "pair_associations.csv",
    index=False
)

smu_stats.to_csv(
    OUTPUT_DIR / "smu_associations.csv",
    index=False
)

repeat_stats.to_csv(
    OUTPUT_DIR / "repeatability.csv",
    index=False
)

print("\nSaved to:", OUTPUT_DIR.resolve())
