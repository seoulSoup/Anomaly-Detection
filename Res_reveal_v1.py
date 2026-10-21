
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. FULL DATA SHARED PATH + INDIVIDUAL FIT
# ============================================================

all_idx = np.arange(len(C_TRUE))

# From the previous Joint experiment
base_theta_full, opt = fit_base_joint(all_idx)

y_pred = predict_base(all_idx, base_theta_full)

c_pred = y_pred.imag / B_PER_PF
g_pred = y_pred.real

error_fF = (c_pred - C_TRUE) * 1000
abs_error_fF = np.abs(error_fF)

# Outlier cutoff: larger of 95th percentile
# and 3 x median absolute deviation around median.
med = np.median(error_fF)
mad = np.median(np.abs(error_fF - med))
robust_sigma = 1.4826 * mad

OUTLIER_THRESHOLD_FF = max(
    np.percentile(abs_error_fF, 95),
    3.0 * robust_sigma
)

is_outlier = abs_error_fF >= OUTLIER_THRESHOLD_FF

print(f"Outlier threshold : {OUTLIER_THRESHOLD_FF:.2f} fF")
print(f"Outlier count     : {is_outlier.sum()} / {len(C_TRUE)}")


# ============================================================
# 2. PIN LIST CONVERSION
# ============================================================

def pin_list(row):
    return list(np.flatnonzero(row > 0) + 1)

high_lists = [pin_list(row) for row in H]
low_lists = [pin_list(row) for row in L]

result = pd.DataFrame({
    "Row": np.arange(len(C_TRUE)),
    "High": high_lists,
    "Low": low_lists,
    "Signature": SIGNATURE,
    "C_actual_pF": C_TRUE,
    "C_pred_pF": c_pred,
    "Error_fF": error_fF,
    "AbsError_fF": abs_error_fF,
    "G_actual_nS": G_TRUE,
    "G_pred_nS": g_pred,
    "Outlier": is_outlier
})

outliers = (
    result[result["Outlier"]]
    .sort_values("AbsError_fF", ascending=False)
)

print("\n=== TOP 30 OUTLIER PIN COMBINATIONS ===")
print(
    outliers.head(30)[[
        "Row", "High", "Low", "Signature",
        "C_actual_pF", "C_pred_pF",
        "Error_fF"
    ]].to_string(index=False)
)


# ============================================================
# 3. SCATTER PLOT WITH OUTLIER LABELS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]

ax.scatter(
    C_TRUE[~is_outlier],
    c_pred[~is_outlier],
    alpha=0.4,
    s=20,
    label="Normal"
)

ax.scatter(
    C_TRUE[is_outlier],
    c_pred[is_outlier],
    s=45,
    marker="x",
    label="Outlier"
)

lo = min(C_TRUE.min(), c_pred.min())
hi = max(C_TRUE.max(), c_pred.max())

ax.plot([lo, hi], [lo, hi], "k--", lw=1)

# Label only 15 largest outliers to avoid clutter.
top_idx = np.argsort(abs_error_fF)[-15:]

for k in top_idx:
    ax.annotate(
        str(k),
        (C_TRUE[k], c_pred[k]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

ax.set_xlabel("Actual Open C (pF)")
ax.set_ylabel("Predicted Open C (pF)")
ax.set_title("SharedPath Individual: Outlier Identification")
ax.legend()
ax.grid(alpha=0.3)

# Residual vs Actual
ax = axes[1]

ax.scatter(
    C_TRUE[~is_outlier],
    error_fF[~is_outlier],
    alpha=0.4,
    s=20
)

ax.scatter(
    C_TRUE[is_outlier],
    error_fF[is_outlier],
    s=45,
    marker="x"
)

ax.axhline(0, color="black", ls="--")
ax.axhline(
    OUTLIER_THRESHOLD_FF, color="red", ls=":"
)
ax.axhline(
    -OUTLIER_THRESHOLD_FF, color="red", ls=":"
)

ax.set_xlabel("Actual Open C (pF)")
ax.set_ylabel("Prediction Error (fF)")
ax.set_title("Residual Distribution")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# 4. PIN FREQUENCY ENRICHMENT
#
# Compare:
# P(outlier | pin selected)
# vs
# P(outlier | pin not selected)
#
# Calculate separately for High and Low.
# ============================================================

pin_rows = []

for polarity, matrix in [("High", H), ("Low", L)]:
    for pin in range(1, 45):
        selected = matrix[:, pin - 1] > 0

        n_selected = int(selected.sum())
        n_not = int((~selected).sum())

        if n_selected == 0 or n_not == 0:
            continue

        rate_selected = is_outlier[selected].mean()
        rate_not = is_outlier[~selected].mean()

        pin_rows.append({
            "Polarity": polarity,
            "Pin": pin,
            "Count": n_selected,
            "OutlierCount": int(
                np.sum(is_outlier & selected)
            ),
            "OutlierRate": rate_selected,
            "OtherRate": rate_not,
            "RateDifference": rate_selected - rate_not,
            "MeanAbsError_fF": abs_error_fF[selected].mean()
        })

pin_stats = pd.DataFrame(pin_rows)

print("\n=== PINS ASSOCIATED WITH OUTLIERS ===")
print(
    pin_stats[
        pin_stats["Count"] >= 5
    ].sort_values(
        ["RateDifference", "OutlierCount"],
        ascending=False
    ).head(20).round(4).to_string(index=False)
)


# ============================================================
# 5. SMU ASSOCIATION
# ============================================================

smu_rows = []

for g, smu in enumerate(SMUS):
    cols = np.array(list(smu)) - 1

    for polarity, matrix in [("High", H), ("Low", L)]:
        selected = matrix[:, cols].sum(axis=1) > 0

        if selected.all() or (~selected).all():
            continue

        smu_rows.append({
            "SMU": g + 1,
            "Polarity": polarity,
            "SelectedCount": int(selected.sum()),
            "OutlierRate": is_outlier[selected].mean(),
            "OtherRate": is_outlier[~selected].mean(),
            "MeanAbsError_fF": abs_error_fF[selected].mean()
        })

smu_stats = pd.DataFrame(smu_rows)

smu_stats["RateDifference"] = (
    smu_stats["OutlierRate"]
    - smu_stats["OtherRate"]
)

print("\n=== SMU ASSOCIATION ===")
print(
    smu_stats.sort_values(
        "RateDifference", ascending=False
    ).round(4).to_string(index=False)
)


# ============================================================
# 6. CONFIGURATION REPEATABILITY
#
# Distinguish genuine configuration-dependent error
# from unstable measurement noise.
# ============================================================

repeat_stats = (
    result.assign(
        Config=CONFIG
    )
    .groupby("Config")
    .agg(
        N=("C_actual_pF", "size"),
        ActualMean_pF=("C_actual_pF", "mean"),
        ActualStd_fF=(
            "C_actual_pF",
            lambda x: x.std(ddof=1) * 1000
        ),
        MeanError_fF=("Error_fF", "mean"),
        MeanAbsError_fF=("AbsError_fF", "mean"),
        OutlierCount=("Outlier", "sum")
    )
    .reset_index()
)

print("\n=== REPEATED CONFIGURATION OUTLIERS ===")
print(
    repeat_stats[
        repeat_stats["N"] >= 2
    ].sort_values(
        "MeanAbsError_fF",
        ascending=False
    ).head(20).round(3).to_string(index=False)
)


# ============================================================
# 7. SAVE REPORTS
# ============================================================

result.to_csv(
    "sharedpath_outlier_rows.csv", index=False
)

outliers.to_csv(
    "sharedpath_outlier_combinations.csv", index=False
)

pin_stats.to_csv(
    "sharedpath_outlier_pin_stats.csv", index=False
)

smu_stats.to_csv(
    "sharedpath_outlier_smu_stats.csv", index=False
)

repeat_stats.to_csv(
    "sharedpath_repeatability.csv", index=False
)
