
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. FIXED MODEL
# ============================================================

IMP_MODE = "Parallel"
PAIR_MODE = "Sum"

all_idx = np.arange(len(Y))

# Stage 1: common Pin + cross-SMU pair coefficient
t_fit = fit_common(
    all_idx, IMP_MODE, PAIR_MODE
)

# Stage 2: individual Pin deviations
# Stage 1 parameters remain fixed.
v_fit = fit_individual(
    all_idx, t_fit, IMP_MODE, PAIR_MODE
)

# ============================================================
# 2. DECOMPOSE PREDICTION
# ============================================================

common_c, cross_pf, shared = decode_common(
    t_fit, IMP_MODE
)

c_pin = pin_coefficients(
    t_fit, v_fit, IMP_MODE
)

# Parallel impedance aggregation
ch = X["H"] @ c_pin
cl = X["L"] @ c_pin

c_line = np.divide(
    ch * cl,
    ch + cl,
    out=np.zeros_like(ch),
    where=(ch + cl) > 0
)

# Pair Sum aggregation
coeff = np.r_[LOCAL, cross_pf]

zone_contribution = (
    X["counts"] * coeff[None, :]
)

c_pair = zone_contribution.sum(axis=1)

c_pred = c_line + c_pair

# Actual measurements (pF)
c_actual = Y.copy()

error_ff = (c_pred - c_actual) * 1000
abs_error_ff = np.abs(error_ff)

# ============================================================
# 3. TOTAL ACCURACY
# ============================================================

rmse = np.sqrt(np.mean(error_ff**2))
mae = np.mean(abs_error_ff)
median_ae = np.median(abs_error_ff)
p95 = np.percentile(abs_error_ff, 95)
bias = np.mean(error_ff)

print("=" * 65)
print("PARALLEL + PAIR SUM + INDIVIDUAL PIN")
print("=" * 65)

print(f"Common Pin C  : {common_c:.6f} pF")
print(f"Cross Pair C  : {cross_pf:.6f} pF")
print(f"Pin C min/max : {c_pin.min():.4f} / "
      f"{c_pin.max():.4f} pF")

print("\nFULL-DATA FIT ACCURACY")
print(f"RMSE         : {rmse:.2f} fF")
print(f"MAE          : {mae:.2f} fF")
print(f"Median AE    : {median_ae:.2f} fF")
print(f"P95 AE       : {p95:.2f} fF")
print(f"Bias         : {bias:+.2f} fF")

# ============================================================
# 4. RESULT DATAFRAME
# ============================================================

results = pd.DataFrame({
    "High": data["high_pins"],
    "Low": data["low_pins"],
    "H_count": X["H"].sum(axis=1).astype(int),
    "L_count": X["L"].sum(axis=1).astype(int),
    "Signature": X["signature"],

    "C_actual_pF": c_actual,
    "C_pred_pF": c_pred,

    "C_line_pF": c_line,
    "C_pair_pF": c_pair,

    "C_even_pF": zone_contribution[:, 0],
    "C_odd_pF": zone_contribution[:, 1],
    "C_dist2_pF": zone_contribution[:, 2],
    "C_cross_pF": zone_contribution[:, 3],

    "Error_fF": error_ff,
    "AbsError_fF": abs_error_ff
})

results["Error_pct"] = (
    100 * (c_pred - c_actual)
    / np.where(
        np.abs(c_actual) > 1e-12,
        c_actual,
        np.nan
    )
)

# ============================================================
# 5. WORST MEASUREMENTS
# ============================================================

cols = [
    "High", "Low",
    "H_count", "L_count",
    "C_actual_pF", "C_pred_pF",
    "C_line_pF", "C_pair_pF",
    "Error_fF", "Error_pct"
]

print("\n" + "=" * 65)
print("WORST 20 CONFIGURATIONS")
print("=" * 65)

print(
    results.sort_values(
        "AbsError_fF", ascending=False
    )[cols]
    .head(20)
    .round(4)
    .to_string(index=False)
)

# ============================================================
# 6. SIGNATURE ERROR
# ============================================================

signature_stats = (
    results.groupby("Signature")
    .agg(
        N=("Error_fF", "size"),
        H_count=("H_count", "first"),
        L_count=("L_count", "first"),
        RMSE_fF=(
            "Error_fF",
            lambda x: np.sqrt(np.mean(x**2))
        ),
        MAE_fF=("AbsError_fF", "mean"),
        Bias_fF=("Error_fF", "mean")
    )
    .reset_index()
    .sort_values("RMSE_fF", ascending=False)
)

print("\n" + "=" * 65)
print("WORST 15 SIGNATURES")
print("=" * 65)

print(
    signature_stats.head(15)
    .round(2)
    .to_string(index=False)
)

# ============================================================
# 7. PIN IMPEDANCES
# ============================================================

pin_results = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "SMU": [
        PIN_SMU[p] + 1 for p in range(1, 45)
    ],
    "C_pin_pF": c_pin,
    "AbsZ_MOhm": 1 / (
        OMEGA * c_pin * 1e-12
    ) / 1e6
})

print("\n" + "=" * 65)
print("INDIVIDUAL PIN PARAMETERS")
print("=" * 65)

print(
    pin_results.round(4).to_string(index=False)
)

# ============================================================
# 8. ACTUAL VS PREDICTED
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    c_actual,
    c_pred,
    alpha=0.65,
    s=24
)

lo = min(c_actual.min(), c_pred.min())
hi = max(c_actual.max(), c_pred.max())

ax.plot(
    [lo, hi], [lo, hi],
    "k--", linewidth=1
)

ax.set_xlabel("Measured C (pF)")
ax.set_ylabel("Predicted C (pF)")
ax.set_title(
    f"Absolute C: RMSE={rmse:.1f} fF"
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 9. RESIDUAL VS PIN COUNTS
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

sc = ax.scatter(
    results["H_count"],
    results["Error_fF"],
    c=results["L_count"],
    alpha=0.7
)

ax.axhline(0, color="black", linestyle="--")

ax.set_xlabel("High Pin Count")
ax.set_ylabel("Prediction - Actual (fF)")
ax.set_title("Residual vs High Count")
ax.grid(True, alpha=0.3)

fig.colorbar(
    sc, ax=ax, label="Low Pin Count"
)

plt.tight_layout()
plt.show()

# ============================================================
# 10. COMPONENTS VS ACTUAL
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

order = np.argsort(c_actual)

ax.plot(
    c_actual[order],
    label="Measured C",
    linewidth=2
)

ax.plot(
    c_line[order],
    label="Impedance contribution",
    alpha=0.85
)

ax.plot(
    c_pair[order],
    label="Pair increment",
    alpha=0.85
)

ax.plot(
    c_pred[order],
    label="Total predicted C",
    linestyle="--"
)

ax.set_xlabel("Measurements sorted by actual C")
ax.set_ylabel("Capacitance (pF)")
ax.set_title("Contribution Decomposition")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# 11. SAVE
# ============================================================

results.to_csv(
    "parallel_sum_actual_vs_pred.csv",
    index=False,
    encoding="utf-8-sig"
)

signature_stats.to_csv(
    "parallel_sum_signature_errors.csv",
    index=False,
    encoding="utf-8-sig"
)

pin_results.to_csv(
    "parallel_sum_pin_impedances.csv",
    index=False
)

print("\nSaved result CSV files.")







import numpy as np

# ==========================================================
# Independent Capacitance Prediction
#
# Required fitted parameters:
#   c_pin   : array of 44 effective pin C values [pF]
#   cross_pf: fitted cross-SMU pair coefficient [pF]
#
# Input:
#   high: iterable of pin numbers (1~44)
#   low : iterable of pin numbers (1~44)
#
# Output:
#   Predicted capacitance [pF]
# ==========================================================

def predict_cap(high, low):
    high = tuple(int(p) for p in high)
    low = tuple(int(p) for p in low)

    if not high or not low:
        raise ValueError("High/Low cannot be empty")

    if len(set(high)) != len(high):
        raise ValueError("Duplicate High pins")

    if len(set(low)) != len(low):
        raise ValueError("Duplicate Low pins")

    if set(high) & set(low):
        raise ValueError("High and Low overlap")

    if any(p < 1 or p > 44 for p in high + low):
        raise ValueError("Pin must be between 1 and 44")

    # 1. Parallel aggregation on each side,
    #    followed by series combination.
    ch = sum(c_pin[p - 1] for p in high)
    cl = sum(c_pin[p - 1] for p in low)

    c_line = ch * cl / (ch + cl)

    # 2. Pair Sum correction.
    ne = no = n2 = ncross = 0

    for h in high:
        for l in low:
            if PIN_SMU[h] != PIN_SMU[l]:
                ncross += 1
                continue

            d = abs(h - l)

            if d == 1:
                if max(h, l) % 2 == 0:
                    ne += 1
                else:
                    no += 1
            elif d == 2:
                n2 += 1

    c_pair = (
        1.544 * ne
        + 0.605 * no
        + 0.175 * n2
        + cross_pf * ncross
    )

    return float(c_line + c_pair)


# ==========================================================
# Example
# ==========================================================

high = list(range(15, 45))
low = list(range(7, 14))

c_pred_pf = predict_cap(high, low)

print(f"Predicted C = {c_pred_pf:.6f} pF")

