
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PIN 7 / 14 SWAP REPRODUCIBILITY TEST
#
# Low A: 7~13 (Pin 7 included)
# Low B: 8~14 (Pin 14 included)
#
# Input:
#   df["high_pins"]
#   df["low_pins"]
#   df["C"]             [F]
#   df["C_local_pred"]  [F]
#
# No z_pin, a_pin, or impedance fitting.
# ============================================================

LOW_A = tuple(range(7, 14))
LOW_B = tuple(range(8, 15))

def parse_pins(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        pins = [int(v) for v in value]
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                pins = [int(v) for v in parsed]
            else:
                pins = [
                    int(v) for v in re.findall(r"\d+", value)
                ]
        except (ValueError, SyntaxError):
            pins = [
                int(v) for v in re.findall(r"\d+", value)
            ]
    else:
        raise ValueError(f"Invalid Pin format: {value}")

    if len(pins) != len(set(pins)):
        raise ValueError(f"Duplicate pins: {value}")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError(f"Pin out of range: {value}")

    return tuple(sorted(pins))


# ============================================================
# 1. PREPARE DATA
# ============================================================

data = df.copy()

required = [
    "high_pins",
    "low_pins",
    "C",
    "C_local_pred"
]

for name in required:
    if name not in data.columns:
        raise ValueError(f"Missing column: {name}")

data["high_key"] = data["high_pins"].apply(parse_pins)
data["low_key"] = data["low_pins"].apply(parse_pins)

data["C"] = pd.to_numeric(data["C"])
data["C_local_pred"] = pd.to_numeric(
    data["C_local_pred"]
)

if data["C"].abs().max() > 1e-9:
    raise ValueError("C is not in Farads.")

if data["C_local_pred"].abs().max() > 1e-9:
    raise ValueError("C_local_pred is not in Farads.")

data["C_base"] = (
    data["C"] - data["C_local_pred"]
)

# Only Low A and B
data = data[
    data["low_key"].isin([LOW_A, LOW_B])
].copy()

data["low_type"] = np.where(
    data["low_key"] == LOW_A,
    "A", "B"
)

# ============================================================
# 2. MATCH IDENTICAL HIGH CONFIGURATIONS
# ============================================================

# 동일 조건 반복 측정은 평균으로 통합
grouped = (
    data.groupby(
        ["high_key", "low_type"],
        as_index=False
    )
    .agg(
        C_meas=("C", "mean"),
        C_local=("C_local_pred", "mean"),
        C_base=("C_base", "mean"),
        repetitions=("C", "size")
    )
)

wide = grouped.pivot(
    index="high_key",
    columns="low_type",
    values=[
        "C_meas",
        "C_local",
        "C_base",
        "repetitions"
    ]
)

# A와 B가 모두 있는 High만
wide = wide.dropna(
    subset=[
        ("C_base", "A"),
        ("C_base", "B")
    ]
)

if len(wide) < 3:
    raise ValueError(
        "Need at least 3 matched High configurations."
    )

pairs = pd.DataFrame({
    "High": list(wide.index),
    "C_A": wide[("C_meas", "A")].to_numpy(),
    "C_B": wide[("C_meas", "B")].to_numpy(),
    "Local_A": wide[("C_local", "A")].to_numpy(),
    "Local_B": wide[("C_local", "B")].to_numpy(),
    "Base_A": wide[("C_base", "A")].to_numpy(),
    "Base_B": wide[("C_base", "B")].to_numpy()
})

pairs["H_count"] = pairs["High"].apply(len)

# SMU1의 Pin 1,6 포함 여부도 기록
pairs["H_has_1"] = pairs["High"].apply(
    lambda x: 1 in x
)
pairs["H_has_6"] = pairs["High"].apply(
    lambda x: 6 in x
)

# 모든 차이는 B - A
pairs["delta_meas_fF"] = (
    pairs["C_B"] - pairs["C_A"]
) * 1e15

pairs["delta_local_fF"] = (
    pairs["Local_B"] - pairs["Local_A"]
) * 1e15

pairs["D_base_fF"] = (
    pairs["Base_B"] - pairs["Base_A"]
) * 1e15

# ============================================================
# 3. LEAVE-ONE-HIGH-OUT CROSS VALIDATION
#
# 매번 하나의 High 조합을 제외하고
# 나머지 조합의 평균으로 beta 추정.
# ============================================================

d = pairs["D_base_fF"].to_numpy(dtype=float)
n = len(d)

pred_constant = np.empty(n)
pred_zero = np.zeros(n)

for k in range(n):
    train_mask = np.ones(n, dtype=bool)
    train_mask[k] = False

    # 고정 Pin 교체 효과
    pred_constant[k] = d[train_mask].mean()

pairs["pred_constant_fF"] = pred_constant

pairs["error_constant_fF"] = (
    pairs["D_base_fF"]
    - pairs["pred_constant_fF"]
)

pairs["error_zero_fF"] = pairs["D_base_fF"]

# ============================================================
# 4. METRICS
# ============================================================

def metrics(err):
    err = np.asarray(err, dtype=float)
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MedianAE": np.median(np.abs(err)),
        "MaxAE": np.max(np.abs(err))
    }

zero_score = metrics(pairs["error_zero_fF"])
fixed_score = metrics(pairs["error_constant_fF"])

print("\n" + "=" * 68)
print("PIN 7 / 14 SWAP REPRODUCIBILITY")
print("=" * 68)

print("Matched High configurations:", n)

print(f"Mean D_base  : {d.mean():+.3f} fF")
print(f"Std D_base   : {d.std():.3f} fF")
print(f"Min D_base   : {d.min():+.3f} fF")
print(f"Max D_base   : {d.max():+.3f} fF")

print("\nLOOCV comparison")
print(
    f"{'Model':<20}"
    f"{'MAE':>10}"
    f"{'RMSE':>10}"
    f"{'MedianAE':>12}"
    f"{'MaxAE':>10}"
)

for name, score in [
    ("No swap correction", zero_score),
    ("Constant swap", fixed_score)
]:
    print(
        f"{name:<20}"
        f"{score['MAE']:10.2f}"
        f"{score['RMSE']:10.2f}"
        f"{score['MedianAE']:12.2f}"
        f"{score['MaxAE']:10.2f}"
    )

improvement = (
    1 - fixed_score["RMSE"] / zero_score["RMSE"]
) * 100 if zero_score["RMSE"] > 0 else np.nan

print(f"\nRMSE reduction: {improvement:.2f}%")

# ============================================================
# 5. CHECK HIGH-CONFIGURATION DEPENDENCE
#
# 진단용: LOOCV 오차가 High 개수 및
# Pin 1,6 선택과 연관되는지 확인
# ============================================================

print("\n" + "=" * 68)
print("PER-HIGH VALIDATION RESULTS")
print("=" * 68)

print(
    pairs[[
        "High",
        "H_count",
        "H_has_1",
        "H_has_6",
        "delta_meas_fF",
        "delta_local_fF",
        "D_base_fF",
        "pred_constant_fF",
        "error_constant_fF"
    ]]
    .sort_values("error_constant_fF",
                 key=lambda s: s.abs(),
                 ascending=False)
    .round(2)
    .to_string(index=False)
)

# ============================================================
# 6. PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 고정 효과 가설: 예측 vs 실측
axes[0].scatter(
    pred_constant, d, alpha=0.75
)

lo = min(pred_constant.min(), d.min())
hi = max(pred_constant.max(), d.max())

axes[0].plot([lo, hi], [lo, hi], "--")

axes[0].set_xlabel(
    "LOOCV predicted swap effect (fF)"
)
axes[0].set_ylabel(
    "Measured swap effect after local correction (fF)"
)
axes[0].set_title("Constant Pin-Swap Hypothesis")
axes[0].grid(True)

# H 개수와 고정 효과 모델의 잔차
axes[1].scatter(
    pairs["H_count"],
    pairs["error_constant_fF"],
    alpha=0.75
)

axes[1].axhline(0, linestyle="--")

axes[1].set_xlabel("High Pin count")
axes[1].set_ylabel("LOOCV residual (fF)")
axes[1].set_title("Residual vs High Configuration")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# ============================================================
# 7. EXPORT
# ============================================================

pairs.to_csv(
    "pin7_14_swap_validation.csv",
    index=False
)

swap_validation = pairs.copy()
