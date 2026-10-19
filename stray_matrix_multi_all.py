
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# TEST: SMU Occupancy Invariance
# Input: df from previous experiment
# Units: C, C_local_pred = Farad
# ============================================================

data = df.copy()

required = ["C", "C_local_pred", "signature"]
missing = [x for x in required if x not in data.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

data["C"] = pd.to_numeric(data["C"])
data["C_local_pred"] = pd.to_numeric(
    data["C_local_pred"]
)

# 단위 오류를 방지하기 위한 검증
if data["C"].abs().max() > 1e-9:
    raise ValueError(
        "C seems not to be in Farads. Check units."
    )

if data["C_local_pred"].abs().max() > 1e-9:
    raise ValueError(
        "C_local_pred seems not to be in Farads."
    )

# ------------------------------------------------------------
# 1. Local 기생 성분 제거
# ------------------------------------------------------------

data["C_base"] = (
    data["C"] - data["C_local_pred"]
)

# 단위 변환: F -> pF
data["C_pF"] = data["C"] * 1e12
data["C_base_pF"] = data["C_base"] * 1e12

# ------------------------------------------------------------
# 2. 동일 SMU 점유 구성 내부의 잔차
# ------------------------------------------------------------

grouped = data.groupby("signature")

# 그룹별 평균을 각 행에 부여
data["C_group_mean"] = grouped["C"].transform("mean")
data["Base_group_mean"] = (
    grouped["C_base"].transform("mean")
)

# 원본 / Local 제거 후 그룹 내 편차
data["raw_residual_fF"] = (
    data["C"] - data["C_group_mean"]
) * 1e15

data["base_residual_fF"] = (
    data["C_base"] - data["Base_group_mean"]
) * 1e15

data["group_size"] = (
    grouped["C"].transform("size")
)

# 실제 비교가 가능한 그룹만 분석
valid = data[data["group_size"] >= 2].copy()

if valid.empty:
    raise ValueError(
        "At least one signature needs 2+ samples."
    )

def scores(values):
    values = np.asarray(values, dtype=float)
    return (
        np.mean(np.abs(values)),
        np.sqrt(np.mean(values**2)),
        np.max(np.abs(values))
    )

raw_mae, raw_rmse, raw_max = scores(
    valid["raw_residual_fF"]
)
base_mae, base_rmse, base_max = scores(
    valid["base_residual_fF"]
)

print("\n" + "=" * 65)
print("WITHIN-SIGNATURE RESIDUAL ANALYSIS")
print("=" * 65)

print("Total experiments:", len(data))
print("Total signatures:", data["signature"].nunique())
print("Repeated signatures:",
      valid["signature"].nunique())
print("Experiments in repeated groups:", len(valid))

print("\n                         RAW          CORRECTED")
print(f"MAE  (fF)       {raw_mae:12.3f} {base_mae:14.3f}")
print(f"RMSE (fF)       {raw_rmse:12.3f} {base_rmse:14.3f}")
print(f"MaxAE(fF)       {raw_max:12.3f} {base_max:14.3f}")

if raw_rmse > 0:
    improvement = 100 * (1 - base_rmse/raw_rmse)
    print(f"\nRMSE reduction: {improvement:.2f}%")

# ------------------------------------------------------------
# 3. SMU 점유 구성별 표준편차와 범위
# ------------------------------------------------------------

summary = (
    valid.groupby("signature")
    .agg(
        count=("C_base", "size"),
        raw_std=("C", "std"),
        base_std=("C_base", "std"),
        base_min=("C_base", "min"),
        base_max=("C_base", "max"),
        base_mean=("C_base", "mean")
    )
    .copy()
)

summary["raw_std_fF"] = summary["raw_std"] * 1e15
summary["base_std_fF"] = summary["base_std"] * 1e15

summary["base_range_fF"] = (
    summary["base_max"] - summary["base_min"]
) * 1e15

summary["base_mean_pF"] = (
    summary["base_mean"] * 1e12
)

summary = summary.sort_values(
    "base_std_fF",
    ascending=False
)

print("\n" + "=" * 65)
print("GROUP SUMMARY")
print("=" * 65)

print(summary[[
    "count",
    "raw_std_fF",
    "base_std_fF",
    "base_range_fF",
    "base_mean_pF"
]].to_string())

print("\nMedian corrected group std (fF):",
      summary["base_std_fF"].median())

# ------------------------------------------------------------
# 4. 그래프: 제거 전후 그룹 내부 잔차
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    valid["C_local_pred"] * 1e12,
    valid["raw_residual_fF"],
    alpha=0.6,
    label="Before local correction"
)

plt.scatter(
    valid["C_local_pred"] * 1e12,
    valid["base_residual_fF"],
    alpha=0.6,
    label="After local correction"
)

plt.axhline(0, linestyle="--")
plt.xlabel("Predicted local capacitance (pF)")
plt.ylabel("Deviation from group mean (fF)")
plt.title("Effect of Local Admittance Correction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. 추후 분석을 위해 저장
# ------------------------------------------------------------

data.to_csv(
    "smu_occupancy_corrected.csv",
    index=False
)

summary.to_csv(
    "smu_occupancy_group_summary.csv"
)
