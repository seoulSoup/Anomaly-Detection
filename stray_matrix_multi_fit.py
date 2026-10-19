
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PIN SERIES IMPEDANCE VALIDATION
#
# 필요 변수:
#   df: 기존 Multi-Pin 데이터
#   results["z_pin"]: Pair Matrix에서 추정한 44개 z_i [Ohm]
#
# df["C"], df["C_local_pred"]: Farad
# 측정 주파수: 10 kHz
# ============================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

z_pin = np.asarray(
    results["z_pin"], dtype=complex
)

data = df.copy()

required = [
    "high_pins",
    "low_pins",
    "signature",
    "C",
    "C_local_pred"
]

for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

if len(z_pin) != 44:
    raise ValueError("z_pin must have 44 elements.")

# ------------------------------------------------------------
# 1. Pin별 선로 임피던스 합성
# ------------------------------------------------------------

def aggregate_line_admittance(high, low):
    h = np.array(high, dtype=int) - 1
    l = np.array(low, dtype=int) - 1

    if len(h) == 0 or len(l) == 0:
        return np.nan + 1j * np.nan

    zh = z_pin[h]
    zl = z_pin[l]

    YH = np.sum(1.0 / zh)
    YL = np.sum(1.0 / zl)

    if abs(YH) < 1e-30 or abs(YL) < 1e-30:
        return np.nan + 1j * np.nan

    ZH = 1.0 / YH
    ZL = 1.0 / YL

    if abs(ZH + ZL) < 1e-30:
        return np.nan + 1j * np.nan

    return 1.0 / (ZH + ZL)


y_line = np.array([
    aggregate_line_admittance(h, l)
    for h, l in zip(
        data["high_pins"],
        data["low_pins"]
    )
])

# ------------------------------------------------------------
# 2. 어드미턴스 -> 등가 정전용량
# ------------------------------------------------------------

data["C_line"] = y_line.imag / OMEGA

data["C_local_corrected"] = (
    data["C"] - data["C_local_pred"]
)

data["C_hybrid_corrected"] = (
    data["C"]
    - data["C_local_pred"]
    - data["C_line"]
)

# ------------------------------------------------------------
# 3. 동일 SMU 점유 구성 내에서 중심화
#
# SMU 기본 성분이 같은 그룹 내에서는
# 차분 시 소거된다는 가정을 검증한다.
# ------------------------------------------------------------

size = data.groupby("signature")["C"].transform("size")
valid = data[(size >= 2)].copy()

columns = {
    "Raw": "C",
    "Local only": "C_local_corrected",
    "Local + Line": "C_hybrid_corrected"
}

print("=" * 65)
print("WITHIN-SIGNATURE MODEL COMPARISON")
print("=" * 65)

print("Number of experiments:", len(valid))
print("Number of signatures:",
      valid["signature"].nunique())

residuals = {}

for name, col in columns.items():
    centered = (
        valid[col]
        - valid.groupby("signature")[col].transform("mean")
    )

    r = centered.to_numpy() * 1e15  # F -> fF

    residuals[name] = r

    mae = np.mean(np.abs(r))
    rmse = np.sqrt(np.mean(r**2))
    maxae = np.max(np.abs(r))

    print(
        f"{name:<18}"
        f"MAE={mae:9.3f} fF  "
        f"RMSE={rmse:9.3f} fF  "
        f"MaxAE={maxae:9.3f} fF"
    )

# ------------------------------------------------------------
# 4. 특히 Local 변화가 작은 실험에서 검증
#
# 동일 signature 내에서 C_local의 평균 대비
# 차이가 0.1 pF 미만인 측정만 선별.
# ------------------------------------------------------------

local_center = (
    valid["C_local_pred"]
    - valid.groupby("signature")["C_local_pred"].transform("mean")
)

low_local_mask = (
    np.abs(local_center.to_numpy()) < 0.1e-12
)

print("\n" + "=" * 65)
print("LOW-LOCAL-VARIATION REGION")
print("=" * 65)

print("Number of points:", np.sum(low_local_mask))

for name, r in residuals.items():
    rr = r[low_local_mask]

    if len(rr) == 0:
        continue

    print(
        f"{name:<18}"
        f"RMSE={np.sqrt(np.mean(rr**2)):9.3f} fF"
    )

# ------------------------------------------------------------
# 5. Pin Line 성분이 실제 Local 제거 후
#    잔차를 설명하는지 Scatter Plot
# ------------------------------------------------------------

target = (
    valid["C_local_corrected"]
    - valid.groupby("signature")[
        "C_local_corrected"
    ].transform("mean")
) * 1e12

line_pred = (
    valid["C_line"]
    - valid.groupby("signature")[
        "C_line"
    ].transform("mean")
) * 1e12

x = line_pred.to_numpy()
y = target.to_numpy()

mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

plt.figure(figsize=(6, 6))
plt.scatter(x, y, alpha=0.6)

lo = min(x.min(), y.min(), 0)
hi = max(x.max(), y.max(), 0)

plt.plot(
    [lo, hi], [lo, hi],
    "--", label="Ideal y=x"
)

plt.xlabel("Predicted line delta C (pF)")
plt.ylabel("Observed residual delta C (pF)")
plt.title("Pin Series Impedance Validation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

if len(x) > 1 and np.std(x) > 0:
    slope, intercept = np.polyfit(x, y, 1)
    corr = (
        np.corrcoef(x, y)[0, 1]
        if np.std(y) > 0 else np.nan
    )

    print("\nLine prediction slope:", slope)
    print("Line prediction intercept (pF):", intercept)
    print("Line prediction correlation:", corr)

# ------------------------------------------------------------
# 6. 다음 분석을 위해 저장
# ------------------------------------------------------------

data.to_csv(
    "multipin_hybrid_validation.csv",
    index=False
)
