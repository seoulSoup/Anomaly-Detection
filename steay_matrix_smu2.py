9
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

# =========================================================
# LOW A/B SWAP RESIDUAL ANALYSIS
#
# Low A = 7~13
# Low B = 8~14
#
# 기존 변수:
#   df
#   pin_fit_new["a_pin"]
#
# 단위:
#   C, C_local_pred : F
#   a_pin           : Siemens
# =========================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

LOW_A = tuple(range(7, 14))
LOW_B = tuple(range(8, 15))

a_pin = np.asarray(
    pin_fit_new["a_pin"],
    dtype=complex
)

if a_pin.shape != (44,):
    raise ValueError("a_pin must contain 44 values.")


def parse_pins(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(sorted(int(x) for x in value))

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple, set)):
                return tuple(sorted(int(x) for x in parsed))
        except (ValueError, SyntaxError):
            pass

        return tuple(sorted(
            int(x) for x in re.findall(r"\d+", value)
        ))

    raise ValueError(f"Unsupported Pin format: {value}")


def line_capacitance(high, low):
    h = np.array(high, dtype=int) - 1
    l = np.array(low, dtype=int) - 1

    ah = np.sum(a_pin[h])
    al = np.sum(a_pin[l])

    if abs(ah + al) < 1e-30:
        raise ZeroDivisionError("Invalid line admittance")

    y_line = ah * al / (ah + al)

    return y_line.imag / OMEGA


# =========================================================
# 1. DATA PREPARATION
# =========================================================

data_ab = df.copy().reset_index(drop=True)

data_ab["high_key"] = data_ab["high_pins"].apply(
    parse_pins
)
data_ab["low_key"] = data_ab["low_pins"].apply(
    parse_pins
)

for col in ["C", "C_local_pred"]:
    data_ab[col] = pd.to_numeric(
        data_ab[col], errors="raise"
    )

# F 단위 검증
if data_ab["C"].abs().max() > 1e-9:
    raise ValueError("C must be in Farads.")

if data_ab["C_local_pred"].abs().max() > 1e-9:
    raise ValueError("C_local_pred must be in Farads.")

data_ab["C_line"] = [
    line_capacitance(h, l)
    for h, l in zip(
        data_ab["high_key"],
        data_ab["low_key"]
    )
]

data_ab["C_model"] = (
    data_ab["C_line"] + data_ab["C_local_pred"]
)

# =========================================================
# 2. LOW A / LOW B 추출
# =========================================================

a = data_ab[
    data_ab["low_key"] == LOW_A
].copy()

b = data_ab[
    data_ab["low_key"] == LOW_B
].copy()

print("Low A measurements:", len(a))
print("Low B measurements:", len(b))

# 동일 High / Low 구성에 반복 측정이 있을 경우,
# 먼저 평균을 사용해 중복 Cartesian merge 방지.
def aggregate_repeats(frame):
    return (
        frame.groupby("high_key", as_index=False)
        .agg(
            C=("C", "mean"),
            C_line=("C_line", "mean"),
            C_local_pred=("C_local_pred", "mean"),
            C_model=("C_model", "mean"),
            n=("C", "size")
        )
    )

a = aggregate_repeats(a)
b = aggregate_repeats(b)

matched = a.merge(
    b,
    on="high_key",
    suffixes=("_A", "_B"),
    validate="one_to_one"
)

if matched.empty:
    raise ValueError(
        "No identical High configurations with both Low A/B."
    )

# =========================================================
# 3. MEASURED DELTA / PREDICTED DELTA
# =========================================================

matched["dC_meas_fF"] = (
    matched["C_B"] - matched["C_A"]
) * 1e15

matched["dC_line_fF"] = (
    matched["C_line_B"] - matched["C_line_A"]
) * 1e15

matched["dC_local_fF"] = (
    matched["C_local_pred_B"]
    - matched["C_local_pred_A"]
) * 1e15

matched["dC_pred_fF"] = (
    matched["dC_line_fF"]
    + matched["dC_local_fF"]
)

matched["D_fF"] = (
    matched["dC_meas_fF"]
    - matched["dC_pred_fF"]
)

matched["abs_D_fF"] = matched["D_fF"].abs()

matched["High"] = matched["high_key"].apply(
    lambda x: ",".join(map(str, x))
)

matched["H_count"] = matched["high_key"].apply(len)

# =========================================================
# 4. STATISTICS
# =========================================================

d = matched["D_fF"].to_numpy()

print("\n" + "=" * 70)
print("LOW 7~13 vs 8~14 : MATCHED CONTRAST")
print("=" * 70)

print("Matched High configurations:", len(matched))
print(f"Mean D   : {np.mean(d):+.3f} fF")
print(f"Std D    : {np.std(d):.3f} fF")
print(f"MAE      : {np.mean(np.abs(d)):.3f} fF")
print(f"RMSE     : {np.sqrt(np.mean(d**2)):.3f} fF")
print(f"Median D : {np.median(d):+.3f} fF")
print(f"Min D    : {np.min(d):+.3f} fF")
print(f"Max D    : {np.max(d):+.3f} fF")

print("\n" + "=" * 70)
print("PER-HIGH RESIDUALS")
print("=" * 70)

columns = [
    "High", "H_count",
    "dC_meas_fF",
    "dC_line_fF",
    "dC_local_fF",
    "dC_pred_fF",
    "D_fF"
]

print(
    matched[columns]
    .sort_values("abs_D_fF", key=lambda s: matched.loc[s.index, "abs_D_fF"],
                 ascending=False)
    .round(3)
    .to_string(index=False)
)

# =========================================================
# 5. VISUALIZATION
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 왼쪽: 예측 vs 실측 변화량
x = matched["dC_pred_fF"].to_numpy()
y = matched["dC_meas_fF"].to_numpy()

axes[0].scatter(x, y, alpha=0.75)

lo = min(x.min(), y.min(), 0)
hi = max(x.max(), y.max(), 0)

axes[0].plot(
    [lo, hi], [lo, hi],
    "--", label="Ideal y=x"
)

axes[0].set_xlabel("Predicted B-A (fF)")
axes[0].set_ylabel("Measured B-A (fF)")
axes[0].set_title("Low A/B Swap")
axes[0].legend()
axes[0].grid(True)

# 오른쪽: High Pin 개수 vs 모델 잔차
axes[1].scatter(
    matched["H_count"],
    matched["D_fF"],
    alpha=0.75
)

axes[1].axhline(0, linestyle="--")
axes[1].set_xlabel("Number of High Pins")
axes[1].set_ylabel("Residual D (fF)")
axes[1].set_title("Residual vs High Count")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 6. SAVE
# =========================================================

matched[columns].to_csv(
    "low_AB_swap_residual.csv",
    index=False
)

ab_results = matched.copy()


/////////////////////

import numpy as np

h = ab_results["H_count"].to_numpy(dtype=float)
d = ab_results["D_fF"].to_numpy(dtype=float)

# 동일 High 개수별 평균
print("H count | Mean D (fF) | Std D (fF) | N")
for n in np.unique(h):
    v = d[h == n]
    print(
        f"{int(n):7d} | "
        f"{v.mean():11.2f} | "
        f"{v.std():10.2f} | "
        f"{len(v)}"
    )

# 1차 함수와 2차 함수 비교 (진단용)
p1 = np.polyfit(h, d, 1)
p2 = np.polyfit(h, d, 2)

rmse1 = np.sqrt(np.mean((d - np.polyval(p1, h))**2))
rmse2 = np.sqrt(np.mean((d - np.polyval(p2, h))**2))

print("\nLinear RMSE :", rmse1, "fF")
print("Quadratic RMSE:", rmse2, "fF")
print("Quadratic coefficients:", p2)

if p2[0] < 0:
    peak = -p2[1] / (2 * p2[0])
    print("Estimated peak H count:", peak)