
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ============================================================
# MULTI-PIN DATA -> NEW PIN IMPEDANCE FIT
#
# 기존 results["z_pin"] 사용하지 않음
# C와 C_local_pred 단위: F
# 이번 버전은 C 성분만 피팅
# ============================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ
N = 44

SMU = np.array(
    [0]*6 + [1]*8 + [2]*8 +
    [3]*8 + [4]*8 + [5]*6
)

# 기준 어드미턴스 크기
# |z| = 30 MOhm에 해당하는 값
A0 = 1.0 / (30e6)

TRAIN_RATIO = 0.7
RANDOM_SEED = 42

# v_i = log(a_i / a0)에 대한 정규화
# 기존 z_i에 맞추는 정규화가 아님
REG_FF = 20.0
REG_SIGMA = 0.30

# ============================================================
# 1. DATA PREPARATION
# ============================================================

data = df.copy().reset_index(drop=True)

required = [
    "high_pins",
    "low_pins",
    "signature",
    "C",
    "C_local_pred"
]

for name in required:
    if name not in data.columns:
        raise ValueError(f"Missing column: {name}")

data["C"] = pd.to_numeric(data["C"])
data["C_local_pred"] = pd.to_numeric(
    data["C_local_pred"]
)

if data["C"].abs().max() > 1e-9:
    raise ValueError("C must be in Farads.")

if data["C_local_pred"].abs().max() > 1e-9:
    raise ValueError("C_local_pred must be in Farads.")

# Local 기생 제거
data["C_target"] = (
    data["C"] - data["C_local_pred"]
)

# High/Low 연결 행렬
H = np.zeros((len(data), N))
L = np.zeros((len(data), N))

for k, row in data.iterrows():
    high = [int(x) for x in row["high_pins"]]
    low = [int(x) for x in row["low_pins"]]

    if not high or not low:
        raise ValueError(f"Empty H/L in row {k}")

    if set(high) & set(low):
        raise ValueError(f"H/L overlap in row {k}")

    if any(p < 1 or p > 44 for p in high + low):
        raise ValueError(f"Invalid pin in row {k}")

    H[k, np.array(high)-1] = 1
    L[k, np.array(low)-1] = 1

# signature를 문자열로 바꿔 그룹화
data["group_id"] = pd.factorize(
    data["signature"].astype(str)
)[0]

# ============================================================
# 2. TRAIN / TEST SPLIT
#
# 같은 signature 내부에서 분리.
# Singleton 그룹은 피팅에서 제외.
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

train = []
test = []

for _, group in data.groupby("group_id"):
    idx = group.index.to_numpy()

    if len(idx) < 2:
        continue

    idx = rng.permutation(idx)

    n_train = int(
        np.ceil(len(idx) * TRAIN_RATIO)
    )
    n_train = min(n_train, len(idx)-1)

    train.extend(idx[:n_train])
    test.extend(idx[n_train:])

train = np.array(train, dtype=int)
test = np.array(test, dtype=int)

if len(train) == 0 or len(test) == 0:
    raise ValueError("Not enough repeated signatures.")

print("Train rows:", len(train))
print("Test rows :", len(test))

# ============================================================
# 3. PIN PARAMETERIZATION
#
# a_i = j * A0 * exp(v_i)
#
# 각 SMU 안에서 sum(v_i) = 0.
# 자유 파라미터: 44 - 6 = 38개
# ============================================================

groups = [
    np.where(SMU == g)[0]
    for g in range(6)
]

def decode_params(x):
    v = np.zeros(N)

    pos = 0

    for pins in groups:
        m = len(pins)-1

        v[pins[:-1]] = x[pos:pos+m]
        v[pins[-1]] = -np.sum(
            x[pos:pos+m]
        )

        pos += m

    return v

NUM_PARAMS = N - 6

# ============================================================
# 4. LINE AGGREGATION
# ============================================================

def predict_c_line(x):
    v = decode_params(x)

    # Capacitive pin admittance
    a = 1j * A0 * np.exp(v)

    ah = H @ a
    al = L @ a

    y_line = ah * al / (ah + al)

    return y_line.imag / OMEGA


# ============================================================
# 5. GROUP CENTERING
#
# 동일 signature 내 평균을 제거하면
# 미지의 SMU 기준 C가 사라진다는 가정.
# ============================================================

target_fF = data["C_target"].to_numpy() * 1e15
group_ids = data["group_id"].to_numpy()

def center_train(values):
    out = np.zeros(len(train))

    for g in np.unique(group_ids[train]):
        mask = group_ids[train] == g

        sub = values[train][mask]
        out[mask] = sub - np.mean(sub)

    return out


target_centered = center_train(target_fF)

# ============================================================
# 6. FIT PIN IMPEDANCE
# ============================================================

def residual(x):
    pred_fF = predict_c_line(x) * 1e15

    model_centered = center_train(pred_fF)

    fit_error = (
        model_centered - target_centered
    )

    # 정규화: 과도한 Pin 편차 억제
    v = decode_params(x)
    reg = REG_FF * v / REG_SIGMA

    return np.r_[fit_error, reg]


x0 = np.zeros(NUM_PARAMS)

result = least_squares(
    residual,
    x0,
    max_nfev=3000,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10
)

print("\nOptimization success:", result.success)
print("Message:", result.message)

# 새롭게 얻은 z_i
v_fit = decode_params(result.x)

a_pin_new = (
    1j * A0 * np.exp(v_fit)
)

z_pin_new = 1.0 / a_pin_new

# ============================================================
# 7. TRAIN 그룹 평균으로 TEST 예측
#
# Baseline:
#   C_target = signature 평균
#
# New:
#   C_target = signature 평균 잔차 + C_line
# ============================================================

line_fF = (
    predict_c_line(result.x) * 1e15
)

def evaluate(indices):
    baseline_errors = []
    hybrid_errors = []

    for k in indices:
        g = group_ids[k]

        reference = train[
            group_ids[train] == g
        ]

        if len(reference) == 0:
            continue

        # 기존: 그룹 평균만 사용
        baseline = np.mean(
            target_fF[reference]
        )

        # 신규: 그룹 평균 잔차 + 새 선로 기여
        offset = np.mean(
            target_fF[reference]
            - line_fF[reference]
        )

        hybrid = offset + line_fF[k]

        baseline_errors.append(
            target_fF[k] - baseline
        )

        hybrid_errors.append(
            target_fF[k] - hybrid
        )

    return (
        np.asarray(baseline_errors),
        np.asarray(hybrid_errors)
    )


def report(title, indices):
    e_base, e_hybrid = evaluate(indices)

    def calc(e):
        return (
            np.mean(np.abs(e)),
            np.sqrt(np.mean(e**2)),
            np.max(np.abs(e))
        )

    b = calc(e_base)
    h = calc(e_hybrid)

    print("\n" + title)
    print("-" * 55)
    print("                 MAE(fF)    RMSE(fF)   MaxAE(fF)")
    print(
        f"Baseline       {b[0]:10.3f}"
        f" {b[1]:11.3f} {b[2]:11.3f}"
    )
    print(
        f"Local + Line   {h[0]:10.3f}"
        f" {h[1]:11.3f} {h[2]:11.3f}"
    )


report("TRAIN", train)
report("TEST", test)

# ============================================================
# 8. TEST SCATTER
# ============================================================

x_plot = []
y_plot = []

for k in test:
    g = group_ids[k]

    reference = train[
        group_ids[train] == g
    ]

    base_mean = np.mean(
        target_fF[reference]
    )

    line_mean = np.mean(
        line_fF[reference]
    )

    x_plot.append(
        line_fF[k] - line_mean
    )

    y_plot.append(
        target_fF[k] - base_mean
    )

x_plot = np.array(x_plot)
y_plot = np.array(y_plot)

plt.figure(figsize=(6, 6))
plt.scatter(x_plot, y_plot, alpha=0.7)

lo = min(x_plot.min(), y_plot.min(), 0)
hi = max(x_plot.max(), y_plot.max(), 0)

plt.plot([lo, hi], [lo, hi], "--")
plt.xlabel("Predicted line delta C (fF)")
plt.ylabel("Measured residual delta C (fF)")
plt.title("New Pin Impedance - TEST")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# 9. PARAMETER SUMMARY
# ============================================================

print("\nNEW PIN PARAMETERS")
print("Pin  SMU       |z_i| (MOhm)")

for p in range(N):
    print(
        f"{p+1:2d}   "
        f"{SMU[p]+1}       "
        f"{abs(z_pin_new[p])/1e6:10.4f}"
    )

# 후속 분석을 위해 변수 유지
pin_fit_new = {
    "z_pin": z_pin_new,
    "a_pin": a_pin_new,
    "v_pin": v_fit,
    "train": train,
    "test": test,
    "optimization": result,
    "line_fF": line_fF
}


///////////////////////////0
import numpy as np
import pandas as pd

rows = []

for k in test:
    g = group_ids[k]
    ref = train[group_ids[train] == g]

    base_mean = target_fF[ref].mean()
    offset = (target_fF[ref] - line_fF[ref]).mean()

    pred = offset + line_fF[k]
    residual = target_fF[k] - pred

    rows.append({
        "index": int(k),
        "signature": str(data.loc[k, "signature"]),
        "High": data.loc[k, "high_pins"],
        "Low": data.loc[k, "low_pins"],
        "line_delta_fF": line_fF[k] - line_fF[ref].mean(),
        "measured_delta_fF": target_fF[k] - base_mean,
        "residual_fF": residual,
        "abs_error_fF": abs(residual)
    })

diagnostics = pd.DataFrame(rows)

print("=== TEST ERROR QUANTILES ===")
print(
    diagnostics["abs_error_fF"].quantile(
        [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]
    )
)

print("\n=== TOP 10 TEST ERRORS ===")
print(
    diagnostics.sort_values(
        "abs_error_fF", ascending=False
    ).head(10).to_string(index=False)
)

# 원점 주변과 바깥쪽 성능 분리
for threshold in [25, 50, 100, 200]:
    sub = diagnostics[
        diagnostics["line_delta_fF"].abs() >= threshold
    ]
    if len(sub):
        print(
            f"|Predicted delta| >= {threshold} fF:",
            f"N={len(sub)}",
            f"RMSE={np.sqrt(np.mean(sub.residual_fF**2)):.2f} fF"
        )

/////////<__/_<=<=_=>=>


import numpy as np
import pandas as pd

# diagnostics:
# 기존 코드의 Test 결과 테이블
# columns:
# High, Low, residual_fF, abs_error_fF

rows = []

for pin in range(1, 45):
    for side in ["High", "Low"]:

        used = diagnostics[side].apply(
            lambda pins: pin in pins
        )

        sub = diagnostics.loc[used]

        if len(sub) < 3:
            continue

        r = sub["residual_fF"].to_numpy()

        rows.append({
            "Pin": pin,
            "SMU": int(SMU[pin-1]) + 1,
            "Side": side,
            "Count": len(sub),
            "Bias_fF": np.mean(r),
            "MAE_fF": np.mean(np.abs(r)),
            "RMSE_fF": np.sqrt(np.mean(r**2))
        })

pin_errors = pd.DataFrame(rows)

print("\n=== PIN ERROR RANKING ===")
print(
    pin_errors.sort_values(
        "RMSE_fF", ascending=False
    ).head(20).to_string(index=False)
)

print("\n=== SMU2 ONLY ===")
print(
    pin_errors[
        pin_errors["SMU"] == 2
    ].sort_values(
        ["Side", "RMSE_fF"],
        ascending=[True, False]
    ).to_string(index=False)
)
