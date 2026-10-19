
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# SETTINGS
# ======================================================

CSV_PATH = "multipin_experiments.csv"

# C 컬럼의 단위: pF
# G 컬럼의 단위: uS라고 가정
# 다르면 파일 읽은 직후 환산할 것

SMU = np.array(
    [0]*6 + [1]*8 + [2]*8 +
    [3]*8 + [4]*8 + [5]*6
)

# Pair Matrix의 10 kHz 잔차에서 추정한 값 (pF)
C_LOCAL = np.array([
    1.544,  # Adjacent Even
    0.605,  # Adjacent Odd
    0.175   # Distance 2
])


def parse_pins(value):
    if pd.isna(value):
        return []

    pins = [
        int(x) for x in
        re.findall(r"\d+", str(value))
    ]

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError(f"Invalid pin: {value}")

    if len(pins) != len(set(pins)):
        raise ValueError(f"Duplicate pin: {value}")

    return sorted(pins)


def features(high, low):
    if set(high) & set(low):
        raise ValueError("High and Low overlap")

    hcount = np.zeros(6, dtype=int)
    lcount = np.zeros(6, dtype=int)

    for p in high:
        hcount[SMU[p - 1]] += 1

    for p in low:
        lcount[SMU[p - 1]] += 1

    ne = no = n2 = 0

    for a in high:
        for b in low:
            if SMU[a-1] != SMU[b-1]:
                continue

            dist = abs(a-b)

            if dist == 1:
                if max(a, b) % 2 == 0:
                    ne += 1
                else:
                    no += 1
            elif dist == 2:
                n2 += 1

    # 동일한 SMU별 H/L 점유 구조
    signature = tuple(hcount) + tuple(lcount)

    return ne, no, n2, signature


# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(CSV_PATH)

df["high_pins"] = df["High"].apply(parse_pins)
df["low_pins"] = df["Low"].apply(parse_pins)

computed = [
    features(h, l)
    for h, l in zip(
        df["high_pins"], df["low_pins"]
    )
]

df["Ne"] = [x[0] for x in computed]
df["No"] = [x[1] for x in computed]
df["N2"] = [x[2] for x in computed]
df["signature"] = [x[3] for x in computed]

df["C_local_pred"] = (
    C_LOCAL[0] * df["Ne"]
    + C_LOCAL[1] * df["No"]
    + C_LOCAL[2] * df["N2"]
)


# ======================================================
# MATCHED CONTRASTS
#
# 같은 SMU별 H/L 점유 구성에서
# N_local이 가장 적은 실험을 기준으로 선택.
#
# 차분 비교를 통해 공통 성분을 부분 제거.
# ======================================================

rows = []

for signature, group in df.groupby("signature"):
    if len(group) < 2:
        continue

    group = group.sort_values("C_local_pred")
    ref = group.iloc[0]

    for idx in range(1, len(group)):
        r = group.iloc[idx]

        dc_meas = float(r["C"] - ref["C"])
        dc_pred = float(
            r["C_local_pred"] - ref["C_local_pred"]
        )

        if abs(dc_pred) < 1e-12:
            continue

        rows.append({
            "signature": str(signature),
            "delta_C_meas": dc_meas,
            "delta_C_pred": dc_pred,
            "delta_G_meas": float(
                r["G"] - ref["G"]
            ),
            "delta_Ne": int(r["Ne"] - ref["Ne"]),
            "delta_No": int(r["No"] - ref["No"]),
            "delta_N2": int(r["N2"] - ref["N2"])
        })

contrast = pd.DataFrame(rows)

if len(contrast) == 0:
    print("No matched contrasts.")
    print("SMU별 H/L 점유가 같은 조합이 필요합니다.")

else:
    x = contrast["delta_C_pred"].to_numpy()
    y = contrast["delta_C_meas"].to_numpy()

    err = y - x

    # Pair Matrix 계수 그대로 사용한 고정 예측
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))

    # y = a*x + b 진단용 회귀
    slope, intercept = np.polyfit(x, y, 1)

    corr = (
        np.corrcoef(x, y)[0, 1]
        if np.std(x) > 0 and np.std(y) > 0
        else np.nan
    )

    print("\n===== LOCAL COUPLING VALIDATION =====")
    print("Number of contrasts:", len(x))
    print(f"MAE             : {mae*1000:.2f} fF")
    print(f"RMSE            : {rmse*1000:.2f} fF")
    print(f"Slope           : {slope:.4f}")
    print(f"Intercept       : {intercept:.4f} pF")
    print(f"Correlation     : {corr:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.65)

    lower = min(x.min(), y.min(), 0)
    upper = max(x.max(), y.max(), 0)

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Ideal y=x"
    )

    plt.xlabel("Predicted delta C (pF)")
    plt.ylabel("Measured delta C (pF)")
    plt.title("Local Admittance Transfer Validation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    contrast.to_csv(
        "local_coupling_contrasts.csv",
        index=False
    )

# 정리된 특징량 저장
df.drop(
    columns=["high_pins", "low_pins", "signature"]
).to_csv(
    "multipin_features.csv",
    index=False
)
