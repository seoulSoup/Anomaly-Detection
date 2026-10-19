
import numpy as np

# ============================================================
# RESIDUAL ADMITTANCE ANALYSIS
# 기존 변수: Z, results, SMU
# ============================================================

# 측정 주파수를 알고 있다면 Hz 단위로 입력.
# 모르면 None으로 두면 됨.
FREQ_HZ = None

# 예: FREQ_HZ = 1000.0

N = 44
EPS = 1e-30

i = results["pair_i"]
j = results["pair_j"]
z_pin = results["z_pin"]
y_smu = results["parallel_parasitic"]
actual = results["actual"]
same = results["same_smu"]

distance = j - i

# 기존 피팅에서 사용한 Pair
train_idx = results["train_indices"]
test_idx = results["test_indices"]

# ------------------------------------------------------------
# 1. Pair별 잔여 Admittance
# ------------------------------------------------------------

z_base = z_pin[i] + z_pin[j]

if np.any(np.abs(z_base) < EPS):
    raise ValueError("Near-zero base impedance.")

if np.any(np.abs(actual) < EPS):
    raise ValueError("Near-zero measured impedance.")

# 실측 Admittance
y_actual = 1.0 / actual

# 직렬 Pin 경로의 Admittance
y_line = 1.0 / z_base

# SMU 내부 기본 기생 Admittance
y_common = np.zeros(len(actual), dtype=complex)
y_common[same] = y_smu[SMU[i[same]]]

# 잔여 Admittance
delta_y = y_actual - y_line - y_common


# ------------------------------------------------------------
# 2. Pair 유형 분류
# ------------------------------------------------------------

def get_category(a, b):
    dist = b - a
    higher_pin = b + 1  # 1-based Pin 번호

    if dist == 1:
        return (
            "Adjacent Even"
            if higher_pin % 2 == 0
            else "Adjacent Odd"
        )

    if dist == 2:
        return "Distance 2"

    return "Distance >=3"


categories = np.array(
    [get_category(a, b) for a, b in zip(i, j)]
)

category_names = [
    "Adjacent Even",
    "Adjacent Odd",
    "Distance 2",
    "Distance >=3"
]


# ------------------------------------------------------------
# 3. 복소 Admittance 통계
#
# 출력 단위:
#   Y: uS (microSiemens)
#   C: pF, 주파수가 지정된 경우
# ------------------------------------------------------------

def print_y_stats(title, mask):
    values = delta_y[mask]
    n = len(values)

    print("\n" + title)
    print("-" * 62)

    if n == 0:
        print("No pairs")
        return

    mean_y = np.mean(values)
    std_real = np.std(values.real)
    std_imag = np.std(values.imag)

    rms_y = np.sqrt(np.mean(np.abs(values)**2))

    print(f"Count                 : {n}")
    print(f"Mean Real Y (uS)      : {mean_y.real*1e6:.6f}")
    print(f"Mean Imag Y (uS)      : {mean_y.imag*1e6:.6f}")
    print(f"Std Real Y (uS)       : {std_real*1e6:.6f}")
    print(f"Std Imag Y (uS)       : {std_imag*1e6:.6f}")
    print(f"RMS |Delta Y| (uS)    : {rms_y*1e6:.6f}")

    if FREQ_HZ is not None:
        omega = 2 * np.pi * FREQ_HZ
        cp = values.imag / omega * 1e12

        print(f"Mean Delta C (pF)     : {np.mean(cp):.6f}")
        print(f"Std Delta C (fF)      : {np.std(cp)*1000:.3f}")


# ------------------------------------------------------------
# 4. 전체 거리별 통계
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("RESIDUAL ADMITTANCE BY DISTANCE")
print("=" * 62)

for name in category_names:
    mask = same & (categories == name)
    print_y_stats(name, mask)


# ------------------------------------------------------------
# 5. SMU별 / 거리별 통계
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("RESIDUAL ADMITTANCE BY SMU")
print("=" * 62)

for g in range(6):
    print(f"\n########## SMU {g+1} ##########")

    for name in category_names:
        mask = (
            same &
            (SMU[i] == g) &
            (categories == name)
        )
        print_y_stats(name, mask)


# ------------------------------------------------------------
# 6. 근접 Pair에 대한 Admittance 추가 보정
#
# 거리 >=3은 기존 모델 유지
# 거리 1,2는 Delta Y 평균을 추가
#
# 주의:
# 이 부분은 모델의 설명력 확인용.
# 검증용 성능은 별도 Train/Test 분리가 필요함.
# ------------------------------------------------------------

local_y = {}

for name in category_names:
    mask = same & (categories == name)

    if name == "Distance >=3":
        local_y[name] = 0.0 + 0.0j
    elif np.any(mask):
        local_y[name] = np.mean(delta_y[mask])
    else:
        local_y[name] = 0.0 + 0.0j


y_pred_old = y_line + y_common
y_pred_new = y_pred_old.copy()

for name in category_names:
    mask = same & (categories == name)
    y_pred_new[mask] += local_y[name]

# 동일 SMU에 대해서만 비교
same_idx = np.where(same)[0]

def calculate_error(y_pred, indices):
    z_pred = 1.0 / y_pred[indices]
    z_true = actual[indices]

    abs_error = np.abs(z_pred - z_true)
    rel_error = abs_error / np.abs(z_true)

    return (
        np.mean(abs_error) / 1e6,
        np.mean(rel_error) * 100,
        np.max(rel_error) * 100
    )


print("\n" + "=" * 62)
print("LOCAL ADMITTANCE CORRECTION - IN SAMPLE")
print("=" * 62)

print("Model             MAE(MOhm)   MeanErr(%)   MaxErr(%)")

for name, yp in [
    ("Parallel Y", y_pred_old),
    ("Parallel Y+Local", y_pred_new)
]:
    mae, avg, mx = calculate_error(yp, same_idx)
    print(f"{name:<18} {mae:10.4f} {avg:12.3f} {mx:11.3f}")


# ------------------------------------------------------------
# 7. 근접 효과 추정치 출력
# ------------------------------------------------------------

print("\n" + "=" * 62)
print("FITTED LOCAL ADMITTANCE")
print("=" * 62)

for name in category_names:
    v = local_y[name]

    print(
        f"{name:<16}: "
        f"Real={v.real*1e6:+.6f} uS, "
        f"Imag={v.imag*1e6:+.6f} uS"
    )

    if FREQ_HZ is not None:
        cp = v.imag / (2*np.pi*FREQ_HZ) * 1e12
        print(f"                  Delta C = {cp:+.6f} pF")


# 후속 분석을 위해 보관
residual_results = {
    "delta_y": delta_y,
    "categories": categories,
    "local_y": local_y,
    "y_pred_old": y_pred_old,
    "y_pred_new": y_pred_new,
    "same": same,
    "train_idx": train_idx,
    "test_idx": test_idx
}
