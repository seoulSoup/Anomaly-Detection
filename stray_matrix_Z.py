
import numpy as np
from scipy.optimize import least_squares

# ============================================================
# 0. INPUT / SETTINGS
# ============================================================

# Z: 44 x 44 complex impedance matrix [Ohm]
# Python index 0 = Pin 1, index 43 = Pin 44
#
# Example:
# Z = np.load("open_impedance.npy")
#
# 실제 회사 PC의 Matrix를 여기에 연결할 것.

N = 44

SMU = np.array(
    [0] * 6 +
    [1] * 8 +
    [2] * 8 +
    [3] * 8 +
    [4] * 8 +
    [5] * 6
)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70

# True: 기생계수 피팅에 거리 3 이상 Pair만 사용
# False: 동일 SMU의 모든 Pair 사용
FIT_FAR_ONLY = True

# 피팅 조건에 관계없이 전체 Pair의 오차도 출력
EPS = 1e-30
SCALE_Z = 1e6  # MegaOhm 기준 수치 안정화


# ============================================================
# 1. MATRIX VALIDATION
# ============================================================

def validate_matrix(Z):
    Z = np.asarray(Z, dtype=np.complex128)

    if Z.shape != (N, N):
        raise ValueError("Z must be a 44x44 matrix.")

    i, j = np.triu_indices(N, 1)

    if not np.all(np.isfinite(Z[i, j])):
        raise ValueError("Invalid values in upper triangle.")

    if np.any(np.abs(Z[i, j]) < EPS):
        raise ValueError("Zero impedance found in upper triangle.")

    # 상삼각을 기준으로 대칭 Matrix 생성.
    # 대각선은 측정 대상이 아니므로 0.
    result = np.zeros_like(Z)

    result[i, j] = Z[i, j]
    result[j, i] = Z[i, j]

    return result


def make_pairs(Z):
    i, j = np.triu_indices(N, 1)

    actual = Z[i, j]
    same = SMU[i] == SMU[j]
    distance = j - i

    return i, j, actual, same, distance


# ============================================================
# 2. CROSS-SMU PIN IMPEDANCE FIT
#
# Cross-SMU:
#   Z_ij ~ z_i + z_j
#
# 동일 SMU Pair는 이 단계에서 사용하지 않는다.
# ============================================================

def fit_pin_impedance(i, j, actual, same):
    idx = np.where(~same)[0]

    A = np.zeros((len(idx), N), dtype=float)

    rows = np.arange(len(idx))
    A[rows, i[idx]] = 1.0
    A[rows, j[idx]] = 1.0

    z_pin, *_ = np.linalg.lstsq(
        A,
        actual[idx],
        rcond=None
    )

    return z_pin


# ============================================================
# 3. SAME-SMU TRAIN / TEST SPLIT
#
# 각 SMU별로 무작위 분리
# ============================================================

def make_train_test(i, same, distance):
    rng = np.random.default_rng(RANDOM_SEED)

    train = []
    test = []

    for g in range(6):
        group = np.where(
            same & (SMU[i] == g)
        )[0]

        if FIT_FAR_ONLY:
            eligible = group[distance[group] >= 3]
        else:
            eligible = group.copy()

        eligible = rng.permutation(eligible)

        n_train = int(
            round(len(eligible) * TRAIN_RATIO)
        )
        n_train = max(1, min(n_train, len(eligible) - 1))

        train.extend(eligible[:n_train])
        test.extend(eligible[n_train:])

        # 근접 Pair는 피팅에 사용하지 않고 별도 평가
        if FIT_FAR_ONLY:
            excluded = group[distance[group] < 3]
            test.extend(excluded)

    return (
        np.array(train, dtype=int),
        np.array(test, dtype=int)
    )


# ============================================================
# 4. MODEL Z
#
# Z_pred = Z_base + d_g
#
# d_g: SMU별 직렬 기생 임피던스
# ============================================================

def fit_model_z(i, actual, base, train):
    d = np.zeros(6, dtype=complex)

    for g in range(6):
        idx = train[SMU[i[train]] == g]

        d[g] = np.mean(
            actual[idx] - base[idx]
        )

    return d


def predict_model_z(i, base, same, d):
    pred = base.copy()

    idx = np.where(same)[0]
    pred[idx] += d[SMU[i[idx]]]

    return pred


# ============================================================
# 5. MODEL Y
#
# Z_pred = 1 / (1/Z_base + y_g)
#
# y_g: SMU별 병렬 기생 어드미턴스
#
# Z-domain 최소제곱으로 최적화.
# Model Z와 동일한 오차 단위를 사용.
# ============================================================

def fit_model_y(i, actual, base, train):
    y = np.zeros(6, dtype=complex)

    for g in range(6):
        idx = train[SMU[i[train]] == g]

        za = actual[idx] / SCALE_Z
        zb = base[idx] / SCALE_Z

        if np.any(np.abs(zb) < 1e-15):
            raise ValueError(
                f"Near-zero base impedance in SMU {g+1}"
            )

        # 단위: 1/MOhm
        initial_y = np.mean(
            1.0 / za - 1.0 / zb
        )

        def residual(x):
            yg = x[0] + 1j * x[1]

            denom = 1.0 / zb + yg

            # 분모가 너무 작은 해는 큰 패널티 부여
            if np.any(np.abs(denom) < 1e-12):
                return np.full(
                    2 * len(idx),
                    1e8,
                    dtype=float
                )

            zp = 1.0 / denom
            diff = zp - za

            return np.concatenate([
                diff.real,
                diff.imag
            ])

        x0 = np.array([
            initial_y.real,
            initial_y.imag
        ])

        result = least_squares(
            residual,
            x0,
            max_nfev=5000,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11
        )

        # 1/MOhm -> Siemens
        y[g] = (
            result.x[0] + 1j * result.x[1]
        ) / SCALE_Z

    return y


def predict_model_y(i, base, same, y):
    pred = base.copy()

    idx = np.where(same)[0]

    denom = (
        1.0 / base[idx]
        + y[SMU[i[idx]]]
    )

    if np.any(np.abs(denom) < EPS):
        raise ZeroDivisionError(
            "Near-zero denominator in Model Y prediction."
        )

    pred[idx] = 1.0 / denom

    return pred


# ============================================================
# 6. METRICS
# ============================================================

def metrics(actual, pred, idx):
    if len(idx) == 0:
        return np.nan, np.nan, np.nan

    a = actual[idx]
    p = pred[idx]

    absolute_error = np.abs(p - a)

    relative_error = (
        absolute_error /
        np.maximum(np.abs(a), EPS)
    )

    mae_mohm = (
        np.mean(absolute_error) / 1e6
    )

    mean_relative = (
        100 * np.mean(relative_error)
    )

    max_relative = (
        100 * np.max(relative_error)
    )

    return (
        mae_mohm,
        mean_relative,
        max_relative
    )


def print_metrics(
    title, actual, predictions, idx
):
    print("\n" + title)
    print("-" * 68)

    print(
        f"{'Model':<18}"
        f"{'MAE(MOhm)':>13}"
        f"{'MeanErr(%)':>15}"
        f"{'MaxErr(%)':>14}"
    )

    for name, pred in predictions.items():
        mae, avg, mx = metrics(
            actual, pred, idx
        )

        print(
            f"{name:<18}"
            f"{mae:13.4f}"
            f"{avg:15.3f}"
            f"{mx:14.3f}"
        )


# ============================================================
# 7. PAIR CATEGORY
# ============================================================

def pair_category(i, j):
    # i, j are Python zero-based indices
    pin_i = i + 1
    pin_j = j + 1

    distance = abs(pin_i - pin_j)

    if distance == 1:
        if max(pin_i, pin_j) % 2 == 0:
            return "Adjacent Even"
        return "Adjacent Odd"

    if distance == 2:
        return "Distance 2"

    return "Distance >=3"


# ============================================================
# 8. MAIN
# ============================================================

def run_comparison(Z):
    Z = validate_matrix(Z)

    i, j, actual, same, distance = make_pairs(Z)

    print("=" * 68)
    print("HYBRID PARASITIC MODEL COMPARISON")
    print("=" * 68)

    print("Total pairs:", len(actual))
    print("Same SMU pairs:", np.sum(same))
    print("Cross SMU pairs:", np.sum(~same))
    print("FIT_FAR_ONLY:", FIT_FAR_ONLY)

    # --------------------------------------------------------
    # 8-1. Cross-SMU Pin Impedance Fit
    # --------------------------------------------------------

    z_pin = fit_pin_impedance(
        i, j, actual, same
    )

    base = z_pin[i] + z_pin[j]

    # --------------------------------------------------------
    # 8-2. Same-SMU Train/Test Split
    # --------------------------------------------------------

    train, test = make_train_test(
        i, same, distance
    )

    # --------------------------------------------------------
    # 8-3. Fit Model Z & Model Y
    # --------------------------------------------------------

    d = fit_model_z(
        i, actual, base, train
    )

    y = fit_model_y(
        i, actual, base, train
    )

    pred_base = base.copy()

    pred_z = predict_model_z(
        i, base, same, d
    )

    pred_y = predict_model_y(
        i, base, same, y
    )

    predictions = {
        "No Parasitic": pred_base,
        "Series Z": pred_z,
        "Parallel Y": pred_y
    }

    # --------------------------------------------------------
    # 8-4. Overall Same-SMU Results
    # --------------------------------------------------------

    print_metrics(
        "SAME SMU - TRAIN",
        actual, predictions, train
    )

    print_metrics(
        "SAME SMU - TEST",
        actual, predictions, test
    )

    cross_idx = np.where(~same)[0]

    print_metrics(
        "CROSS SMU - BASELINE",
        actual, predictions, cross_idx
    )

    # --------------------------------------------------------
    # 8-5. SMU-by-SMU Test Results
    # --------------------------------------------------------

    for g in range(6):
        idx = test[
            SMU[i[test]] == g
        ]

        print_metrics(
            f"SMU {g+1} - TEST",
            actual, predictions, idx
        )

    # --------------------------------------------------------
    # 8-6. Pair Distance Results
    # --------------------------------------------------------

    categories = [
        "Adjacent Even",
        "Adjacent Odd",
        "Distance 2",
        "Distance >=3"
    ]

    for category in categories:
        idx = np.array(
            [
                k for k in test
                if pair_category(i[k], j[k])
                == category
            ],
            dtype=int
        )

        print_metrics(
            f"{category} - TEST",
            actual, predictions, idx
        )

    # --------------------------------------------------------
    # 8-7. Distance >= 3 Only
    # --------------------------------------------------------

    far_test = test[
        distance[test] >= 3
    ]

    print_metrics(
        "SAME SMU - DISTANCE >= 3 - TEST",
        actual, predictions, far_test
    )

    # --------------------------------------------------------
    # 8-8. Fitted Parameters
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("FITTED SMU PARASITIC PARAMETERS")
    print("=" * 68)

    print(
        f"{'SMU':<7}"
        f"{'d_real(MOhm)':>17}"
        f"{'d_imag(MOhm)':>17}"
        f"{'y_real(uS)':>17}"
        f"{'y_imag(uS)':>17}"
    )

    for g in range(6):
        print(
            f"{g+1:<7}"
            f"{d[g].real/1e6:17.6f}"
            f"{d[g].imag/1e6:17.6f}"
            f"{y[g].real*1e6:17.6f}"
            f"{y[g].imag*1e6:17.6f}"
        )

    # --------------------------------------------------------
    # 8-9. Pin Series Impedance Summary
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("PIN SERIES IMPEDANCE SUMMARY")
    print("=" * 68)

    print(
        "Mean |z_pin| (MOhm):",
        np.mean(np.abs(z_pin)) / 1e6
    )
    print(
        "Min  |z_pin| (MOhm):",
        np.min(np.abs(z_pin)) / 1e6
    )
    print(
        "Max  |z_pin| (MOhm):",
        np.max(np.abs(z_pin)) / 1e6
    )

    # --------------------------------------------------------
    # 8-10. Return Results
    # --------------------------------------------------------

    return {
        "z_pin": z_pin,
        "series_parasitic": d,
        "parallel_parasitic": y,
        "actual": actual,
        "pred_base": pred_base,
        "pred_z": pred_z,
        "pred_y": pred_y,
        "train_indices": train,
        "test_indices": test,
        "pair_i": i,
        "pair_j": j,
        "same_smu": same,
        "distance": distance
    }


# ============================================================
# 9. EXECUTE
# ============================================================

# Z Matrix가 준비된 후 아래 한 줄 실행
results = run_comparison(Z)
