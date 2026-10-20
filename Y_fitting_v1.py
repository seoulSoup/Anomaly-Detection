
import ast
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# ============================================================
# 0. SETTINGS
# ============================================================

FREQ_HZ = 10000.0
OMEGA = 2 * np.pi * FREQ_HZ

# 기존 실험에서 C는 F 단위로 사용했음.
C_UNIT = "F"

# 중요: 실제 df["G"]의 단위를 선택!
# "S", "uS", "nS" 중 하나
G_UNIT = "S"

RANDOM_SEED = 42
TEST_RATIO = 0.25

# 고정된 경험적 Local C 계수 [pF]
LOCAL_C_PF = np.array([
    1.544,  # Adjacent Even
    0.605,  # Adjacent Odd
    0.175   # Distance 2
])

# Ridge 정규화 강도
# Pin 모델을 상대적으로 더 강하게 정규화
ALPHA_SMU = 20.0
ALPHA_PIN = 100.0

SMU = np.array(
    [0]*6 + [1]*8 + [2]*8 +
    [3]*8 + [4]*8 + [5]*6
)

SMU_SIZE = np.bincount(SMU, minlength=6)

# ------------------------------------------------------------
# Units
# ------------------------------------------------------------

C_FACTORS = {
    "F": 1.0,
    "pF": 1e-12
}

G_FACTORS = {
    "S": 1.0,
    "uS": 1e-6,
    "nS": 1e-9
}

if C_UNIT not in C_FACTORS:
    raise ValueError("Invalid C_UNIT")

if G_UNIT not in G_FACTORS:
    raise ValueError("Invalid G_UNIT")


# ============================================================
# 1. PIN PARSER
# ============================================================

def parse_pins(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        pins = [int(v) for v in value]

    elif isinstance(value, str):
        try:
            obj = ast.literal_eval(value)

            if isinstance(obj, (list, tuple, set)):
                pins = [int(v) for v in obj]
            else:
                pins = [
                    int(v)
                    for v in re.findall(r"\d+", value)
                ]

        except (SyntaxError, ValueError):
            pins = [
                int(v)
                for v in re.findall(r"\d+", value)
            ]
    else:
        raise ValueError(f"Invalid pin format: {value}")

    if not pins:
        raise ValueError("Empty pin list")

    if len(set(pins)) != len(pins):
        raise ValueError("Duplicate pins")

    if any(p < 1 or p > 44 for p in pins):
        raise ValueError("Pin must be 1~44")

    return tuple(sorted(pins))


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(high, low):
    high = parse_pins(high)
    low = parse_pins(low)

    if set(high) & set(low):
        raise ValueError("High and Low overlap")

    h = np.zeros(44)
    l = np.zeros(44)

    h[np.array(high)-1] = 1
    l[np.array(low)-1] = 1

    hc = np.bincount(
        SMU, weights=h, minlength=6
    )
    lc = np.bincount(
        SMU, weights=l, minlength=6
    )

    # ----------------------------------------
    # Local pair counting
    # ----------------------------------------

    ne = no = n2 = 0

    for a in high:
        for b in low:
            if SMU[a-1] != SMU[b-1]:
                continue

            distance = abs(a-b)

            if distance == 1:
                if max(a, b) % 2 == 0:
                    ne += 1
                else:
                    no += 1

            elif distance == 2:
                n2 += 1

    c_local_pf = (
        LOCAL_C_PF[0]*ne +
        LOCAL_C_PF[1]*no +
        LOCAL_C_PF[2]*n2
    )

    # ----------------------------------------
    # M0: SMU occupancy model
    # ----------------------------------------

    features = [1.0]

    # SMU별 H/L Pin 수
    features.extend(hc)
    features.extend(lc)

    # 동일 SMU 내부 H-L 조합 수
    features.extend(hc * lc)

    # 서로 다른 SMU 사이의 H-L 조합 수
    for g in range(6):
        for k in range(g+1, 6):
            features.append(
                hc[g]*lc[k] + hc[k]*lc[g]
            )

    nh = len(high)
    nl = len(low)

    # 전체 연결 수에 대한 낮은 차수의 비선형 특성
    features.extend([
        nh*nl,
        np.log1p(nh),
        np.log1p(nl)
    ])

    x_smu = np.asarray(features, dtype=float)

    # ----------------------------------------
    # M1: Pin-specific deviations
    #
    # 각 SMU 내 Pin 평균 선택효과를 제거.
    # SMU 점유 효과와 Pin 편차를 구분하기 위함.
    # ----------------------------------------

    pin_h = h.copy()
    pin_l = l.copy()

    for g in range(6):
        idx = np.where(SMU == g)[0]

        pin_h[idx] -= hc[g] / SMU_SIZE[g]
        pin_l[idx] -= lc[g] / SMU_SIZE[g]

    x_pin = np.r_[pin_h, pin_l]

    signature = tuple(
        int(v) for v in np.r_[hc, lc]
    )

    return x_smu, x_pin, c_local_pf, signature


# ============================================================
# 3. DATA PREPARATION
# ============================================================

data = df.copy().reset_index(drop=True)

required = ["high_pins", "low_pins", "C", "G"]

for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

parsed = [
    extract_features(h, l)
    for h, l in zip(
        data["high_pins"],
        data["low_pins"]
    )
]

X_smu = np.vstack([p[0] for p in parsed])
X_pin = np.vstack([p[1] for p in parsed])

X_full = np.hstack([X_smu, X_pin])

c_local_pf = np.array([p[2] for p in parsed])

signatures = [
    p[3] for p in parsed
]

# 문자열 그룹 ID
group_ids = np.array(
    [str(s) for s in signatures]
)

# ----------------------------------------
# Targets
# ----------------------------------------

C_F = (
    pd.to_numeric(data["C"]).to_numpy()
    * C_FACTORS[C_UNIT]
)

G_S = (
    pd.to_numeric(data["G"]).to_numpy()
    * G_FACTORS[G_UNIT]
)

if not np.all(np.isfinite(C_F)):
    raise ValueError("Invalid C values")

if not np.all(np.isfinite(G_S)):
    raise ValueError("Invalid G values")

# Local 계수는 고정!
# C_base를 pF 단위로 학습
target_C = C_F * 1e12 - c_local_pf

# G는 nS 단위로 학습
target_G = G_S * 1e9

print("Experiments:", len(data))
print("Signatures:", len(set(group_ids)))

print("C range [pF]:",
      C_F.min()*1e12,
      C_F.max()*1e12)

print("G range [nS]:",
      G_S.min()*1e9,
      G_S.max()*1e9)


# ============================================================
# 4. MODEL TRAINING
# ============================================================

def create_model(alpha):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha)
    )


def fit_models(train_idx):
    models = {}

    for model_name, X, alpha in [
        ("M0_Smu", X_smu, ALPHA_SMU),
        ("M1_SmuPin", X_full, ALPHA_PIN)
    ]:
        c_model = create_model(alpha)
        g_model = create_model(alpha)

        c_model.fit(X[train_idx], target_C[train_idx])
        g_model.fit(X[train_idx], target_G[train_idx])

        models[model_name] = {
            "C": c_model,
            "G": g_model,
            "X": X
        }

    return models


# ============================================================
# 5. ABSOLUTE C/G EVALUATION
# ============================================================

def score(actual, predicted):
    err = predicted - actual

    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MaxAE": np.max(np.abs(err)),
        "Bias": np.mean(err)
    }


def evaluate_split(title, train_idx, test_idx):
    models = fit_models(train_idx)

    print("\n" + "="*70)
    print(title)
    print("="*70)

    print("Train:", len(train_idx))
    print("Test :", len(test_idx))

    print("Test signatures:",
          len(set(group_ids[test_idx])))

    rows = []

    for model_name, model in models.items():
        X = model["X"]

        # 예측 C_base + 고정 Local
        c_pred = (
            model["C"].predict(X[test_idx])
            + c_local_pf[test_idx]
        )

        # G는 직접 예측
        g_pred = model["G"].predict(X[test_idx])

        c_actual = C_F[test_idx] * 1e12
        g_actual = G_S[test_idx] * 1e9

        cs = score(c_actual, c_pred)
        gs = score(g_actual, g_pred)

        rows.append({
            "Model": model_name,
            "C_MAE_fF": cs["MAE"]*1000,
            "C_RMSE_fF": cs["RMSE"]*1000,
            "C_MaxAE_fF": cs["MaxAE"]*1000,
            "G_MAE_nS": gs["MAE"],
            "G_RMSE_nS": gs["RMSE"],
            "G_MaxAE_nS": gs["MaxAE"]
        })

    result_table = pd.DataFrame(rows)

    print(
        result_table.round(3).to_string(index=False)
    )

    return models, result_table


# ============================================================
# 6. RANDOM SPLIT
#
# 새로운 Pin 조합 예측 성능
# ============================================================

indices = np.arange(len(data))

train_random, test_random = train_test_split(
    indices,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED
)

random_models, random_scores = evaluate_split(
    "RANDOM HOLDOUT",
    train_random,
    test_random
)


# ============================================================
# 7. SIGNATURE HOLDOUT
#
# 학습에 없는 SMU 점유 구성 예측 성능
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED
)

train_sig, test_sig = next(
    splitter.split(
        X_full,
        groups=group_ids
    )
)

sig_models, sig_scores = evaluate_split(
    "UNSEEN SIGNATURE HOLDOUT",
    train_sig,
    test_sig
)


# ============================================================
# 8. PLOT: ABSOLUTE CAPACITANCE
# ============================================================

fig, axes = plt.subplots(
    1, 2, figsize=(12, 5)
)

for ax, model_set, idx, title in [
    (
        axes[0],
        random_models,
        test_random,
        "Random Holdout"
    ),
    (
        axes[1],
        sig_models,
        test_sig,
        "Unseen Signature"
    )
]:
    model = model_set["M1_SmuPin"]

    c_pred = (
        model["C"].predict(
            model["X"][idx]
        )
        + c_local_pf[idx]
    )

    c_actual = C_F[idx] * 1e12

    ax.scatter(
        c_actual, c_pred,
        alpha=0.7
    )

    lo = min(c_actual.min(), c_pred.min())
    hi = max(c_actual.max(), c_pred.max())

    ax.plot([lo, hi], [lo, hi], "--")

    ax.set_xlabel("Measured C (pF)")
    ax.set_ylabel("Predicted C (pF)")
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# 9. FULL DATA FIT FOR DEPLOYMENT
#
# 평가가 끝난 뒤 전체 데이터로 다시 피팅.
# 실제 예측 시 이 모델만 사용.
# ============================================================

full_idx = np.arange(len(data))

full_models = fit_models(full_idx)

deployed_model = full_models["M1_SmuPin"]

joblib.dump(
    {
        "C": deployed_model["C"],
        "G": deployed_model["G"],
        "local_coefficients_pf": LOCAL_C_PF,
        "frequency_hz": FREQ_HZ,
        "model_name": "Absolute Admittance v1"
    },
    "absolute_admittance_v1.joblib"
)


# ============================================================
# 10. INDEPENDENT PREDICTION
#
# 측정값, 기준값, 비교 조합 불필요.
# ============================================================

def predict_admittance(high, low, model=None):
    if model is None:
        model = deployed_model

    x_smu, x_pin, local_pf, _ = (
        extract_features(high, low)
    )

    x = np.r_[x_smu, x_pin].reshape(1, -1)

    c_pf = (
        model["C"].predict(x)[0]
        + local_pf
    )

    g_ns = model["G"].predict(x)[0]

    c_F_pred = c_pf * 1e-12
    g_S_pred = g_ns * 1e-9

    Y = (
        g_S_pred
        + 1j * OMEGA * c_F_pred
    )

    return {
        "C_pF": c_pf,
        "G_nS": g_ns,
        "Y_S": Y
    }


# ============================================================
# 11. EXAMPLE
# ============================================================

prediction = predict_admittance(
    high=list(range(15, 45)),
    low=list(range(7, 14))
)

print("\nINDEPENDENT PREDICTION")
print("C (pF):", prediction["C_pF"])
print("G (nS):", prediction["G_nS"])
print("Y (S):", prediction["Y_S"])
