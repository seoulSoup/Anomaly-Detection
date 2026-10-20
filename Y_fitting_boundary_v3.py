
import ast
import re
import numpy as np
import pandas as pd
import joblib

from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# SETTINGS
# ============================================================

FREQ = 10000.0
OMEGA = 2 * np.pi * FREQ

# Existing local increments [pF] - FIXED
LOCAL_PF = np.array([1.544, 0.605, 0.175])

SMUS = [
    range(1, 7),
    range(7, 15),
    range(15, 23),
    range(23, 31),
    range(31, 39),
    range(39, 45)
]

PIN_SMU = {
    p: g for g, group in enumerate(SMUS)
    for p in group
}

# SMU boundary endpoint pairs
EDGES = [(6, 7), (14, 15), (22, 23),
         (30, 31), (38, 39)]

N_FOLDS = 3
TEST_SIZE = 0.25
SEED = 42

# Regularization starting values.
# These are tuning parameters, NOT measured physics.
PIN_LOG_SIGMA = 0.25
GROUP_LOG_SIGMA = 0.60
REG_SCALE_PF = 0.15
BOUNDARY_SIGMA_PF = 0.35

# Pin starting capacitance (pF)
INITIAL_C_PF = 0.6

# theta = 44 log(pin C) + 10 boundary C coefficients
N_PIN = 44
N_BOUNDARY = 10
N_PARAMS = 54


# ============================================================
# PARSE AND FEATURES
# ============================================================

def pins_tuple(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        p = [int(v) for v in value]
    elif isinstance(value, str):
        try:
            obj = ast.literal_eval(value)
            if isinstance(obj, (list, tuple, set)):
                p = [int(v) for v in obj]
            else:
                p = [int(v) for v in re.findall(r"\d+", value)]
        except (SyntaxError, ValueError):
            p = [int(v) for v in re.findall(r"\d+", value)]
    else:
        raise ValueError(f"Invalid pins: {value}")

    if not p or len(set(p)) != len(p):
        raise ValueError(f"Empty/duplicate pins: {value}")

    if not all(1 <= k <= 44 for k in p):
        raise ValueError("Pin out of 1..44")

    return tuple(sorted(p))


def make_features(high, low):
    high = pins_tuple(high)
    low = pins_tuple(low)

    if set(high) & set(low):
        raise ValueError("High/Low overlap")

    h = np.zeros(44)
    l = np.zeros(44)

    h[np.array(high) - 1] = 1
    l[np.array(low) - 1] = 1

    hc = np.array([
        sum(p in high for p in group)
        for group in SMUS
    ])
    lc = np.array([
        sum(p in low for p in group)
        for group in SMUS
    ])

    # Fixed local increments: same-SMU crossing pairs
    ne = no = n2 = 0

    for a in high:
        for b in low:
            if PIN_SMU[a] != PIN_SMU[b]:
                continue

            d = abs(a - b)

            if d == 1:
                if max(a, b) % 2 == 0:
                    ne += 1
                else:
                    no += 1
            elif d == 2:
                n2 += 1

    local = np.dot(LOCAL_PF, [ne, no, n2])

    # Boundary A:
    # Adjacent SMUs occupied across opposite polarities
    cross = np.array([
        float(
            (hc[g] > 0 and lc[g+1] > 0)
            or
            (lc[g] > 0 and hc[g+1] > 0)
        )
        for g in range(5)
    ])

    # Boundary B:
    # Exact adjacent pins across SMU boundary
    near = np.array([
        h[a-1]*l[b-1] + l[a-1]*h[b-1]
        for a, b in EDGES
    ])

    boundary = np.r_[cross, near]

    signature = tuple(np.r_[hc, lc])
    configuration = (high, low)

    return h, l, local, boundary, signature, configuration


# ============================================================
# PREPARE MEASUREMENT DATA
# ============================================================

data = df.copy().reset_index(drop=True)

records = [
    make_features(h, l)
    for h, l in zip(
        data["high_pins"], data["low_pins"]
    )
]

H = np.vstack([r[0] for r in records])
L = np.vstack([r[1] for r in records])
LOCAL = np.array([r[2] for r in records])
BOUNDARY = np.vstack([r[3] for r in records])

SIGNATURE = np.array(
    [str(r[4]) for r in records]
)
CONFIG = np.array(
    [str(r[5]) for r in records]
)

# df C must be Farads.
C_TRUE = pd.to_numeric(
    data["C"]
).to_numpy(dtype=float) * 1e12

C_BASE = C_TRUE - LOCAL

if not np.isfinite(C_TRUE).all():
    raise ValueError("Non-finite C values")

print("Measurements:", len(data))
print("Unique Configurations:", len(set(CONFIG)))
print("Unique Signatures:", len(set(SIGNATURE)))
print("Boundary A counts:", BOUNDARY[:, :5].sum(axis=0))
print("Boundary B counts:", BOUNDARY[:, 5:].sum(axis=0))


# ============================================================
# PIN ADMITTANCE + BOUNDARY MODEL
#
# c_i = exp(q_i)  [pF]
#
# Ch = sum_H(c_i)
# Cl = sum_L(c_i)
# Cline = Ch*Cl/(Ch+Cl)
#
# Cpred = Cline + Clocal + B @ coefficients
# ============================================================

SMU_IDX = [
    np.array(list(group)) - 1
    for group in SMUS
]


def decode_theta(theta):
    c_pin = np.exp(theta[:44])
    b = theta[44:54]
    return c_pin, b


def predict_c_pf(theta, h, l, local, boundary,
                 use_boundary=True):

    c_pin, b = decode_theta(theta)

    ch = h @ c_pin
    cl = l @ c_pin

    c_line = np.divide(
        ch * cl, ch + cl,
        out=np.zeros_like(ch),
        where=(ch + cl) > 0
    )

    if use_boundary:
        c_boundary = boundary @ b
    else:
        c_boundary = np.zeros(len(h))

    return c_line + local + c_boundary


def regularization(theta, use_boundary):
    q = theta[:44]
    b = theta[44:54]

    reg = []

    group_means = np.array([
        np.mean(q[idx]) for idx in SMU_IDX
    ])

    # Pin deviations from SMU group means
    for g, idx in enumerate(SMU_IDX):
        reg.extend(
            REG_SCALE_PF
            * (q[idx] - group_means[g])
            / PIN_LOG_SIGMA
        )

    # Shrink SMU means toward their global mean.
    reg.extend(
        REG_SCALE_PF
        * (group_means - group_means.mean())
        / GROUP_LOG_SIGMA
    )

    if use_boundary:
        reg.extend(
            REG_SCALE_PF * b / BOUNDARY_SIGMA_PF
        )

    return np.asarray(reg)


def fit_model(train, use_boundary=True):
    q0 = np.full(44, np.log(INITIAL_C_PF))
    theta0 = np.r_[q0, np.zeros(10)]

    if not use_boundary:
        # Fit only pin parameters.
        initial = theta0[:44]
    else:
        initial = theta0

    def unpack(t):
        if use_boundary:
            return t
        return np.r_[t, np.zeros(10)]

    def residual(t):
        theta = unpack(t)

        pred = predict_c_pf(
            theta,
            H[train],
            L[train],
            LOCAL[train],
            BOUNDARY[train],
            use_boundary=use_boundary
        )

        reg = regularization(
            theta, use_boundary
        )

        return np.r_[
            pred - C_TRUE[train],
            reg
        ]

    lower = np.full(len(initial), -np.inf)
    upper = np.full(len(initial), np.inf)

    # Prevent unrealistically extreme effective branches.
    lower[:44] = np.log(0.02)
    upper[:44] = np.log(20.0)

    result = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        max_nfev=600,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9
    )

    if not result.success:
        print("Fit warning:", result.message)

    return unpack(result.x)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(true, pred):
    e = (pred - true) * 1000  # pF -> fF

    return {
        "RMSE_fF": np.sqrt(np.mean(e**2)),
        "MAE_fF": np.mean(np.abs(e)),
        "P95_fF": np.percentile(np.abs(e), 95),
        "Bias_fF": np.mean(e)
    }


scores = []
fitted = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    splitter = GroupShuffleSplit(
        n_splits=N_FOLDS,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    for fold, (tr, te) in enumerate(
        splitter.split(H, groups=groups),
        start=1
    ):
        for name, enable_boundary in [
            ("PinOnly", False),
            ("PinBoundary", True)
        ]:
            theta = fit_model(
                tr, use_boundary=enable_boundary
            )

            c_pred = predict_c_pf(
                theta,
                H[te],
                L[te],
                LOCAL[te],
                BOUNDARY[te],
                use_boundary=enable_boundary
            )

            stats = evaluate(
                C_TRUE[te], c_pred
            )

            scores.append({
                "Split": split_name,
                "Fold": fold,
                "Model": name,
                **stats
            })

            fitted.append({
                "Split": split_name,
                "Fold": fold,
                "Model": name,
                "theta": theta
            })


score_df = pd.DataFrame(scores)

summary = (
    score_df.groupby(["Split", "Model"])
    .agg(
        RMSE_mean=("RMSE_fF", "mean"),
        RMSE_std=("RMSE_fF", "std"),
        MAE_mean=("MAE_fF", "mean"),
        P95_mean=("P95_fF", "mean"),
        Bias_mean=("Bias_fF", "mean")
    )
    .reset_index()
)

print("\n=== ABSOLUTE C MODEL COMPARISON ===")
print(summary.round(2).to_string(index=False))


# ============================================================
# FINAL FIT USING ALL MEASUREMENTS
# ============================================================

all_idx = np.arange(len(data))
theta_final = fit_model(all_idx, use_boundary=True)

c_pin, b = decode_theta(theta_final)

# Effective capacitive impedance at 10 kHz
z_pin = -1j / (OMEGA * c_pin * 1e-12)

pin_table = pd.DataFrame({
    "Pin": np.arange(1, 45),
    "SMU": [PIN_SMU[i] + 1 for i in range(1, 45)],
    "Effective_C_pF": c_pin,
    "Abs_Z_MOhm": np.abs(z_pin) / 1e6
})

boundary_table = pd.DataFrame({
    "Boundary": [
        "SMU1-2", "SMU2-3", "SMU3-4",
        "SMU4-5", "SMU5-6"
    ],
    "CrossOccupied_pF": b[:5],
    "DirectAdjacent_pF": b[5:]
})

print("\n=== PIN IMPEDANCES ===")
print(
    pin_table.round(4).to_string(index=False)
)

print("\n=== BOUNDARY CORRECTIONS ===")
print(
    boundary_table.round(4).to_string(index=False)
)


# ============================================================
# INDEPENDENT ABSOLUTE PREDICTION
# ============================================================

def predict_capacitance(high, low, theta=None):
    if theta is None:
        theta = theta_final

    h, l, local, boundary, _, _ = (
        make_features(high, low)
    )

    c_pf = predict_c_pf(
        theta,
        h[None, :],
        l[None, :],
        np.array([local]),
        boundary[None, :],
        use_boundary=True
    )[0]

    # Capacitive component of admittance.
    y_imag = OMEGA * c_pf * 1e-12

    return {
        "C_pF": float(c_pf),
        "ImagY_S": float(y_imag),
        "Y_cap_S": complex(0, y_imag)
    }


print("\n=== EXAMPLE ===")
example = predict_capacitance(
    high=list(range(15, 45)),
    low=list(range(7, 14))
)
print(example)

joblib.dump(
    {
        "theta": theta_final,
        "frequency": FREQ,
        "local_coefficients_pf": LOCAL_PF,
        "boundary_edges": EDGES
    },
    "pin_boundary_cap_model.joblib"
)

score_df.to_csv(
    "pin_boundary_cv.csv", index=False
)
pin_table.to_csv(
    "pin_effective_impedances.csv", index=False
)
boundary_table.to_csv(
    "boundary_corrections.csv", index=False
)
