
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 0. CONFIG
# ============================================================

MODE = "SharedPath"
PAIR_MODES = ["Sum", "MaxPerPin", "Sqrt", "PartialSum"]
ZONE_NAMES = ["Even", "Odd", "Dist2"]

N_FOLDS = 3
TEST_SIZE = 0.25
SEED = 42

# Coefficient shrinkage, units pF
PAIR_SIGMA = np.array([0.5, 0.5, 0.2])
RIDGE_WEIGHT = 1.0

# ============================================================
# 1. BUILD ZONE MASKS (44 x 44)
#
# Even: adjacent, larger pin is even
# Odd : adjacent, larger pin is odd
# Dist2: same SMU, distance = 2
#
# Cross-SMU excluded in this first comparison.
# ============================================================

smu_id = np.zeros(44, dtype=int)

for g, pins in enumerate(SMUS):
    for p in pins:
        smu_id[p - 1] = g

zone_masks = np.zeros((3, 44, 44), dtype=float)

for a in range(44):
    for b in range(44):
        if a == b or smu_id[a] != smu_id[b]:
            continue

        d = abs(a - b)

        if d == 1:
            z = 0 if max(a + 1, b + 1) % 2 == 0 else 1
        elif d == 2:
            z = 2
        else:
            continue

        zone_masks[z, a, b] = 1.0


# ============================================================
# 2. FOUR AGGREGATION OPERATORS
#
# Build one positive contribution per zone, then fit a
# signed coefficient per zone.
#
# For a given zone:
#   Sum = number of selected H-L pairs
#   MaxPerPin = 0.5 * (
#       number of H pins having >=1 pair in zone
#       + number of L pins having >=1 pair in zone
#   )
# ============================================================

def zone_aggregation_features(h, l, mode):
    feats = []

    for z in range(3):
        mask = zone_masks[z]

        # Shape: (N, 44)
        h_to_l = (h @ mask) * l
        l_to_h = (l @ mask.T) * h

        pair_sum = h_to_l.sum(axis=1)

        # Binary incidence count, not magnitude-weighted.
        h_covered = (l_to_h.sum(axis=1) > 0)
        l_covered = (h_to_l.sum(axis=1) > 0)

        # Correct per-row pin coverage requires counting
        # per-pin zone incidences, not scalar total.
        h_neighbors = h * (l @ mask.T)
        l_neighbors = l * (h @ mask)

        max_per_pin = 0.5 * (
            (h_neighbors > 0).sum(axis=1)
            + (l_neighbors > 0).sum(axis=1)
        )

        if mode == "Sum":
            value = pair_sum

        elif mode == "MaxPerPin":
            value = max_per_pin

        elif mode == "Sqrt":
            value = np.sqrt(
                np.maximum(pair_sum * max_per_pin, 0)
            )

        elif mode == "PartialSum":
            value = (
                max_per_pin
                + 0.25 * (pair_sum - max_per_pin)
            )

        else:
            raise ValueError(mode)

        feats.append(value)

    return np.column_stack(feats)


PAIR_FEATURES = {
    mode: zone_aggregation_features(H, L, mode)
    for mode in PAIR_MODES
}

# Sanity check: all feature values must be >= 0.
for mode, xp in PAIR_FEATURES.items():
    assert xp.shape == (len(H), 3)
    assert np.isfinite(xp).all()
    assert np.min(xp) >= 0


# ============================================================
# 3. FIT COEFFICIENTS TO BASE RESIDUALS
#
# Base predictions are given from the training split.
# No data leakage from holdout observations.
# ============================================================

def fit_pair_coefficients(train, c_base_train, xp):
    x = xp[train]
    target = C_TRUE[train] - c_base_train

    # Ridge penalty: alpha_z / sigma_z
    aug_x = np.vstack([
        x,
        np.diag(RIDGE_WEIGHT / PAIR_SIGMA)
    ])

    aug_y = np.r_[
        target,
        np.zeros(3)
    ]

    return np.linalg.lstsq(
        aug_x, aug_y, rcond=None
    )[0]


def metrics(actual, predicted):
    e = (predicted - actual) * 1000
    return {
        "RMSE_fF": float(np.sqrt(np.mean(e**2))),
        "MAE_fF": float(np.mean(np.abs(e))),
        "P95_fF": float(np.percentile(np.abs(e), 95)),
        "Bias_fF": float(np.mean(e))
    }


# ============================================================
# 4. IDENTICAL SPLITS FOR ALL MODELS
# ============================================================

rows = []
coeff_rows = []
pred_rows = []

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
        splitter.split(H, groups=groups), start=1
    ):
        # SharedPath + Individual baseline:
        # fitted using train rows only.
        t = fit_common(tr, MODE)
        v = fit_individual(tr, t, MODE)

        y_pin, z_shared = unpack_individual(t, v, MODE)

        c_base_tr, _ = predict_cg(
            H[tr], L[tr], y_pin, MODE, z_shared
        )
        c_base_te, _ = predict_cg(
            H[te], L[te], y_pin, MODE, z_shared
        )

        predictions = {"Base": c_base_te}

        for pair_mode, xp in PAIR_FEATURES.items():
            alpha = fit_pair_coefficients(
                tr, c_base_tr, xp
            )

            predictions[pair_mode] = (
                c_base_te + xp[te] @ alpha
            )

            coeff_rows.append({
                "Split": split_name,
                "Fold": fold,
                "Mode": pair_mode,
                "Even_pF": alpha[0],
                "Odd_pF": alpha[1],
                "Dist2_pF": alpha[2]
            })

        for name, pred in predictions.items():
            m = metrics(C_TRUE[te], pred)

            rows.append({
                "Split": split_name,
                "Fold": fold,
                "Model": name,
                **m
            })

            for row_idx, actual, estimated in zip(
                te, C_TRUE[te], pred
            ):
                pred_rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Row": int(row_idx),
                    "Model": name,
                    "Actual_pF": float(actual),
                    "Pred_pF": float(estimated)
                })

        print(
            split_name, fold,
            " | ".join(
                f"{name}: "
                f"{metrics(C_TRUE[te], p)['RMSE_fF']:.1f} fF"
                for name, p in predictions.items()
            )
        )


# ============================================================
# 5. SUMMARIES
# ============================================================

score_df = pd.DataFrame(rows)
coeff_df = pd.DataFrame(coeff_rows)
pred_df = pd.DataFrame(pred_rows)

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

print("\n=== PAIR AGGREGATION COMPARISON ===")
print(summary.round(3).to_string(index=False))

print("\n=== COEFFICIENT STABILITY ===")
print(
    coeff_df.groupby(["Split", "Mode"])[
        ["Even_pF", "Odd_pF", "Dist2_pF"]
    ].agg(["mean", "std"])
    .round(4)
    .to_string()
)

# Paired comparison against baseline by exact fold.
base_rmse = score_df[
    score_df["Model"] == "Base"
][["Split", "Fold", "RMSE_fF"]].rename(
    columns={"RMSE_fF": "Base_RMSE_fF"}
)

improvement = score_df.merge(
    base_rmse, on=["Split", "Fold"]
)

improvement["Improvement_fF"] = (
    improvement["Base_RMSE_fF"]
    - improvement["RMSE_fF"]
)

print("\n=== RMSE IMPROVEMENT OVER BASE ===")
print(
    improvement.groupby(["Split", "Model"])[
        "Improvement_fF"
    ].agg(["mean", "std"])
    .round(3)
    .to_string()
)

# ============================================================
# 6. OPTIONAL: FULL-DATA REFIT OF BEST MODE
#
# CV results must be examined before deployment.
# No automatic selection here.
# ============================================================

BEST_PAIR_MODE = "Sqrt"

all_idx = np.arange(len(H))

t_final = fit_common(all_idx, MODE)
v_final = fit_individual(all_idx, t_final, MODE)

y_pin_final, z_shared_final = unpack_individual(
    t_final, v_final, MODE
)

c_base_all, g_base_all = predict_cg(
    H, L, y_pin_final, MODE, z_shared_final
)

alpha_final = fit_pair_coefficients(
    all_idx,
    c_base_all,
    PAIR_FEATURES[BEST_PAIR_MODE]
)

print("\n=== FINAL COEFFICIENTS ===")
print(dict(zip(ZONE_NAMES, alpha_final)))

print("\n=== FULL-DATA RMSE ===")
print(
    metrics(
        C_TRUE,
        c_base_all
        + PAIR_FEATURES[BEST_PAIR_MODE] @ alpha_final
    )
)

joblib.dump({
    "base_model": "SharedPathIndividual",
    "pair_mode": BEST_PAIR_MODE,
    "y_pin_nS": y_pin_final,
    "z_shared_MOhm": z_shared_final,
    "pair_coeff_pF": alpha_final,
    "frequency_hz": FREQ
}, "shared_individual_pair_aggregation.joblib")

summary.to_csv(
    "pair_aggregation_comparison.csv", index=False
)

coeff_df.to_csv(
    "pair_aggregation_coefficients.csv", index=False
)

===============

import numpy as np

ref = PAIR_FEATURES["Sum"]

print("=== Feature differences from Sum ===")

for mode, xp in PAIR_FEATURES.items():
    diff = xp - ref

    print(
        f"{mode:12}",
        "max_diff =", np.max(np.abs(diff)),
        "different_rows =",
        np.sum(np.any(np.abs(diff) > 1e-10, axis=1))
    )

print("\n=== Pair multiplicity ===")

for z, name in enumerate(ZONE_NAMES):
    s = PAIR_FEATURES["Sum"][:, z]
    m = PAIR_FEATURES["MaxPerPin"][:, z]

    print(
        f"{name:8}",
        "Sum > Max rows =",
        np.sum(s > m + 1e-10),
        "Max difference =",
        np.max(s - m)
    )

print("\n=== Fold scores ===")
print(
    score_df.pivot_table(
        index=["Split", "Fold"],
        columns="Model",
        values="RMSE_fF"
    ).round(6).to_string()
)
