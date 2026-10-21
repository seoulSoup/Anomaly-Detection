
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import pandas as pd

PAIR_MODES = ["Sum", "MaxPerPin", "Sqrt", "PartialSum"]
MODE = "SharedPath"

# Positive physical-increment candidates [pF].
# Initial estimates are only starting values.
ALPHA_INIT = np.array([1.0, 0.6, 0.15])

# Weak zero-centered regularization.
ALPHA_SIGMA = np.array([2.0, 2.0, 1.0])
REG_WEIGHT = 0.1


def aggregate_pair_matrix(h, l, alpha, mode):
    """
    h, l: (N, 44) binary matrices
    alpha: Even/Odd/Dist2 coefficients, pF

    Returns total pair correction, pF.
    """

    # Weighted pair matrix W, shape (44, 44)
    W = np.einsum("z,zab->ab", alpha, zone_masks)

    # Only cross-selected High/Low pairs.
    # Shape (N, 44, 44).
    pairs = h[:, :, None] * l[:, None, :] * W[None, :, :]

    pair_sum = pairs.sum(axis=(1, 2))

    max_h = pairs.max(axis=2).sum(axis=1)
    max_l = pairs.max(axis=1).sum(axis=1)

    max_pin = 0.5 * (max_h + max_l)

    if mode == "Sum":
        return pair_sum

    if mode == "MaxPerPin":
        return max_pin

    if mode == "Sqrt":
        return np.sqrt(np.maximum(pair_sum * max_pin, 0.0))

    if mode == "PartialSum":
        return max_pin + 0.25 * (pair_sum - max_pin)

    raise ValueError(mode)


def fit_pair_nonlinear(tr, c_base_tr, mode):
    target = C_TRUE[tr] - c_base_tr

    def residual(alpha):
        correction = aggregate_pair_matrix(
            H[tr], L[tr], alpha, mode
        )

        reg = REG_WEIGHT * alpha / ALPHA_SIGMA

        return np.r_[
            (correction - target) / 0.5,
            reg
        ]

    result = least_squares(
        residual,
        ALPHA_INIT,
        bounds=(np.zeros(3), np.full(3, 5.0)),
        max_nfev=300,
        x_scale="jac"
    )

    return result.x


rows = []
coef_rows = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    cv = GroupShuffleSplit(
        n_splits=3,
        test_size=0.25,
        random_state=42
    )

    for fold, (tr, te) in enumerate(
        cv.split(H, groups=groups), 1
    ):

        # Fit Base on train only.
        t = fit_common(tr, MODE)
        v = fit_individual(tr, t, MODE)

        yp, zs = unpack_individual(t, v, MODE)

        c_tr, _ = predict_cg(
            H[tr], L[tr], yp, MODE, zs
        )

        c_te, _ = predict_cg(
            H[te], L[te], yp, MODE, zs
        )

        # Base
        rows.append({
            "Split": split_name,
            "Fold": fold,
            "Mode": "Base",
            **metrics(C_TRUE[te], c_te)
        })

        for pm in PAIR_MODES:
            alpha = fit_pair_nonlinear(tr, c_tr, pm)

            correction = aggregate_pair_matrix(
                H[te], L[te], alpha, pm
            )

            prediction = c_te + correction

            rows.append({
                "Split": split_name,
                "Fold": fold,
                "Mode": pm,
                **metrics(C_TRUE[te], prediction)
            })

            coef_rows.append({
                "Split": split_name,
                "Fold": fold,
                "Mode": pm,
                "Even_pF": alpha[0],
                "Odd_pF": alpha[1],
                "Dist2_pF": alpha[2],
                "Correction_MAE_fF":
                    np.mean(np.abs(correction)) * 1000
            })

result_df = pd.DataFrame(rows)
coef_df = pd.DataFrame(coef_rows)

print("\n=== RMSE PER FOLD ===")
print(
    result_df.pivot_table(
        index=["Split", "Fold"],
        columns="Mode",
        values="RMSE_fF"
    ).round(3).to_string()
)

print("\n=== AVERAGE SCORES ===")
print(
    result_df.groupby(["Split", "Mode"])[
        ["RMSE_fF", "MAE_fF", "P95_fF"]
    ].agg(["mean", "std"])
    .round(3).to_string()
)

print("\n=== COEFFICIENTS ===")
print(coef_df.round(4).to_string(index=False))




===============<=<÷>÷[÷[÷>=</<=>÷


import numpy as np
import pandas as pd

diagnostics = {}
predictions = {}

for pm in PAIR_MODES:
    alpha = fit_pair_nonlinear(tr, c_tr, pm)

    correction = aggregate_pair_matrix(
        H[te], L[te], alpha, pm
    )

    predictions[pm] = c_te + correction

    diagnostics[pm] = {
        "Even_pF": alpha[0],
        "Odd_pF": alpha[1],
        "Dist2_pF": alpha[2],
        "Correction_mean_fF": correction.mean() * 1000,
        "Correction_std_fF": correction.std() * 1000,
        "Correction_max_fF": correction.max() * 1000,
        "RMSE_fF": np.sqrt(
            np.mean(
                (predictions[pm] - C_TRUE[te])**2
            )
        ) * 1000
    }

print(pd.DataFrame(diagnostics).T.round(7))

print("\n=== Pairwise prediction differences [fF] ===")
for i, m1 in enumerate(PAIR_MODES):
    for m2 in PAIR_MODES[i+1:]:
        delta = (
            predictions[m1] - predictions[m2]
        ) * 1000

        print(
            f"{m1:12} vs {m2:12}",
            f"MeanAbs={np.abs(delta).mean():.9f}",
            f"MaxAbs={np.abs(delta).max():.9f}"
        )

print("\n=== Base residual sign ===")
residual = C_TRUE[tr] - c_tr

print("Positive residual ratio:", np.mean(residual > 0))
print("Negative residual ratio:", np.mean(residual < 0))
print("Mean residual [fF]:", residual.mean() * 1000)

