
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

ABLATION_MODELS = [
    ("M1", "Sum"),
    ("M1", "Sqrt"),
    ("M2", "Sum"),
    ("M2", "Sqrt"),
]

RL_EPS_MOHM = 1e-6
N_ABLATION_FOLDS = 3

# Free: G/C (10:20), SharedPath (20:22)
GC_FREE_IDX = np.arange(10, 22)

lb, ub = bounds()

def remove_rl(t):
    x = np.asarray(t, dtype=float).copy()
    x[:5] = RL_EPS_MOHM
    x[5:10] = 0.0
    return x


# ============================================================
# 2. REFIT GC AND SHARED PATH WITH RL FROZEN
# ============================================================

def fit_gc_only(train, structure, aggregation):
    x0 = remove_rl(initial_params())

    def residual(free):
        t = x0.copy()
        t[GC_FREE_IDX] = free

        # Regularize SharedPath, not frozen R/L.
        reg_shared = REG_SHARED * t[20:22] / 10.0

        return np.r_[
            residual_data(train, t, structure, aggregation),
            reg_shared
        ]

    result = least_squares(
        residual,
        x0[GC_FREE_IDX],
        bounds=(lb[GC_FREE_IDX], ub[GC_FREE_IDX]),
        max_nfev=350,
        x_scale="jac",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8
    )

    t = x0.copy()
    t[GC_FREE_IDX] = result.x

    return t, result


# ============================================================
# 3. FULL RLGC REFIT
#
# Use GC-only optimum as one starting point,
# and the original two-stage fit as another.
#
# Compare their TRAIN objective and select
# the better converged solution.
# ============================================================

def fit_full_from_gc(train, structure, agg, gc_t):
    def objective(t):
        return np.r_[
            residual_data(train, t, structure, agg),
            regularization(t)
        ]

    opt = least_squares(
        objective,
        gc_t,
        bounds=(lb, ub),
        max_nfev=350,
        x_scale="jac",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8
    )

    return opt.x, opt


def fit_full_robust(train, structure, agg, gc_t):
    t1, opt1 = fit_full_from_gc(
        train, structure, agg, gc_t
    )

    t2, _, opt2 = fit_model(
        train, structure, agg
    )

    def loss(t):
        r = np.r_[
            residual_data(train, t, structure, agg),
            regularization(t)
        ]
        return np.sum(r*r)

    if loss(t1) <= loss(t2):
        return t1, opt1

    return t2, opt2


# ============================================================
# 4. DIAGNOSTICS
#
# A. Effect of pair RL in fitted Full model
# B. Effect of transmission-line factor
# ============================================================

def line_effect_diagnostics(indices, t, structure, agg):
    Z, Y, _ = decode(t)

    # M1: effective RLGC AFTER aggregation.
    # M2: category-level RLGC BEFORE aggregation.
    if structure == "M1":
        combinations = [
            predict_M1(PAIR_MATRICES[k], Z, Y, agg)[1:]
            for k in indices
        ]
    else:
        combinations = [(z, y) for z, y in zip(Z, Y)]

    x2 = np.array([
        1e-3*z*y for z, y in combinations
    ], dtype=complex)

    x = np.sqrt(x2)

    factor = np.ones_like(x)
    small = np.abs(x2) < 1e-7

    factor[small] = (
        1 - x2[small]/3 + 2*x2[small]**2/15
    )
    factor[~small] = (
        np.tanh(x[~small]) / x[~small]
    )

    epsilon = np.abs(factor - 1)

    return {
        "GammaL_Mean": float(np.mean(np.abs(x))),
        "GammaL_Max": float(np.max(np.abs(x))),
        "LineEffect_Mean": float(np.mean(epsilon)),
        "LineEffect_Max": float(np.max(epsilon))
    }


def calc_rmse(indices, t, structure, agg):
    return metrics(
        indices,
        predict_rows(indices, t, structure, agg)
    )["C_RMSE_fF"]


# ============================================================
# 5. MATCHED CV
# ============================================================

rows = []
param_rows = []

for split_name, groups in [
    ("RandomConfig", CONFIG),
    ("UnseenSignature", SIGNATURE)
]:
    cv = GroupShuffleSplit(
        n_splits=N_ABLATION_FOLDS,
        test_size=0.25,
        random_state=SEED
    )

    for fold, (tr, te) in enumerate(
        cv.split(np.arange(N), groups=groups),
        start=1
    ):
        for structure, agg in ABLATION_MODELS:
            key = f"{structure}-{agg}"

            # Model 1: refitted GC-only.
            t_gc, opt_gc = fit_gc_only(tr, structure, agg)

            # Model 2: independently refitted full RLGC.
            t_full, opt_full = fit_full_robust(
                tr, structure, agg, t_gc
            )

            # Model 3: full parameters, remove only R/L.
            t_removed = remove_rl(t_full)

            fitted = {
                "GC-only": t_gc,
                "RLGC-full": t_full,
                "RL-removed": t_removed
            }

            for name, t in fitted.items():
                train_metric = metrics(
                    tr,
                    predict_rows(tr, t, structure, agg)
                )

                test_metric = metrics(
                    te,
                    predict_rows(te, t, structure, agg)
                )

                rows.append({
                    "Split": split_name,
                    "Fold": fold,
                    "Model": key,
                    "Variant": name,
                    "Train_C_RMSE_fF":
                        train_metric["C_RMSE_fF"],
                    **test_metric
                })

            diag = line_effect_diagnostics(
                te, t_full, structure, agg
            )

            Z, Y, zs = decode(t_full)

            param_rows.append({
                "Split": split_name,
                "Fold": fold,
                "Model": key,
                **diag,
                "R_Mean_MOhm": float(np.mean(t_full[:5])),
                "L_Mean_H": float(np.mean(t_full[5:10])),
                "Shared_Abs_MOhm": float(abs(zs)),
                "GC_Success": bool(opt_gc.success),
                "Full_Success": bool(opt_full.success)
            })

            result_str = " | ".join(
                f"{name}={calc_rmse(te,t,structure,agg):.1f}"
                for name, t in fitted.items()
            )

            print(
                f"{split_name:16} F{fold} "
                f"{key:8} | {result_str} fF | "
                f"epsilon={diag['LineEffect_Mean']:.5g}"
            )


# ============================================================
# 6. REPORT
# ============================================================

ablation_df = pd.DataFrame(rows)
ablation_params = pd.DataFrame(param_rows)

summary = (
    ablation_df.groupby(["Split", "Model", "Variant"])
    .agg(
        C_RMSE_Mean=("C_RMSE_fF", "mean"),
        C_RMSE_Std=("C_RMSE_fF", "std"),
        G_RMSE_Mean=("G_RMSE_nS", "mean"),
        Train_C_RMSE=("Train_C_RMSE_fF", "mean")
    )
    .reset_index()
)

print("\n=== ABLATION SUMMARY ===")
print(summary.round(4).to_string(index=False))

# Paired comparison by the SAME Split and Fold.
wide = ablation_df.pivot_table(
    index=["Split", "Fold", "Model"],
    columns="Variant",
    values="C_RMSE_fF"
).reset_index()

wide["RL_Benefit_fF"] = (
    wide["GC-only"] - wide["RLGC-full"]
)

wide["RL_Removal_Penalty_fF"] = (
    wide["RL-removed"] - wide["RLGC-full"]
)

print("\n=== RL CONTRIBUTION ===")
print(
    wide.groupby(["Split", "Model"])[
        ["RL_Benefit_fF", "RL_Removal_Penalty_fF"]
    ].agg(["mean", "std"])
    .round(4).to_string()
)

print("\n=== TRANSMISSION-LINE EFFECT ===")
print(
    ablation_params.groupby(["Split", "Model"])[
        [
            "GammaL_Mean",
            "LineEffect_Mean",
            "LineEffect_Max",
            "R_Mean_MOhm",
            "L_Mean_H",
            "Shared_Abs_MOhm"
        ]
    ].agg(["mean", "std"])
    .round(6).to_string()
)

ablation_df.to_csv(
    "rlgc_ablation_scores.csv", index=False
)

ablation_params.to_csv(
    "rlgc_ablation_diagnostics.csv", index=False
)
