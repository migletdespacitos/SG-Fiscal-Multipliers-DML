import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

RNG_SEED = 42
np.random.seed(RNG_SEED)


# CONFIG

INPUT_CSV = "master_fiscal_multipliers_quarterly.csv"

OUTCOME_COL = "GDP_YoY_Growth"
CONFOUNDER_COLS = ["SORA_lag1", "SORA_lag2", "CPI_YoY_Inflation_lag1", "CPI_YoY_Inflation_lag2"]
TREATMENT_COLS = [
    "Exp_Economic_Development_QoQ_pct",
    "Exp_Social_Development_QoQ_pct",
    "Exp_Security_QoQ_pct",
]
SECTOR_LABELS = {
    "Exp_Economic_Development_QoQ_pct": "Economic Development",
    "Exp_Social_Development_QoQ_pct": "Social Development",
    "Exp_Security_QoQ_pct": "Security",
}

ML_METHOD = "ridge"
N_SPLITS = 5



# Nuisance model factory

def make_nuisance_model():
    if ML_METHOD == "ridge":
        return RidgeCV(alphas=np.logspace(-3, 3, 50))
    elif ML_METHOD == "forest":
        return RandomForestRegressor(
            n_estimators=500, max_depth=4, min_samples_leaf=3, random_state=RNG_SEED
        )
    else:
        raise ValueError(f"Unknown ML_METHOD: {ML_METHOD}")



# Cross-fitted residualization

def cross_fitted_residuals(X, target, n_splits=N_SPLITS):
    X_arr = X.to_numpy()
    y_arr = target.to_numpy()
    oof_pred = np.zeros_like(y_arr, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(X_arr):
        model = make_nuisance_model()
        model.fit(X_arr[train_idx], y_arr[train_idx])
        oof_pred[test_idx] = model.predict(X_arr[test_idx])

    residual = y_arr - oof_pred
    return pd.Series(residual, index=target.index, name=f"{target.name}_tilde")



# Final-stage causal estimation: OLS of Y_tilde on W_tilde, no intercept

def estimate_multiplier(y_tilde, w_tilde):
    w_tilde_2d = w_tilde.to_numpy().reshape(-1, 1)
    model = sm.OLS(y_tilde.to_numpy(), w_tilde_2d)
    result = model.fit(cov_type="HC3")

    coef = result.params[0]
    se = result.bse[0]
    pval = result.pvalues[0]
    ci_low, ci_high = result.conf_int(alpha=0.05)[0]

    return {
        "multiplier": coef,
        "std_error": se,
        "p_value": pval,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "n_obs": int(result.nobs),
        "r_squared": result.rsquared,
    }


# Main DML pipeline

def run_dml(df):
    Y = df[OUTCOME_COL]
    X = df[CONFOUNDER_COLS]

    y_tilde = cross_fitted_residuals(X, Y)

    results = {}
    for treatment_col in TREATMENT_COLS:
        sector_label = SECTOR_LABELS[treatment_col]
        W = df[treatment_col]

        w_tilde = cross_fitted_residuals(X, W)

        stats = estimate_multiplier(y_tilde, w_tilde)
        results[sector_label] = stats

        print("-" * 70)
        print(f"Sector: {sector_label}  ({treatment_col})")
        print(f"  Fiscal Multiplier (dY per 1pp QoQ spending shock): {stats['multiplier']:.4f}")
        print(f"  Std. Error (HC3-robust):                          {stats['std_error']:.4f}")
        print(f"  p-value:                                          {stats['p_value']:.4f}")
        print(f"  95% CI:                                           [{stats['ci_95_low']:.4f}, {stats['ci_95_high']:.4f}]")
        print(f"  N obs / Pseudo R^2:                                {stats['n_obs']} / {stats['r_squared']:.4f}")

    return results


def summarize(results):
    summary = pd.DataFrame(results).T
    summary = summary[["multiplier", "std_error", "p_value", "ci_95_low", "ci_95_high", "n_obs", "r_squared"]]
    summary.columns = ["Multiplier", "Std. Error", "p-value", "95% CI Low", "95% CI High", "N", "R²"]
    summary.index.name = "Sector"

    summary["Significant (5%)"] = summary["p-value"] < 0.05
    return summary.round(4)


def main():
    df = pd.read_csv(INPUT_CSV, index_col=0, parse_dates=True)

    missing_cols = set([OUTCOME_COL] + CONFOUNDER_COLS + TREATMENT_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV is missing expected columns: {missing_cols}")

    print("=" * 70)
    print(f"DOUBLE MACHINE LEARNING — SECTORAL FISCAL MULTIPLIERS")
    print(f"Nuisance ML method: {ML_METHOD}  |  Cross-fitting folds: {N_SPLITS}")
    print(f"Sample: {df.index.min().date()} -> {df.index.max().date()}  (N={len(df)})")
    print("=" * 70)

    results = run_dml(df)

    summary = summarize(results)
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE SUMMARY")
    print("=" * 70)
    print(summary.to_string())

    return summary


if __name__ == "__main__":
    summary_table = main()
