 sg-fiscal-multipliers-dml
 
**Causal estimates of Singapore's sectoral fiscal multipliers, using Double Machine Learning.**
 
This project asks a simple question with a genuinely hard causal inference problem underneath it: *does an unexpected quarter of government spending actually move GDP growth?* Naively regressing GDP on spending conflates the causal effect with reverse causality (a slowdown triggers more spending) and common-cause confounding (both spending and growth respond to the same macro conditions). This pipeline uses **Double Machine Learning (DML)** — Chernozhukov et al. (2018) — to strip that bias out and isolate the effect of a genuine, unanticipated fiscal shock.
 
Full write-up with the intuition, methodology and results: *[link to your Medium article here]*
 
---
 
## What's in this repo
 
| File | Purpose |
|---|---|
| `build_fiscal_master.py` | Ingests raw SingStat/MAS CSVs, cleans and aligns them into one quarterly master DataFrame |
| `dml_fiscal_multipliers.py` | Runs the DML estimation engine on the master DataFrame and outputs the causal multipliers |
| `master_fiscal_multipliers_quarterly.csv` | Generated output of the pipeline (produced by step 1, consumed by step 2) |
 
### Data sources (not redistributed — download fresh from source)
 
- `GrossDomesticProductYearOnYearGrowthRateQuarterly.csv` — SingStat, real GDP YoY growth
- `GovernmentOperatingExpenditureBySectorQuarterly.csv` — SingStat, expenditure by sector
- `ConsumerPriceIndexCPI2024AsBaseYearMonthly.csv` — SingStat, monthly CPI index (2024 base)
- `Domestic_Interest_Rates.csv` — MAS, daily SORA
Place all four in the repo root (or point the `*_FILE` constants at the top of `build_fiscal_master.py` to wherever you've saved them).
 
---
 
## Pipeline overview
 
**Stage 1 — `build_fiscal_master.py`**
- Reshapes SingStat's wide quarter-as-columns format into a tidy quarterly `DatetimeIndex`
- Resamples daily SORA and monthly CPI to quarterly, deriving YoY CPI inflation from the raw index level
- Converts expenditure levels to quarter-on-quarter % change (stationarity)
- Creates 1- and 2-quarter lags of SORA and CPI inflation (no look-ahead bias)
- Outputs `master_fiscal_multipliers_quarterly.csv`
**Stage 2 — `dml_fiscal_multipliers.py`**
- For each of the three spending sectors (Economic Development, Social Development, Security):
  - Predicts the sector's spending from lagged SORA/CPI confounders using a **cross-fitted** ML model (RidgeCV by default; RandomForestRegressor also supported)
  - Predicts GDP growth from the same confounders, same cross-fitting
  - Takes the residuals of both (the "surprise" component of each) and regresses one on the other, no intercept, HC3-robust SEs
- Cross-fitting (5-fold, chronological blocks) is what makes this genuinely "double" ML rather than a naive residual-on-residual regression — it prevents the nuisance models from overfitting and mechanically shrinking the treatment effect toward zero
---
 
## Requirements
 
```
python >= 3.10
pandas >= 2.2
numpy
scikit-learn
statsmodels
```
 
Install:
 
```bash
pip install pandas numpy scikit-learn statsmodels
```
 
---
 
## How to run
 
```bash
# 1. Build the clean quarterly master dataset from the raw CSVs
python build_fiscal_master.py
 
# 2. Run the DML estimation engine on that dataset
python dml_fiscal_multipliers.py
```
 
Step 1 prints a verification summary (shape, columns, head/tail, missing values) and writes `master_fiscal_multipliers_quarterly.csv` to the working directory. Pass `--no-save` to skip writing the CSV.
 
Step 2 reads that CSV, runs the DML loop for all three sectors, and prints a per-sector multiplier plus a side-by-side summary table.
 
To switch the nuisance estimator from Ridge to Random Forest, change `ML_METHOD = "ridge"` to `ML_METHOD = "forest"` at the top of `dml_fiscal_multipliers.py`.
 
---
 
## Results
 
| Sector | Multiplier | Std. Error | p-value | 95% CI |
|---|---|---|---|---|
| Economic Development | -0.0019 | 0.0113 | 0.87 | [-0.024, 0.020] |
| Social Development | 0.0049 | 0.0179 | 0.78 | [-0.030, 0.040] |
| Security | 0.0050 | 0.0189 | 0.79 | [-0.032, 0.042] |
 
*N = 77 quarters, 2006–2025.*
 
All three sectors show near-zero, statistically insignificant short-term multipliers. Read as a genuine reflection of Singapore's structure — a small, open economy with a high marginal propensity to import (spending leaks abroad before it can circulate domestically) and development expenditure that plays out over years, not quarters — rather than a null result. Full explanation in the linked article.
 
---
 
## Methodology notes / limitations
 
- Treatment (QoQ spending growth) and outcome (YoY GDP growth) are on different time bases, which likely dilutes the estimated signal — a natural next iteration is to test QoQ GDP growth as the outcome.
- Sample size (N=77 quarters) favours a regularised linear nuisance model (RidgeCV) over a more flexible learner (Random Forest); the latter is included for comparison but carries more variance risk at this N.
- Standard DML best practice (Chernozhukov et al., 2018) is followed for cross-fitting, but this is a single-country, single-frequency application — treat the point estimates as indicative rather than definitive causal parameters.
---
 
## Citation / methodology reference
 
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). *Double/debiased machine learning for treatment and structural parameters.* The Econometrics Journal, 21(1), C1–C68.
