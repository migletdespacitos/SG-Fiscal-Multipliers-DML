import re
import argparse

import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 15)


# CONFIG

GDP_FILE = "GrossDomesticProductYearOnYearGrowthRateQuarterly.csv"
EXP_FILE = "GovernmentOperatingExpenditureBySectorQuarterly.csv"
CPI_FILE = "ConsumerPriceIndexCPI2024AsBaseYearMonthly.csv"
SORA_FILE = "Domestic_Interest_Rates.csv"
OUTPUT_CSV = "master_fiscal_multipliers_quarterly.csv"

try:
    pd.Series(dtype="float64", index=pd.DatetimeIndex([])).resample("QE")
    QUARTER_FREQ = "QE"
except ValueError:
    QUARTER_FREQ = "Q"

GDP_SERIES_NAME = "GDP In Chained (2015) Dollars"

EXPENDITURE_SECTORS = {
    "Economic Development": "Economic Development",
    "Social Development": "Social Development",
    "Security": "Security And External Relations",
}

SAMPLE_START = "2006-01-01"  # per spec: "span from 2006 to the latest available quarter"


# STEP 2 (helper): Generic SingStat wide-format loader

def _find_header_row(path, marker="DataSeries", max_scan=30):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > max_scan:
                break
            first_cell = line.split(",")[0].strip().strip('"')
            if first_cell == marker:
                return i
    return 0  # fall back to assuming a clean file


def load_singstat_wide(path, quarter_pattern=r"^(\d{4})(\d)Q$"):
    header_row = _find_header_row(path)
    raw = pd.read_csv(path, skiprows=header_row)

    quarter_cols = [c for c in raw.columns if re.match(quarter_pattern, str(c).strip())]

    raw = raw.dropna(subset=["DataSeries"])
    raw["DataSeries"] = raw["DataSeries"].astype(str).str.strip()

    numeric_block = raw[quarter_cols].apply(pd.to_numeric, errors="coerce")
    has_numeric = numeric_block.notna().any(axis=1)
    raw = raw.loc[has_numeric].copy()
    numeric_block = numeric_block.loc[has_numeric]

    raw = raw.set_index("DataSeries")
    numeric_block.index = raw.index

    wide = numeric_block.T

    def _quarter_col_to_ts(col):
        y, q = re.match(quarter_pattern, col.strip()).groups()
        return pd.Period(f"{y}Q{q}", freq="Q").to_timestamp(how="end").normalize()

    wide.index = [_quarter_col_to_ts(c) for c in wide.index]
    wide.index.name = "quarter"
    wide = wide.sort_index()

    return wide


# STEP 1: Daily -> Quarterly resampling of SORA

def load_sora_quarterly(path):
    raw = pd.read_csv(path, skiprows=5, header=None,
                       names=["Year", "Month", "Day", "PubDate", "SORA"])

    # Keep only rows whose Publication Date matches a real "DD Mon YYYY"
    is_valid_row = raw["PubDate"].astype(str).str.match(r"^\d{1,2}\s\w{3}\s\d{4}$", na=False)
    raw = raw.loc[is_valid_row].copy()

    raw["Year"] = raw["Year"].ffill()
    raw["Month"] = raw["Month"].ffill()

    day_str = raw["Day"].astype(float).astype(int).astype(str)
    raw["ValueDate"] = pd.to_datetime(
        raw["Year"].astype(str) + "-" + raw["Month"].astype(str) + "-" + day_str,
        format="%Y-%b-%d",
        errors="coerce",
    )

    raw["SORA"] = pd.to_numeric(raw["SORA"], errors="coerce")

    sora = raw.set_index("ValueDate")[["SORA"]].sort_index()

    # Forward-fill any non-numeric/missing entries (e.g. public holidays that slipped through, data-vendor gaps) using the last available published rate -- standard practice for overnight rate series.
    sora["SORA"] = sora["SORA"].ffill()

    sora_q = sora["SORA"].resample(QUARTER_FREQ).mean().to_frame("SORA")
    sora_q.index.name = "quarter"
    return sora_q


# STEP 2: GDP, Expenditure, CPI ingestion

def load_gdp_quarterly(path):
    wide = load_singstat_wide(path)
    gdp = wide[[GDP_SERIES_NAME]].rename(columns={GDP_SERIES_NAME: "GDP_YoY_Growth"})
    return gdp


def load_expenditure_quarterly(path):
    wide = load_singstat_wide(path)

    normalized = {re.sub(r"\s+", " ", c).strip(): c for c in wide.columns}

    out = {}
    for clean_name, match_str in EXPENDITURE_SECTORS.items():
        hits = [orig for norm, orig in normalized.items() if match_str.lower() in norm.lower()]
        if not hits:
            raise ValueError(f"Could not find expenditure sector matching '{match_str}' in {path}")
        out[clean_name] = wide[hits[0]]

    exp = pd.DataFrame(out)
    exp.columns = [f"Exp_{c.replace(' ', '_')}" for c in exp.columns]
    return exp


def load_cpi_quarterly(path):
    header_row = _find_header_row(path)
    raw = pd.read_csv(path, skiprows=header_row)

    month_cols = [c for c in raw.columns if re.match(r"^\d{4}[A-Za-z]{3}$", str(c).strip())]

    row = raw.loc[raw["DataSeries"].astype(str).str.strip() == "All Items"]
    if row.empty:
        raise ValueError(f"'All Items' CPI row not found in {path}")

    series = row[month_cols].iloc[0]
    series = pd.to_numeric(series, errors="coerce")  # 'na' strings -> NaN

    dates = pd.to_datetime(month_cols, format="%Y%b") + pd.offsets.MonthEnd(0)
    cpi_m = pd.Series(series.values, index=dates).sort_index()
    cpi_m.name = "CPI_Index"

    # Monthly -> Quarterly average index level
    cpi_q_index = cpi_m.resample(QUARTER_FREQ).mean()

    # Derive YoY inflation (%) from the quarterly index: compare each quarter to the same quarter one year (4 quarters) prior.
    cpi_q_inflation = cpi_q_index.pct_change(periods=4) * 100

    cpi = pd.DataFrame({"CPI_YoY_Inflation": cpi_q_inflation})
    cpi.index.name = "quarter"
    return cpi



# STEP 3: Merge & filter

def build_raw_master():
    gdp = load_gdp_quarterly(GDP_FILE)
    exp = load_expenditure_quarterly(EXP_FILE)
    cpi = load_cpi_quarterly(CPI_FILE)
    sora = load_sora_quarterly(SORA_FILE)

    master = gdp.join(exp, how="inner").join(cpi, how="inner").join(sora, how="inner")

    master = master.loc[master.index >= SAMPLE_START]
    master = master.sort_index()
    return master



# STEP 4: Econometric transformations (stationarity + lags)

def apply_econometric_transforms(master):
    df = master.copy()

    exp_cols = [c for c in df.columns if c.startswith("Exp_")]

    # Stationarity: raw expenditure LEVELS are almost certainly I(1) (trending with nominal GDP/population growth), so we difference them into Quarter-on-Quarter percentage changes. GDP growth and CPI inflation are already rates (stationary-ish by construction), so they are left as-is per spec.
    for col in exp_cols:
        df[f"{col}_QoQ_pct"] = df[col].pct_change(periods=1) * 100
    df = df.drop(columns=exp_cols)  # drop raw levels, keep only QoQ growth

    # Lags: create 1- and 2-quarter lags of the macro confounders (SORA, CPI inflation) so that at time t the model only ever sees information that was actually available before t -- avoiding look-ahead bias.
    for confounder in ["SORA", "CPI_YoY_Inflation"]:
        for lag in (1, 2):
            df[f"{confounder}_lag{lag}"] = df[confounder].shift(lag)

    df = df.dropna(how="any")

    return df


# STEP 5: Verification output

def main(save_csv=True):
    raw_master = build_raw_master()
    final = apply_econometric_transforms(raw_master)

    print("=" * 70)
    print("MASTER DATAFRAME — VERIFICATION")
    print("=" * 70)
    print(f"Shape: {final.shape}")
    print(f"Date range: {final.index.min().date()} -> {final.index.max().date()}")
    print("\nColumns:")
    for c in final.columns:
        print(f"  - {c}")
    print("\nHead:")
    print(final.head())
    print("\nTail:")
    print(final.tail())
    print("\nDtypes:")
    print(final.dtypes)
    print("\nMissing values per column:")
    print(final.isna().sum())

    if save_csv:
        final.to_csv(OUTPUT_CSV)
        print(f"\nSaved master DataFrame -> {OUTPUT_CSV}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the quarterly master macro-fiscal DataFrame for the "
        "Singapore Sectoral Fiscal Multipliers project."
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing the output CSV.")
    args = parser.parse_args()

    master_df = main(save_csv=not args.no_save)
