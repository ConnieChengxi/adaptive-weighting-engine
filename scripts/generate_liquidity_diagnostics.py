from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "logs" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_weighting.data.preprocess import load_and_clean_price_csv
from adaptive_weighting.factors.liquidity import compute_average_dollar_volume
from adaptive_weighting.factors.spread import compute_corwin_schultz_spread
from adaptive_weighting.factors.standardize import cross_sectional_zscore


BASE_CONFIG = ROOT / "config" / "base.yaml"
FACTORS_CONFIG = ROOT / "config" / "factors.yaml"
MONTHLY_PANEL_PATH = ROOT / "data" / "processed" / "monthly_factor_panel.csv"
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
BACKTEST_DIR = ROOT / "outputs" / "backtests"
APPENDIX_I_LIQUIDITY_LOOKUP_PATH = BACKTEST_DIR / "baseline_equal_weight_benchmark_selections.csv"
MAIN_FRAMEWORK = "holding_buffer_top6"
MAIN_RESULT_PREFIX = f"common_shrinkage_{MAIN_FRAMEWORK}"
PRE_TEST_SIGNAL_END = pd.Timestamp("2021-05-31")
PRE_TEST_OUTCOME_END = pd.Timestamp("2021-06-30")
APPENDIX_I_MODEL_ORDER = ["S1", "A1", "L1", "L2", "L3", "T1", "T2"]


LIQUIDITY_CANDIDATE_LABELS = {
    "liquidity_level_z": "High liquidity level",
    "illiquidity_from_volume_z": "High illiquidity from volume",
    "amihud_illiquidity_raw_z": "High Amihud illiquidity (raw)",
    "amihud_illiquidity_winsorized_z": "High Amihud illiquidity (winsorised)",
    "amihud_illiquidity_median_z": "High Amihud illiquidity (median)",
    "spread_illiquidity_z": "High Corwin-Schultz spread",
    "liquidity_shock_z": "Positive liquidity shock",
}

PLOT_RETAINED_LIQUIDITY_CANDIDATES = [
    "High illiquidity from volume",
    "High Corwin-Schultz spread",
    "High Amihud illiquidity (raw)",
    "High Amihud illiquidity (winsorised)",
    "High Amihud illiquidity (median)",
]

FACTOR_DIRECTION_CANDIDATES = {
    "Market-adjusted relative performance": "momentum_score_z",
    "Implementation friction": "liquidity_1m_z",
    "Sector-specific uncertainty": "volatility_score_z",
}


def load_common_vix_threshold() -> float:
    panel = pd.read_csv(MONTHLY_PANEL_PATH, usecols=["Date", "vix_close"], parse_dates=["Date"])
    vix = panel.drop_duplicates(subset=["Date"]).dropna(subset=["vix_close"]).sort_values("Date")
    vix = vix[vix["Date"] <= PRE_TEST_OUTCOME_END]
    return float(vix["vix_close"].quantile(0.75))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def month_end_snapshot(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    return (
        df.set_index("Date")[value_columns]
        .resample("ME")
        .last()
        .dropna(how="all")
        .reset_index()
    )


def month_end_average(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    return (
        df.set_index("Date")[value_columns]
        .resample("ME")
        .mean()
        .dropna(how="all")
        .reset_index()
    )


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if series.dropna().empty:
        return series.copy()
    lower = series.expanding(min_periods=1).quantile(lower_q)
    upper = series.expanding(min_periods=1).quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def compute_amihud_daily(close: pd.Series, volume: pd.Series) -> pd.Series:
    daily_return = close.pct_change().abs()
    dollar_volume = (close * volume).replace(0, np.nan)
    return daily_return / dollar_volume


def compute_amihud_illiquidity(close: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    amihud_daily_raw = compute_amihud_daily(close, volume)
    amihud_daily_winsorized = winsorize_series(amihud_daily_raw)

    rolling_mean_raw = amihud_daily_raw.rolling(21, min_periods=21).mean()
    rolling_mean_winsorized = amihud_daily_winsorized.rolling(21, min_periods=21).mean()
    rolling_median_raw = amihud_daily_raw.rolling(21, min_periods=21).median()

    return (
        np.log1p(rolling_mean_raw),
        np.log1p(rolling_mean_winsorized),
        np.log1p(rolling_median_raw),
    )


def build_monthly_liquidity_candidate_panel() -> pd.DataFrame:
    base_cfg = load_yaml(BASE_CONFIG)
    factor_cfg = load_yaml(FACTORS_CONFIG)
    symbols = base_cfg["universe"]["etfs"]
    liquidity_window = factor_cfg["factors"]["liquidity"]["window_days"]

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = load_and_clean_price_csv(RAW_DIR / f"{symbol}.csv")
        df["liquidity_level"] = compute_average_dollar_volume(df["Close"], df["Volume"], liquidity_window)
        df["illiquidity_from_volume"] = -df["liquidity_level"]
        (
            df["amihud_illiquidity_raw"],
            df["amihud_illiquidity_winsorized"],
            df["amihud_illiquidity_median"],
        ) = compute_amihud_illiquidity(df["Close"], df["Volume"])
        df["spread_illiquidity"] = compute_corwin_schultz_spread(df["High"], df["Low"])

        monthly_levels = month_end_snapshot(
            df,
            value_columns=[
                "Close",
                "liquidity_level",
                "illiquidity_from_volume",
                "amihud_illiquidity_raw",
                "amihud_illiquidity_winsorized",
                "amihud_illiquidity_median",
            ],
        )
        monthly_spread = month_end_average(df, value_columns=["spread_illiquidity"])
        monthly = monthly_levels.merge(monthly_spread, on="Date", how="left")
        monthly["liquidity_shock"] = monthly["liquidity_level"] - monthly["liquidity_level"].rolling(3, min_periods=3).mean()
        monthly["symbol"] = symbol
        frames.append(monthly)

    panel = pd.concat(frames, ignore_index=True).sort_values(["Date", "symbol"]).reset_index(drop=True)
    standardized = cross_sectional_zscore(
        panel,
        [
            "liquidity_level",
            "illiquidity_from_volume",
            "amihud_illiquidity_raw",
            "amihud_illiquidity_winsorized",
            "amihud_illiquidity_median",
            "spread_illiquidity",
            "liquidity_shock",
        ],
    )
    return standardized


def build_master_panel() -> pd.DataFrame:
    monthly_panel = pd.read_csv(MONTHLY_PANEL_PATH, parse_dates=["Date"]).sort_values(["symbol", "Date"]).copy()
    monthly_panel["next_month_return"] = monthly_panel.groupby("symbol")["Close"].shift(-1) / monthly_panel["Close"] - 1.0
    liquidity_panel = build_monthly_liquidity_candidate_panel()
    merged = monthly_panel.merge(
        liquidity_panel[
            [
                "Date",
                "symbol",
                "liquidity_level_z",
                "illiquidity_from_volume_z",
                "amihud_illiquidity_raw_z",
                "amihud_illiquidity_winsorized_z",
                "amihud_illiquidity_median_z",
                "spread_illiquidity_z",
                "liquidity_shock_z",
            ]
        ],
        on=["Date", "symbol"],
        how="left",
    )
    return merged


def compute_monthly_ic_frame(panel: pd.DataFrame, factor_columns: dict[str, str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for factor_name, factor_column in factor_columns.items():
        for date, group in panel.groupby("Date"):
            valid = group[[factor_column, "next_month_return", "vix_close"]].dropna()
            if len(valid) < 2 or valid[factor_column].nunique() < 2 or valid["next_month_return"].nunique() < 2:
                continue
            records.append(
                {
                    "factor": factor_name,
                    "Date": date,
                    "ic": valid[factor_column].corr(valid["next_month_return"], method="spearman"),
                    "vix_close": valid["vix_close"].iloc[0],
                }
            )
    return pd.DataFrame(records).sort_values(["factor", "Date"]).reset_index(drop=True)


def summarize_ic_by_regime(ic_frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    vix_threshold = load_common_vix_threshold()
    regime_masks = {
        "Overall": pd.Series(True, index=ic_frame.index),
        "Normal VIX": ic_frame["vix_close"] <= vix_threshold,
        "High VIX": ic_frame["vix_close"] > vix_threshold,
    }

    rows: list[dict[str, object]] = []
    for factor_name, group in ic_frame.groupby("factor"):
        overall_mean_ic = group["ic"].mean()
        for sample_name, mask in regime_masks.items():
            sample = group.loc[mask.loc[group.index], "ic"]
            if sample.empty:
                continue
            rows.append(
                {
                    "Factor": factor_name,
                    "Sample": sample_name,
                    "Number of months": int(sample.notna().sum()),
                    "Mean IC": sample.mean(),
                    "Median IC": sample.median(),
                    "IC standard deviation": sample.std(),
                    "IC 25th percentile": sample.quantile(0.25),
                    "IC 75th percentile": sample.quantile(0.75),
                    "Share of positive IC months": (sample > 0).mean(),
                    "Diagnostic status": classify_candidate(overall_mean_ic, sample_name, sample.mean()),
                }
            )
    return pd.DataFrame(rows), float(vix_threshold)


def classify_candidate(overall_mean_ic: float, sample_name: str, sample_mean_ic: float) -> str:
    if sample_name == "Overall":
        if overall_mean_ic >= 0.03:
            return "Main-model candidate"
        if overall_mean_ic > 0:
            return "Appendix / robustness candidate"
        return "Weak under current specification"

    if sample_name == "High VIX" and sample_mean_ic > 0.05:
        return "Strong regime signal"
    if sample_mean_ic > 0:
        return "Some regime support"
    return "Weak in this regime"


def compute_factor_direction_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_name, factor_column in FACTOR_DIRECTION_CANDIDATES.items():
        for direction_name, multiplier in (("As specified", 1.0), ("Reversed sign", -1.0)):
            working = panel[["Date", "next_month_return", "vix_close", factor_column]].dropna().copy()
            working["factor_value"] = multiplier * working[factor_column]
            ic_rows = []
            for date, group in working.groupby("Date"):
                if len(group) < 2:
                    continue
                if group["factor_value"].nunique() < 2 or group["next_month_return"].nunique() < 2:
                    continue
                ic_rows.append(
                    {
                        "Date": date,
                        "ic": group["factor_value"].corr(group["next_month_return"], method="spearman"),
                        "vix_close": group["vix_close"].iloc[0],
                    }
                )
            ic_frame = pd.DataFrame(ic_rows)
            if ic_frame.empty:
                continue
            rows.append(
                {
                    "Factor": factor_name,
                    "Direction": direction_name,
                    "Number of months": int(ic_frame["ic"].notna().sum()),
                    "Mean IC": ic_frame["ic"].mean(),
                    "Median IC": ic_frame["ic"].median(),
                    "IC standard deviation": ic_frame["ic"].std(),
                    "Share of positive IC months": (ic_frame["ic"] > 0).mean(),
                }
            )
    return pd.DataFrame(rows)


def format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    percentage_columns = ["Share of positive IC months"]
    for column in percentage_columns:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2%}")
    float_columns = [
        "Mean IC",
        "Median IC",
        "IC standard deviation",
        "IC 25th percentile",
        "IC 75th percentile",
    ]
    for column in float_columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    return formatted


def compute_proxy_selection_summary(panel: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = {
        "High illiquidity from volume": "illiquidity_from_volume_z",
        "High Amihud illiquidity (raw)": "amihud_illiquidity_raw_z",
        "High Amihud illiquidity (winsorised)": "amihud_illiquidity_winsorized_z",
        "High Amihud illiquidity (median)": "amihud_illiquidity_median_z",
        "High Corwin-Schultz spread": "spread_illiquidity_z",
    }
    vix_threshold = load_common_vix_threshold()
    rows: list[dict[str, object]] = []

    for factor_name, factor_column in candidate_columns.items():
        ic_values: list[float] = []
        high_vix_ic_values: list[float] = []
        spread_alignment: list[float] = []
        amihud_raw_alignment: list[float] = []
        rank_diff_vs_raw: list[float] = []

        for _, group in panel.groupby("Date"):
            columns = list(
                dict.fromkeys(
                    [
                        factor_column,
                        "next_month_return",
                        "vix_close",
                        "spread_illiquidity_z",
                        "amihud_illiquidity_raw_z",
                        "amihud_illiquidity_winsorized_z",
                    ]
                )
            )
            valid = group[columns].dropna()
            if len(valid) < 2:
                continue
            if valid[factor_column].nunique() < 2 or valid["next_month_return"].nunique() < 2:
                continue
            monthly_ic = valid[factor_column].corr(valid["next_month_return"], method="spearman")
            ic_values.append(float(monthly_ic))
            if valid["vix_close"].iloc[0] > vix_threshold:
                high_vix_ic_values.append(float(monthly_ic))

            if factor_column != "spread_illiquidity_z":
                spread_corr = (
                    valid[factor_column].corr(valid["spread_illiquidity_z"], method="spearman")
                    if valid["spread_illiquidity_z"].nunique() >= 2
                    else np.nan
                )
                if pd.notna(spread_corr):
                    spread_alignment.append(float(spread_corr))
            if factor_column != "amihud_illiquidity_raw_z":
                amihud_raw_corr = (
                    valid[factor_column].corr(valid["amihud_illiquidity_raw_z"], method="spearman")
                    if valid["amihud_illiquidity_raw_z"].nunique() >= 2
                    else np.nan
                )
                if pd.notna(amihud_raw_corr):
                    amihud_raw_alignment.append(float(amihud_raw_corr))
            if factor_column == "amihud_illiquidity_winsorized_z":
                raw_rank = valid["amihud_illiquidity_raw_z"].rank(method="average")
                win_rank = valid["amihud_illiquidity_winsorized_z"].rank(method="average")
                rank_diff_vs_raw.append(float((raw_rank - win_rank).abs().mean()))

        rows.append(
            {
                "Factor": factor_name,
                "Overall mean IC": np.mean(ic_values) if ic_values else np.nan,
                "High-VIX mean IC": np.mean(high_vix_ic_values) if high_vix_ic_values else np.nan,
                "Mean monthly rank correlation with spread proxy": np.mean(spread_alignment) if spread_alignment else np.nan,
                "Mean monthly rank correlation with raw Amihud": np.mean(amihud_raw_alignment) if amihud_raw_alignment else (1.0 if factor_column == "amihud_illiquidity_raw_z" else np.nan),
                "Mean monthly rank difference vs raw Amihud": np.mean(rank_diff_vs_raw) if rank_diff_vs_raw else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_outlier_robustness_summary() -> pd.DataFrame:
    base_cfg = load_yaml(BASE_CONFIG)
    symbols = base_cfg["universe"]["etfs"]
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        df = load_and_clean_price_csv(RAW_DIR / f"{symbol}.csv")
        df = df[df["Date"] <= PRE_TEST_OUTCOME_END].copy()
        raw_daily = compute_amihud_daily(df["Close"], df["Volume"])
        winsorized_daily = winsorize_series(raw_daily)

        raw_p50 = raw_daily.quantile(0.50)
        raw_p99 = raw_daily.quantile(0.99)
        win_p50 = winsorized_daily.quantile(0.50)
        win_p99 = winsorized_daily.quantile(0.99)

        rows.append(
            {
                "Ticker": symbol,
                "Raw p99 / p50": raw_p99 / raw_p50 if raw_p50 and pd.notna(raw_p50) else np.nan,
                "Winsorised p99 / p50": win_p99 / win_p50 if win_p50 and pd.notna(win_p50) else np.nan,
                "Raw max / p99": raw_daily.max() / raw_p99 if raw_p99 and pd.notna(raw_p99) else np.nan,
                "Winsorised max / p99": winsorized_daily.max() / win_p99 if win_p99 and pd.notna(win_p99) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def format_proxy_selection_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in [
        "Overall mean IC",
        "High-VIX mean IC",
        "Mean monthly rank correlation with spread proxy",
        "Mean monthly rank correlation with raw Amihud",
        "Mean monthly rank difference vs raw Amihud",
    ]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    return formatted


def format_outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in ["Raw p99 / p50", "Winsorised p99 / p50", "Raw max / p99", "Winsorised max / p99"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    return formatted


def build_transaction_cost_linkage_frame() -> pd.DataFrame:
    return build_transaction_cost_linkage_frame_for_source("baseline")


def load_appendix_i_liquidity_lookup() -> pd.DataFrame:
    lookup = pd.read_csv(
        APPENDIX_I_LIQUIDITY_LOOKUP_PATH,
        usecols=["Date", "symbol", "liquidity_1m", "corwin_schultz_spread"],
        parse_dates=["Date"],
    ).sort_values(["Date", "symbol"]).reset_index(drop=True)

    duplicate_mask = lookup.duplicated(subset=["Date", "symbol"])
    if duplicate_mask.any():
        duplicated_rows = lookup.loc[duplicate_mask, ["Date", "symbol"]].head().to_dict("records")
        raise ValueError(f"Appendix I liquidity lookup has duplicate Date-symbol rows: {duplicated_rows}")

    monthly_symbol_counts = lookup.groupby("Date")["symbol"].nunique()
    if not monthly_symbol_counts.eq(9).all():
        bad_counts = monthly_symbol_counts.loc[~monthly_symbol_counts.eq(9)].head().to_dict()
        raise ValueError(f"Appendix I liquidity lookup must contain 9 ETFs per month: {bad_counts}")

    if lookup[["liquidity_1m", "corwin_schultz_spread"]].isna().any().any():
        missing_counts = lookup[["liquidity_1m", "corwin_schultz_spread"]].isna().sum().to_dict()
        raise ValueError(f"Appendix I liquidity lookup contains missing values: {missing_counts}")

    return lookup


def transaction_cost_linkage_model_map(source: str = "baseline") -> dict[str, tuple[str, str]]:
    source = source.lower()
    if source not in {"baseline", "main_result"}:
        raise ValueError("source must be 'baseline' or 'main_result'")

    if source == "baseline":
        return {
            "S1": ("baseline_fixed_weight_selections.csv", "baseline_fixed_weight_portfolio_returns.csv"),
            "A1": ("baseline_rolling_ic_selections.csv", "baseline_rolling_ic_portfolio_returns.csv"),
            "L1": ("baseline_ridge_ic_selections.csv", "baseline_ridge_ic_portfolio_returns.csv"),
            "L2": ("baseline_lasso_ic_selections.csv", "baseline_lasso_ic_portfolio_returns.csv"),
            "L3": ("baseline_elastic_net_ic_selections.csv", "baseline_elastic_net_ic_portfolio_returns.csv"),
            "T1": ("baseline_random_forest_ic_selections.csv", "baseline_random_forest_ic_portfolio_returns.csv"),
            "T2": ("baseline_xgboost_ic_selections.csv", "baseline_xgboost_ic_portfolio_returns.csv"),
        }

    return {
        "S1": (
            f"{MAIN_RESULT_PREFIX}_fixed_weight_selections.csv",
            f"{MAIN_RESULT_PREFIX}_fixed_weight_portfolio_returns.csv",
        ),
        "A1": (
            f"{MAIN_RESULT_PREFIX}_rolling_ic_selections.csv",
            f"{MAIN_RESULT_PREFIX}_rolling_ic_portfolio_returns.csv",
        ),
        "L1": (
            f"{MAIN_RESULT_PREFIX}_ridge_ic_selections.csv",
            f"{MAIN_RESULT_PREFIX}_ridge_ic_portfolio_returns.csv",
        ),
        "L2": (
            f"{MAIN_RESULT_PREFIX}_lasso_ic_selections.csv",
            f"{MAIN_RESULT_PREFIX}_lasso_ic_portfolio_returns.csv",
        ),
        "L3": (
            f"{MAIN_RESULT_PREFIX}_elastic_net_ic_selections.csv",
            f"{MAIN_RESULT_PREFIX}_elastic_net_ic_portfolio_returns.csv",
        ),
        "T1": (
            f"{MAIN_RESULT_PREFIX}_random_forest_ic_selections.csv",
            f"{MAIN_RESULT_PREFIX}_random_forest_ic_portfolio_returns.csv",
        ),
        "T2": (
            f"{MAIN_RESULT_PREFIX}_xgboost_ic_selections.csv",
            f"{MAIN_RESULT_PREFIX}_xgboost_ic_portfolio_returns.csv",
        ),
    }


def build_transaction_cost_linkage_frame_for_source(source: str = "baseline") -> pd.DataFrame:
    model_map = transaction_cost_linkage_model_map(source)
    execution_setting = "F1" if source.lower() == "baseline" else "F3"

    frames: list[pd.DataFrame] = []
    for model_code, (selection_name, returns_name) in model_map.items():
        selections = pd.read_csv(BACKTEST_DIR / selection_name, parse_dates=["Date"])
        portfolio_returns = pd.read_csv(BACKTEST_DIR / returns_name, parse_dates=["Date"])
        if selections[["liquidity_1m", "corwin_schultz_spread"]].isna().any().any():
            missing_counts = selections[["liquidity_1m", "corwin_schultz_spread"]].isna().sum().to_dict()
            raise ValueError(f"{selection_name} contains missing Appendix I liquidity inputs: {missing_counts}")
        monthly = (
            selections.groupby("Date")
            .agg(
                amihud=("liquidity_1m", "mean"),
                spread=("corwin_schultz_spread", "mean"),
                n_selected=("symbol", "count"),
            )
            .reset_index()
        )
        merged = monthly.merge(
            portfolio_returns[["Date", "turnover", "transaction_cost_rate", "net_portfolio_return", "portfolio_return"]],
            on="Date",
            how="inner",
        )
        merged["effective_cost_bps"] = np.where(
            merged["turnover"] > 0,
            merged["transaction_cost_rate"] / merged["turnover"] * 10000.0,
            np.nan,
        )
        merged["log10_amihud"] = np.log10(merged["amihud"].clip(lower=1e-16))
        merged["execution_setting"] = execution_setting
        merged["model"] = model_code
        frames.append(merged)
    return pd.concat(frames, ignore_index=True).sort_values(["model", "Date"]).reset_index(drop=True)


def build_traded_leg_linkage_frame_for_source(source: str = "main_result") -> pd.DataFrame:
    model_map = transaction_cost_linkage_model_map(source)
    liquidity_lookup = load_appendix_i_liquidity_lookup()
    execution_setting = "F3" if source.lower() == "main_result" else "F1"

    frames: list[pd.DataFrame] = []
    for model_code, (selection_name, returns_name) in model_map.items():
        selections = pd.read_csv(BACKTEST_DIR / selection_name, parse_dates=["Date"])
        portfolio_returns = pd.read_csv(BACKTEST_DIR / returns_name, parse_dates=["Date"])

        weights = (
            selections.pivot(index="Date", columns="symbol", values="portfolio_weight")
            .fillna(0.0)
            .sort_index()
        )
        traded_weights = weights.diff().abs()
        traded_weights.iloc[0] = weights.iloc[0].abs()
        traded_long = (
            traded_weights.stack()
            .rename("traded_weight")
            .reset_index()
            .query("traded_weight > 0")
        )

        traded = traded_long.merge(
            liquidity_lookup,
            on=["Date", "symbol"],
            how="left",
            validate="many_to_one",
        )
        if traded[["liquidity_1m", "corwin_schultz_spread"]].isna().any().any():
            missing_rows = traded.loc[
                traded[["liquidity_1m", "corwin_schultz_spread"]].isna().any(axis=1),
                ["Date", "symbol", "traded_weight"],
            ].head()
            raise ValueError(
                f"Appendix I traded-leg merge produced missing liquidity inputs for {selection_name}: "
                f"{missing_rows.to_dict('records')}"
            )

        monthly_rows: list[dict[str, object]] = []
        for date, group in traded.groupby("Date", sort=True):
            monthly_rows.append(
                {
                    "Date": date,
                    "amihud": float(group["liquidity_1m"].mean()),
                    "spread": float(group["corwin_schultz_spread"].mean()),
                    "n_traded_symbols": int(group["symbol"].nunique()),
                    "total_traded_weight": float(group["traded_weight"].sum()),
                }
            )
        traded_monthly = pd.DataFrame(monthly_rows)

        merged = traded_monthly.merge(
            portfolio_returns[["Date", "turnover", "transaction_cost_rate", "net_portfolio_return", "portfolio_return"]],
            on="Date",
            how="inner",
        )
        merged["effective_cost_bps"] = np.where(
            merged["turnover"] > 0,
            merged["transaction_cost_rate"] / merged["turnover"] * 10000.0,
            np.nan,
        )
        merged["log10_amihud"] = np.log10(merged["amihud"].clip(lower=1e-16))
        merged["execution_setting"] = execution_setting
        merged["model"] = model_code
        frames.append(merged)

    return pd.concat(frames, ignore_index=True).sort_values(["model", "Date"]).reset_index(drop=True)


def make_liquidity_candidate_boxplot(ic_frame: pd.DataFrame) -> None:
    plot_df = ic_frame[ic_frame["factor"].isin(PLOT_RETAINED_LIQUIDITY_CANDIDATES)].copy()
    labels = PLOT_RETAINED_LIQUIDITY_CANDIDATES
    series_list = [plot_df.loc[plot_df["factor"] == label, "ic"].dropna().values for label in labels]

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    box = ax.boxplot(series_list, patch_artist=True, tick_labels=labels, showfliers=False)
    palette = ["#4c78a8", "#6d597a", "#e45756", "#72b7b2", "#f58518"]
    for patch, color in zip(box["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in box["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.5)

    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.axvline(2.5, color="#adb5bd", linestyle=":", linewidth=1.2)
    ax.text(1.5, 0.98, "Cross-family screen", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, color="#495057")
    ax.text(4.0, 0.98, "Amihud family", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10, color="#495057")
    ax.set_title("Liquidity Candidate Families: IC Distributions")
    ax.set_ylabel("Monthly factor IC")
    ax.set_xlabel("Liquidity candidate")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_l1_liquidity_candidate_ic_boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_proxy_selection_bar_chart(proxy_selection_df: pd.DataFrame) -> None:
    filtered = proxy_selection_df[proxy_selection_df["Factor"].isin(PLOT_RETAINED_LIQUIDITY_CANDIDATES)].copy()
    plot_df = filtered[
        ["Factor", "Overall mean IC", "High-VIX mean IC"]
    ].melt(id_vars="Factor", var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    sns_colors = {
        "Overall mean IC": "#4c78a8",
        "High-VIX mean IC": "#d62828",
    }
    x = np.arange(filtered["Factor"].nunique())
    width = 0.35
    ordered_factors = PLOT_RETAINED_LIQUIDITY_CANDIDATES
    for idx, metric in enumerate(["Overall mean IC", "High-VIX mean IC"]):
        sample = plot_df[plot_df["metric"] == metric].set_index("Factor").reindex(ordered_factors)
        bars = ax.bar(
            x + (idx - 0.5) * width,
            sample["value"],
            width=width,
            label=metric,
            color=sns_colors[metric],
        )
        finite_values = [abs(float(v)) for v in sample["value"].dropna()]
        offset = max(max(finite_values) * 0.012, 0.002) if finite_values else 0.002
        for bar, value in zip(bars, sample["value"]):
            if pd.isna(value):
                continue
            x_pos = bar.get_x() + bar.get_width() / 2
            if value >= 0:
                y_pos = float(value) + offset
                va = "bottom"
            else:
                y_pos = float(value) - offset
                va = "top"
            ax.text(x_pos, y_pos, f"{float(value):.3f}", ha="center", va=va, fontsize=9)
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_factors, rotation=10)
    ax.set_ylabel("Mean monthly factor IC")
    ax.set_xlabel("Liquidity proxy candidate")
    ax.axvline(1.5, color="#adb5bd", linestyle=":", linewidth=1.2)
    ax.text(0.14, 0.98, "Cross-family screen", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#495057")
    ax.text(0.77, 0.98, "Amihud family selection", transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#495057")
    ax.set_title("Liquidity Proxy Selection: Family Screen and Amihud Choice")
    ax.legend(frameon=True, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    ax.grid(axis="y", alpha=0.25)
    upper = float(filtered[["Overall mean IC", "High-VIX mean IC"]].to_numpy().max())
    lower = float(filtered[["Overall mean IC", "High-VIX mean IC"]].to_numpy().min())
    ax.set_ylim(lower - 0.02, upper + 0.025)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    fig.savefig(FIGURES_DIR / "figure_l3_liquidity_proxy_selection_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_outlier_robustness_bar_chart(outlier_df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        [
            {
                "metric": "Average p99 / p50",
                "Raw": outlier_df["Raw p99 / p50"].mean(),
                "Winsorised": outlier_df["Winsorised p99 / p50"].mean(),
            },
            {
                "metric": "Average max / p99",
                "Raw": outlier_df["Raw max / p99"].mean(),
                "Winsorised": outlier_df["Winsorised max / p99"].mean(),
            },
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), constrained_layout=True)
    palette = {"Raw": "#6d597a", "Winsorised": "#2a9d8f"}
    for ax, metric in zip(axes, summary["metric"]):
        sample = summary.loc[summary["metric"] == metric].iloc[0]
        values = [sample["Raw"], sample["Winsorised"]]
        bars = ax.bar(["Raw", "Winsorised"], values, color=[palette["Raw"], palette["Winsorised"]], width=0.55)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    axes[0].set_ylabel("Average ratio across sector ETFs")
    fig.suptitle("Raw vs Winsorised Amihud: Outlier Sensitivity")
    fig.savefig(FIGURES_DIR / "figure_l4_amihud_outlier_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_master_panel()
    panel = panel[panel["Date"] <= PRE_TEST_SIGNAL_END].copy()

    liquidity_factor_columns = {
        label: column
        for column, label in LIQUIDITY_CANDIDATE_LABELS.items()
    }
    liquidity_ic_frame = compute_monthly_ic_frame(panel, liquidity_factor_columns)
    liquidity_summary, vix_threshold = summarize_ic_by_regime(liquidity_ic_frame)
    factor_direction_summary = compute_factor_direction_summary(panel)
    proxy_selection_summary = compute_proxy_selection_summary(panel)
    outlier_robustness_summary = compute_outlier_robustness_summary()
    format_summary_table(liquidity_summary).to_csv(
        TABLES_DIR / "table_l1_liquidity_candidate_summary.csv",
        index=False,
    )
    format_summary_table(factor_direction_summary).to_csv(
        TABLES_DIR / "table_l2_factor_direction_diagnostic.csv",
        index=False,
    )
    format_proxy_selection_table(proxy_selection_summary).to_csv(
        TABLES_DIR / "table_l3_liquidity_proxy_selection_summary.csv",
        index=False,
    )
    format_outlier_table(outlier_robustness_summary).to_csv(
        TABLES_DIR / "table_l4_amihud_outlier_robustness.csv",
        index=False,
    )
    make_liquidity_candidate_boxplot(liquidity_ic_frame)
    make_proxy_selection_bar_chart(proxy_selection_summary)
    make_outlier_robustness_bar_chart(outlier_robustness_summary)

    print("Saved outputs/tables/table_l1_liquidity_candidate_summary.csv")
    print("Saved outputs/tables/table_l2_factor_direction_diagnostic.csv")
    print("Saved outputs/tables/table_l3_liquidity_proxy_selection_summary.csv")
    print("Saved outputs/tables/table_l4_amihud_outlier_robustness.csv")
    print("Saved outputs/figures/figure_l1_liquidity_candidate_ic_boxplot.png")
    print("Saved outputs/figures/figure_l3_liquidity_proxy_selection_bar.png")
    print("Saved outputs/figures/figure_l4_amihud_outlier_robustness.png")
    print(f"VIX high-regime threshold (75th percentile): {vix_threshold:.3f}")


if __name__ == "__main__":
    main()
