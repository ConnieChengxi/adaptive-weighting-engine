from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "logs" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from adaptive_weighting.data.preprocess import load_and_clean_price_csv
from adaptive_weighting.backtest.evaluation import summarize_performance
from adaptive_weighting.ic.compute_ic import compute_monthly_factor_ic

BACKTEST_DIR = ROOT / "outputs" / "backtests"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
RAW_DIR = ROOT / "data" / "raw"

MODEL_LABELS = {
    "equal_weight_benchmark": "B0: Naive benchmark",
    "fixed_weight": "S1: Static equal-dimension model",
    "rolling_ic": "A1: Rolling IC adaptive model",
    "ridge_ic": "L1: Ridge IC model",
    "lasso_ic": "L2: Lasso IC model",
    "elastic_net_ic": "L3: Elastic Net IC model",
    "random_forest_ic": "T1: Random Forest IC model",
    "xgboost_ic": "T2: XGBoost IC model",
    "rolling_ic_no_shrinkage": "A1 Robustness: No shrinkage",
    "rolling_ic_60_40": "A1 Robustness: 60/40 shrinkage",
}

MODEL_CODE_LABELS = {
    "equal_weight_benchmark": "B0",
    "fixed_weight": "S1",
    "rolling_ic": "A1",
    "ridge_ic": "L1",
    "lasso_ic": "L2",
    "elastic_net_ic": "L3",
    "random_forest_ic": "T1",
    "xgboost_ic": "T2",
}

MODEL_SHORT_LABELS = {
    "equal_weight_benchmark": "B0",
    "fixed_weight": "S1",
    "rolling_ic": "A1",
    "ridge_ic": "L1",
    "lasso_ic": "L2",
    "elastic_net_ic": "L3",
    "random_forest_ic": "T1",
    "xgboost_ic": "T2",
}

FRAMEWORK_LABELS = {
    "baseline": "F1: No control",
    "pta": "F2: PTA",
    "holding_buffer_top4": "F3: Holding-buffer rule",
    "holding_buffer_top5": "F3: Holding-buffer rule",
    "holding_buffer_top6": "F3: Holding-buffer rule",
}

FRAMEWORK_SHORT_LABELS = {
    "baseline": "F1: No control",
    "pta": "F2: PTA",
    "holding_buffer_top6": "F3: Holding-buffer rule",
}

MAIN_HOLDING_BUFFER_FRAMEWORK = "holding_buffer_top6"
MAIN_RESULT_PREFIX = f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}"
MAIN_FRAMEWORK_COMPARISON_MODELS = [
    "fixed_weight",
    "rolling_ic",
    "ridge_ic",
    "lasso_ic",
    "elastic_net_ic",
    "random_forest_ic",
    "xgboost_ic",
]

HOLDING_BUFFER_RANK_LABELS = {
    4: "Top 4",
    5: "Top 5",
    6: "Top 6",
}

MODEL_COLORS = {
    "equal_weight_benchmark": "#6d597a",
    "fixed_weight": "#264653",
    "rolling_ic": "#e76f51",
    "ridge_ic": "#8d99ae",
    "lasso_ic": "#bc6c25",
    "elastic_net_ic": "#577590",
    "random_forest_ic": "#43aa8b",
    "xgboost_ic": "#2a9d8f",
    "rolling_ic_no_shrinkage": "#457b9d",
    "rolling_ic_60_40": "#e9c46a",
}

FACTOR_LABELS = {
    "momentum_score_z_weight": "Market-adjusted relative performance",
    "liquidity_1m_z_weight": "Implementation friction",
    "volatility_score_z_weight": "Sector-specific uncertainty",
}

FACTOR_IC_LABELS = {
    "momentum_score_z_ic": "Market-adjusted relative performance",
    "liquidity_1m_z_ic": "Implementation friction",
    "volatility_score_z_ic": "Sector-specific uncertainty",
}
CORE3_FACTOR_COLUMNS = [
    "momentum_score_z",
    "liquidity_1m_z",
    "volatility_score_z",
]

PERCENT_COLUMNS = [
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "turnover",
    "net_return_after_costs",
    "average_transaction_cost_rate",
    "gross_return",
    "net_return",
]
ETF_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLU", "XLP", "XLB"]

DISPLAY_COLUMNS = {
    "research_question": "Research question",
    "focus": "Focus",
    "primary_tables": "Primary tables",
    "primary_figures": "Primary figures",
    "recommended_location": "Recommended location",
    "model_name": "Model code",
    "weighting_logic": "Weighting logic",
    "adaptive_or_not": "Adaptive",
    "ml_or_not": "ML-based",
    "shrinkage_rule": "Shrinkage rule",
    "model": "Model",
    "annualized_return": "Annualised return",
    "annualized_volatility": "Annualised volatility",
    "sharpe_ratio": "Sharpe ratio",
    "max_drawdown": "Maximum drawdown",
    "calmar_ratio": "Calmar ratio",
    "turnover": "Turnover",
    "net_return_after_costs": "Net return after costs",
    "average_transaction_cost_rate": "Average transaction cost rate",
    "framework": "Framework",
    "regime": "VIX regime",
    "months": "Number of months",
    "holding_buffer_rank": "Threshold",
    "ticker": "Ticker",
    "sample_start_date": "Sample start date",
    "sample_end_date": "Sample end date",
    "number_of_monthly_observations": "Number of monthly observations",
    "mean_monthly_return": "Mean monthly return",
    "maximum_drawdown": "Maximum drawdown",
    "average_monthly_dollar_volume": "Average monthly dollar volume",
    "mean_maximum_factor_weight": "Mean maximum dimension weight",
    "share_of_months_max_weight_gt_0_50": "Share of months with maximum weight > 0.50",
    "share_of_months_max_weight_gt_0_60": "Share of months with maximum weight > 0.60",
    "model_code": "Model code",
    "factor": "Dimension",
    "sample": "Sample",
    "n_months": "Number of months",
    "mean_ic": "Mean IC",
    "median_ic": "Median IC",
    "std_ic": "IC standard deviation",
    "p25": "IC 25th percentile",
    "p75": "IC 75th percentile",
    "share_positive": "Share of positive IC months",
    "monthly_volatility": "Monthly volatility",
    "return_skewness": "Return skewness",
    "excess_kurtosis": "Excess kurtosis",
    "return_p05": "5% monthly return quantile",
    "expected_shortfall_5": "Expected shortfall (5%)",
    "metric": "Metric",
    "mean": "Mean",
    "median": "Median",
    "p10": "10th percentile",
    "p90": "90th percentile",
    "min": "Minimum",
    "max": "Maximum",
    "avg_pairwise_corr": "Average pairwise correlation",
    "beta_to_spy": "Beta to SPY",
    "r_squared_to_spy": "R-squared to SPY",
    "residual_volatility": "Residual volatility",
    "first_pc_variance_share": "First principal component variance share",
    "top_1_persistence": "Top-1 persistence",
    "top_3_retention_rate": "Top-3 retention rate",
    "bottom_3_retention_rate": "Bottom-3 retention rate",
    "top_3_membership_turnover": "Top-3 membership turnover",
    "rank_autocorrelation": "Rank autocorrelation",
    "from_bucket": "From tercile",
    "to_bucket": "To tercile",
    "transition_probability": "Transition probability",
    "median_monthly_dollar_volume": "Median monthly dollar volume",
    "p10_monthly_dollar_volume": "10% monthly dollar volume",
    "p90_monthly_dollar_volume": "90% monthly dollar volume",
    "mean_amihud_illiquidity": "Mean log10 Amihud illiquidity",
    "median_amihud_illiquidity": "Median log10 Amihud illiquidity",
    "p90_amihud_illiquidity": "90% log10 Amihud illiquidity",
    "mean_spread": "Mean Corwin-Schultz spread",
    "cross_sectional_std_dollar_volume": "Cross-sectional std of dollar volume",
    "cross_sectional_std_amihud": "Cross-sectional std of Amihud illiquidity",
    "cross_sectional_std_spread": "Cross-sectional std of Corwin-Schultz spread",
    "average_amihud_illiquidity": "Average Amihud illiquidity",
    "average_corwin_schultz_spread": "Average Corwin-Schultz spread",
    "avg_log_dollar_volume": "Average log dollar volume",
    "corr_with_cross_sectional_std": "Correlation with cross-sectional std",
    "corr_with_top_minus_bottom_spread": "Correlation with top-minus-bottom spread",
    "corr_with_avg_pairwise_corr": "Correlation with average pairwise correlation",
    "corr_with_first_pc_share": "Correlation with first principal component share",
    "gross_return": "Gross annualised return",
    "net_return": "Net annualised return",
    "net_sharpe": "Net Sharpe ratio",
}


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10


def load_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(BACKTEST_DIR / name, parse_dates=parse_dates)


def main_result_backtest_name(model_name: str, artifact: str) -> str:
    return f"{MAIN_RESULT_PREFIX}_{model_name}_{artifact}.csv"


def load_table_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name, parse_dates=parse_dates)


def get_common_shrinkage_test_window() -> tuple[pd.Timestamp, pd.Timestamp]:
    summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv").iloc[0]
    test_period = str(summary["test_period"])
    test_start_str, test_end_str = [part.strip() for part in test_period.split(" to ")]
    return pd.Timestamp(test_start_str), pd.Timestamp(test_end_str)


def format_percentage(series: pd.Series, decimals: int = 2) -> pd.Series:
    return (series * 100.0).map(lambda value: f"{value:.{decimals}f}%")


def format_metric_tables(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in PERCENT_COLUMNS:
        if column in formatted.columns:
            formatted[column] = format_percentage(formatted[column])
    if "sharpe_ratio" in formatted.columns:
        formatted["sharpe_ratio"] = formatted["sharpe_ratio"].map(lambda value: f"{value:.2f}")
    if "calmar_ratio" in formatted.columns:
        formatted["calmar_ratio"] = formatted["calmar_ratio"].map(lambda value: f"{value:.2f}")
    if "net_sharpe" in formatted.columns:
        formatted["net_sharpe"] = formatted["net_sharpe"].map(lambda value: f"{value:.2f}")
    return formatted


def rename_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: DISPLAY_COLUMNS.get(column, column) for column in df.columns})


def format_publication_table_file(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    formatted = df.copy()
    if path.name == "table_d1_etf_summary_statistics.csv":
        percentage_columns = [
            "Mean monthly return",
            "Annualised return",
            "Annualised volatility",
            "Maximum drawdown",
        ]
        for column in percentage_columns:
            if column in formatted.columns:
                formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value * 100.0:.2f}%")
        if "Average monthly dollar volume" in formatted.columns:
            formatted["Average monthly dollar volume"] = formatted["Average monthly dollar volume"].map(
                lambda value: "" if pd.isna(value) else f"{value:,.2f}"
            )
        formatted.to_csv(path, index=False)
        return
    for column in formatted.columns:
        series = formatted[column]
        if pd.api.types.is_integer_dtype(series):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            continue
        lower = column.lower()
        if "effective number" in lower:
            formatted[column] = series.map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        elif "month" in lower or "number of" in lower or lower.endswith("months"):
            formatted[column] = series.round(0).astype("Int64")
        elif any(token in lower for token in ["share", "return", "volatility", "drawdown", "turnover", "rate", "spread", "probability"]):
            formatted[column] = series.map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        elif any(token in lower for token in ["ic", "corr", "correlation", "beta", "r-squared", "variance", "alpha"]):
            formatted[column] = series.map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        elif "volume" in lower:
            formatted[column] = series.map(lambda value: "" if pd.isna(value) else f"{value:,.2f}")
        else:
            formatted[column] = series.map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    formatted.to_csv(path, index=False)


def apply_publication_table_formatting() -> None:
    table_names = [
        "table_d1_etf_summary_statistics.csv",
        "table_d2_model_family_entry_timing.csv",
        "table_dd1_sector_panel_characteristics.csv",
        "table_dd2_cross_sectional_opportunity_set.csv",
        "table_dd3_common_component_market_dependence.csv",
        "table_dd4_regime_dependence.csv",
        "table_dd4_regime_mean_difference_summary.csv",
        "table_dd5_persistence_implementation_frictions.csv",
        "table_dd6_transition_matrix.csv",
        "table_ld1_sector_liquidity_distribution.csv",
        "table_ld2_liquidity_by_vix_regime.csv",
        "table_ld3_liquidity_opportunity_commonality_comovement.csv",
        "table_m1_factor_ic_by_vix_regime.csv",
        "table_l1_liquidity_candidate_summary.csv",
        "table_l3_liquidity_proxy_selection_summary.csv",
        "table_l4_amihud_outlier_robustness.csv",
        "table_mr1_momentum_candidate_summary.csv",
        "table_mr3_market_adjusted_relative_performance_summary.csv",
        "table_l2_factor_direction_diagnostic.csv",
        "table_vr1_volatility_candidate_summary.csv",
        "table_vr2_volatility_dimension_justification.csv",
        "table_vr3_uncertainty_ranking_disagreement.csv",
        "table_fc1_score_contribution_summary.csv",
        "table_fc2_weight_contribution_summary.csv",
        "table_fc3_hhi_concentration_summary.csv",
        "table_wf1_repeated_walkforward_fold_selection.csv",
        "table_wf2_repeated_walkforward_test_results.csv",
        "table_wf3_repeated_walkforward_family_comparison.csv",
        "table_sh1_common_shrinkage_validation_grid.csv",
        "table_sh2_common_shrinkage_selection_summary.csv",
        "table_sh3_common_shrinkage_test_comparison.csv",
        "table_sh4_common_shrinkage_selection_conclusion.csv",
        "table_sh5_model_performance_by_vix_regime.csv",
    ]
    for name in table_names:
        format_publication_table_file(TABLES_DIR / name)


def build_cumulative_wealth(df: pd.DataFrame) -> pd.DataFrame:
    wealth = df.copy()
    wealth["wealth"] = (1.0 + wealth["portfolio_return"].fillna(0.0)).cumprod()
    return wealth


def build_net_cumulative_wealth(df: pd.DataFrame) -> pd.DataFrame:
    wealth = df.copy()
    wealth["wealth"] = (1.0 + wealth["net_portfolio_return"].fillna(0.0)).cumprod()
    return wealth


def compute_drawdown_from_returns(return_series: pd.Series) -> float:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def load_monthly_return_panel() -> pd.DataFrame:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    returns = (
        monthly_panel[["Date", "symbol", "Close"]]
        .sort_values(["symbol", "Date"])
        .assign(monthly_return=lambda df: df.groupby("symbol")["Close"].pct_change())
    )
    return returns


def build_monthly_return_matrix() -> pd.DataFrame:
    returns = load_monthly_return_panel()
    matrix = returns.pivot(index="Date", columns="symbol", values="monthly_return")
    return matrix[ETF_TICKERS].copy()


def summarize_portfolio_returns_file(filename: str, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    portfolio = load_csv(filename, parse_dates=["Date"])
    portfolio = portfolio[(portfolio["Date"] >= start) & (portfolio["Date"] <= end)].copy()
    if portfolio.empty:
        raise ValueError(f"No rows found in {filename} for {start.date()} to {end.date()}.")
    return summarize_performance(
        portfolio["portfolio_return"],
        portfolio["turnover"],
        transaction_cost_rate_series=portfolio["transaction_cost_rate"],
    )


def build_framework_comparison_test_window_frame() -> pd.DataFrame:
    test_start, test_end = get_common_shrinkage_test_window()
    ordered_models = [
        "equal_weight_benchmark",
        "fixed_weight",
        "rolling_ic",
        "ridge_ic",
        "lasso_ic",
        "elastic_net_ic",
        "random_forest_ic",
        "xgboost_ic",
    ]
    ordered_frameworks = ["baseline", "pta", MAIN_HOLDING_BUFFER_FRAMEWORK]
    rows: list[dict[str, object]] = []

    for framework in ordered_frameworks:
        for model in ordered_models:
            filename = f"{framework}_{model}_portfolio_returns.csv"
            metrics = summarize_portfolio_returns_file(filename, test_start, test_end)
            metrics["framework"] = framework
            metrics["model"] = model
            rows.append(metrics)

    table = pd.DataFrame(rows)
    table["framework"] = pd.Categorical(table["framework"], categories=ordered_frameworks, ordered=True)
    table["model"] = pd.Categorical(table["model"], categories=ordered_models, ordered=True)
    return table.sort_values(["model", "framework"]).reset_index(drop=True)


def build_holding_buffer_sensitivity_test_window_frame() -> pd.DataFrame:
    test_start, test_end = get_common_shrinkage_test_window()
    ordered_models = [
        "equal_weight_benchmark",
        "fixed_weight",
        "rolling_ic",
        "ridge_ic",
        "lasso_ic",
        "elastic_net_ic",
        "random_forest_ic",
        "xgboost_ic",
    ]
    ordered_frameworks = ["holding_buffer_top4", "holding_buffer_top5", "holding_buffer_top6"]
    rows: list[dict[str, object]] = []

    for framework in ordered_frameworks:
        holding_buffer_rank = int(framework.removeprefix("holding_buffer_top"))
        for model in ordered_models:
            filename = f"{framework}_{model}_portfolio_returns.csv"
            metrics = summarize_portfolio_returns_file(filename, test_start, test_end)
            metrics["framework"] = framework
            metrics["holding_buffer_rank"] = holding_buffer_rank
            metrics["model"] = model
            rows.append(metrics)

    table = pd.DataFrame(rows)
    table["framework"] = pd.Categorical(table["framework"], categories=ordered_frameworks, ordered=True)
    table["model"] = pd.Categorical(table["model"], categories=ordered_models, ordered=True)
    return table.sort_values(["model", "holding_buffer_rank"]).reset_index(drop=True)


def safe_spearman_corr(left: pd.Series, right: pd.Series) -> float | None:
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 2:
        return None
    value = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
    return None if pd.isna(value) else float(value)


def compute_beta_and_r_squared(asset_returns: pd.Series, market_returns: pd.Series) -> tuple[float | None, float | None, float | None]:
    paired = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(paired) < 3:
        return None, None, None

    asset = paired.iloc[:, 0]
    market = paired.iloc[:, 1]
    market_var = float(market.var(ddof=1))
    if market_var == 0.0 or np.isnan(market_var):
        return None, None, None

    beta = float(asset.cov(market) / market_var)
    alpha = float(asset.mean() - beta * market.mean())
    fitted = alpha + beta * market
    residual = asset - fitted
    sse = float((residual**2).sum())
    sst = float(((asset - asset.mean()) ** 2).sum())
    r_squared = None if sst == 0.0 else float(1.0 - sse / sst)
    residual_vol = float(residual.std(ddof=1)) if len(residual) > 1 else None
    return beta, r_squared, residual_vol


def compute_first_pc_share(frame: pd.DataFrame) -> float | None:
    clean = frame.dropna(how="any")
    if len(clean) < 3 or clean.shape[1] < 2:
        return None
    centered = clean - clean.mean(axis=0)
    covariance = np.cov(centered.to_numpy(), rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) == 0:
        return None
    return float(eigenvalues.max() / eigenvalues.sum())


def build_cross_sectional_opportunity_frame() -> pd.DataFrame:
    returns_matrix = build_monthly_return_matrix()
    rows: list[dict[str, object]] = []
    for date, row in returns_matrix.iterrows():
        sample = row.dropna()
        if len(sample) < 2:
            continue
        rows.append(
            {
                "Date": date,
                "cross_sectional_std": float(sample.std(ddof=1)),
                "cross_sectional_iqr": float(sample.quantile(0.75) - sample.quantile(0.25)),
                "top_minus_bottom_spread": float(sample.max() - sample.min()),
            }
        )
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def build_leadership_transition_frame() -> pd.DataFrame:
    returns_matrix = build_monthly_return_matrix().dropna(how="any").copy()
    if returns_matrix.empty:
        return pd.DataFrame()

    ranks = returns_matrix.rank(axis=1, method="first", ascending=False)
    bucket_map = pd.DataFrame(index=ranks.index, columns=ranks.columns, dtype="object")
    for ticker in ranks.columns:
        series = ranks[ticker]
        bucket_map[ticker] = np.select(
            [series <= 3, series >= 7],
            ["Top tercile", "Bottom tercile"],
            default="Middle tercile",
        )

    transitions: list[dict[str, object]] = []
    for idx in range(1, len(bucket_map)):
        prev_row = bucket_map.iloc[idx - 1]
        curr_row = bucket_map.iloc[idx]
        for ticker in bucket_map.columns:
            transitions.append(
                {
                    "Date": bucket_map.index[idx],
                    "symbol": ticker,
                    "from_bucket": prev_row[ticker],
                    "to_bucket": curr_row[ticker],
                }
            )
    return pd.DataFrame(transitions)


def build_persistence_frame() -> pd.DataFrame:
    returns_matrix = build_monthly_return_matrix().dropna(how="any").copy()
    if len(returns_matrix) < 2:
        return pd.DataFrame()

    ranks = returns_matrix.rank(axis=1, method="average", ascending=False)
    records: list[dict[str, object]] = []
    for idx in range(1, len(returns_matrix)):
        prev_date = returns_matrix.index[idx - 1]
        curr_date = returns_matrix.index[idx]
        prev_returns = returns_matrix.loc[prev_date]
        curr_returns = returns_matrix.loc[curr_date]
        prev_ranks = ranks.loc[prev_date]
        curr_ranks = ranks.loc[curr_date]

        prev_top3 = set(prev_returns.nlargest(3).index)
        curr_top3 = set(curr_returns.nlargest(3).index)
        prev_bottom3 = set(prev_returns.nsmallest(3).index)
        curr_bottom3 = set(curr_returns.nsmallest(3).index)
        prev_top1 = prev_returns.idxmax()

        top1_persistence = 1.0 if prev_top1 == curr_returns.idxmax() else 0.0
        top3_retention = len(prev_top3 & curr_top3) / 3.0
        bottom3_retention = len(prev_bottom3 & curr_bottom3) / 3.0
        top3_turnover = 1.0 - top3_retention
        rank_autocorr = safe_spearman_corr(prev_ranks, curr_ranks)

        records.append(
            {
                "Date": curr_date,
                "top_1_persistence": top1_persistence,
                "top_3_retention_rate": top3_retention,
                "bottom_3_retention_rate": bottom3_retention,
                "top_3_membership_turnover": top3_turnover,
                "rank_autocorrelation": rank_autocorr,
            }
        )

    return pd.DataFrame(records)


def build_market_dependence_table() -> pd.DataFrame:
    returns = load_monthly_return_panel()
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    spy = monthly_panel[["Date", "spy_return_1m"]].drop_duplicates(subset=["Date"]).sort_values("Date")
    returns = returns.merge(spy, on="Date", how="left")
    returns_matrix = returns.pivot(index="Date", columns="symbol", values="monthly_return")[ETF_TICKERS]
    common = returns_matrix.dropna(how="any")
    corr = common.corr()
    avg_pairwise_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()) if not common.empty else None
    first_pc_share = compute_first_pc_share(common)

    rows: list[dict[str, object]] = []
    for ticker in ETF_TICKERS:
        asset = returns.loc[returns["symbol"] == ticker, ["Date", "monthly_return", "spy_return_1m"]].sort_values("Date")
        beta, r_squared, residual_vol = compute_beta_and_r_squared(asset["monthly_return"], asset["spy_return_1m"])
        rows.append(
            {
                "ticker": ticker,
                "avg_pairwise_corr": avg_pairwise_corr,
                "beta_to_spy": beta,
                "r_squared_to_spy": r_squared,
                "residual_volatility": residual_vol,
                "first_pc_variance_share": first_pc_share,
            }
        )
    return pd.DataFrame(rows)


def build_monthly_liquidity_diagnostic_panel() -> pd.DataFrame:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    spread_panel = pd.read_csv(ROOT / "data" / "processed" / "daily_spread_panel.csv", parse_dates=["Date"])
    vix, _ = build_vix_regime_series()

    liquidity_frames: list[pd.DataFrame] = []
    for ticker in ETF_TICKERS:
        raw = load_and_clean_price_csv(RAW_DIR / f"{ticker}.csv")
        raw["dollar_volume"] = raw["Close"] * raw["Volume"]
        monthly_dollar_volume = (
            raw.set_index("Date")["dollar_volume"]
            .resample("ME")
            .mean()
            .rename("monthly_dollar_volume")
            .reset_index()
        )
        monthly_spread = (
            spread_panel.loc[spread_panel["symbol"] == ticker, ["Date", "corwin_schultz_spread"]]
            .set_index("Date")["corwin_schultz_spread"]
            .resample("ME")
            .mean()
            .rename("monthly_spread")
            .reset_index()
        )
        base = monthly_panel.loc[monthly_panel["symbol"] == ticker, ["Date", "symbol", "liquidity_1m"]].copy()
        base = base.rename(columns={"liquidity_1m": "amihud_illiquidity"})
        merged = base.merge(monthly_dollar_volume, on="Date", how="left")
        merged = merged.merge(monthly_spread, on="Date", how="left")
        liquidity_frames.append(merged)

    liquidity_panel = pd.concat(liquidity_frames, ignore_index=True)
    liquidity_panel["log_monthly_dollar_volume"] = np.log(liquidity_panel["monthly_dollar_volume"].replace(0, np.nan))
    liquidity_panel = liquidity_panel.merge(vix, on="Date", how="left")
    return liquidity_panel.sort_values(["Date", "symbol"]).reset_index(drop=True)


def build_regime_diagnostic_frame() -> pd.DataFrame:
    vix, _ = build_vix_regime_series()
    opportunity = build_cross_sectional_opportunity_frame()
    persistence = build_persistence_frame()
    commonality_rows: list[dict[str, object]] = []
    returns_matrix = build_monthly_return_matrix()
    for end_idx in range(11, len(returns_matrix)):
        window = returns_matrix.iloc[end_idx - 11 : end_idx + 1].dropna(how="any")
        if len(window) < 6:
            continue
        corr = window.corr()
        avg_pairwise_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean())
        first_pc_share = compute_first_pc_share(window)
        commonality_rows.append(
            {
                "Date": returns_matrix.index[end_idx],
                "avg_pairwise_corr": avg_pairwise_corr,
                "first_pc_variance_share": first_pc_share,
            }
        )

    commonality = pd.DataFrame(commonality_rows)
    regime_frame = vix.merge(opportunity, on="Date", how="left")
    regime_frame = regime_frame.merge(commonality, on="Date", how="left")
    regime_frame = regime_frame.merge(persistence, on="Date", how="left")
    return regime_frame


def first_adaptive_date(weights: pd.DataFrame, baseline_weights: list[float]) -> pd.Timestamp | None:
    cols = list(FACTOR_LABELS)
    mask = (weights[cols].round(10) != baseline_weights).any(axis=1)
    if not mask.any():
        return None
    return pd.to_datetime(weights.loc[mask, "Date"].iloc[0])


def build_vix_regime_series() -> tuple[pd.DataFrame, float]:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    vix = monthly_panel[["Date", "vix_close"]].drop_duplicates(subset=["Date"]).dropna().sort_values("Date")
    threshold = float(vix["vix_close"].quantile(0.75))
    vix["regime"] = np.where(vix["vix_close"] > threshold, "High VIX", "Normal VIX")
    return vix[["Date", "vix_close", "regime"]].copy(), threshold


def build_factor_ic_history() -> pd.DataFrame:
    panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    panel = panel.sort_values(["symbol", "Date"]).reset_index(drop=True)
    panel["next_month_return"] = panel.groupby("symbol")["Close"].shift(-1) / panel["Close"] - 1.0
    return compute_monthly_factor_ic(panel, CORE3_FACTOR_COLUMNS, "next_month_return")


def build_factor_ic_regime_long_frame() -> tuple[pd.DataFrame, float]:
    ic = build_factor_ic_history()
    vix, threshold = build_vix_regime_series()
    long_ic = ic.melt(id_vars="Date", var_name="factor", value_name="ic").dropna()
    long_ic = long_ic.merge(vix[["Date", "regime"]], on="Date", how="left")
    long_ic["factor"] = long_ic["factor"].map(FACTOR_IC_LABELS)
    return long_ic, threshold


def save_table_1_model_specification() -> None:
    shrinkage_rule = "Common validation-selected shrinkage to neutral weights"

    table = pd.DataFrame(
        [
            {
                "model_name": "B0",
                "model_family": "Naive benchmark",
                "weighting_logic": "Equal-weight allocation across all 9 sector ETFs; no factor signal used",
                "time_varying_weights": "No",
                "ml_or_not": "No",
                "shrinkage_treatment": "Not applicable",
            },
            {
                "model_name": "S1",
                "model_family": "Static retained-dimension model",
                "weighting_logic": "Equal weights across the three retained standardised signals",
                "time_varying_weights": "No",
                "ml_or_not": "No",
                "shrinkage_treatment": "Not applicable",
            },
            {
                "model_name": "A1",
                "model_family": "Rolling-IC adaptive model",
                "weighting_logic": "Dimension weights updated using trailing realised IC evidence",
                "time_varying_weights": "Yes",
                "ml_or_not": "No",
                "shrinkage_treatment": shrinkage_rule,
            },
            {
                "model_name": "L1",
                "model_family": "Ridge IC model",
                "weighting_logic": "Dimension weights based on Ridge-predicted dimension ICs",
                "time_varying_weights": "Yes",
                "ml_or_not": "Yes",
                "shrinkage_treatment": shrinkage_rule,
            },
            {
                "model_name": "L2",
                "model_family": "Lasso IC model",
                "weighting_logic": "Dimension weights based on Lasso-predicted dimension ICs",
                "time_varying_weights": "Yes",
                "ml_or_not": "Yes",
                "shrinkage_treatment": shrinkage_rule,
            },
            {
                "model_name": "L3",
                "model_family": "Elastic Net IC model",
                "weighting_logic": "Dimension weights based on Elastic Net-predicted dimension ICs",
                "time_varying_weights": "Yes",
                "ml_or_not": "Yes",
                "shrinkage_treatment": shrinkage_rule,
            },
            {
                "model_name": "T1",
                "model_family": "Random Forest IC model",
                "weighting_logic": "Dimension weights based on Random Forest-predicted dimension ICs",
                "time_varying_weights": "Yes",
                "ml_or_not": "Yes",
                "shrinkage_treatment": shrinkage_rule,
            },
            {
                "model_name": "T2",
                "model_family": "XGBoost IC model",
                "weighting_logic": "Dimension weights based on XGBoost-predicted dimension ICs",
                "time_varying_weights": "Yes",
                "ml_or_not": "Yes",
                "shrinkage_treatment": shrinkage_rule,
            },
        ]
    )
    rename_display_columns(table).to_csv(TABLES_DIR / "table_1_model_specification_summary.csv", index=False)


def save_table_2_main_comparison() -> None:
    table = load_table_csv("table_sh3_common_shrinkage_test_comparison.csv")
    ordered = [
        "equal_weight_benchmark",
        "fixed_weight",
        "rolling_ic",
        "ridge_ic",
        "lasso_ic",
        "elastic_net_ic",
        "random_forest_ic",
        "xgboost_ic",
    ]
    table["model"] = pd.Categorical(table["model"], categories=ordered, ordered=True)
    table = table.sort_values("model").copy()
    table = table[
        [
            "model",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "turnover",
            "net_return_after_costs",
        ]
    ].copy()
    table["model"] = table["model"].map(MODEL_LABELS)
    table = format_metric_tables(table)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_2_main_model_performance_comparison.csv", index=False)


def save_table_sh5_model_performance_by_vix_regime() -> None:
    test_summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv")
    test_period = str(test_summary.loc[0, "test_period"])
    test_start_str, test_end_str = [part.strip() for part in test_period.split(" to ")]
    test_start = pd.Timestamp(test_start_str)
    test_end = pd.Timestamp(test_end_str)

    vix, _ = build_vix_regime_series()
    ordered_models = [
        "equal_weight_benchmark",
        "fixed_weight",
        "rolling_ic",
        "ridge_ic",
        "lasso_ic",
        "elastic_net_ic",
        "random_forest_ic",
        "xgboost_ic",
    ]
    ordered_regimes = ["Normal VIX", "High VIX"]
    rows: list[dict[str, object]] = []

    for model in ordered_models:
        portfolio = load_csv(
            f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_{model}_portfolio_returns.csv",
            parse_dates=["Date"],
        )
        portfolio = portfolio[(portfolio["Date"] >= test_start) & (portfolio["Date"] <= test_end)].copy()
        portfolio = portfolio.merge(vix[["Date", "regime"]], on="Date", how="left")

        for regime in ordered_regimes:
            sample = portfolio[portfolio["regime"] == regime].copy()
            if len(sample) < 2:
                continue
            gross_returns = sample["portfolio_return"].fillna(0.0)
            net_returns = sample["net_portfolio_return"].fillna(0.0)
            gross_return = float((1.0 + gross_returns).prod() ** (12.0 / len(sample)) - 1.0)
            net_return = float((1.0 + net_returns).prod() ** (12.0 / len(sample)) - 1.0)
            net_volatility = float(net_returns.std(ddof=1) * np.sqrt(12.0))
            net_sharpe = np.nan if net_volatility == 0.0 or np.isnan(net_volatility) else float(net_return / net_volatility)
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    "months": len(sample),
                    "gross_return": gross_return,
                    "net_return": net_return,
                    "net_sharpe": net_sharpe,
                    "turnover": float(sample["turnover"].mean()),
                }
            )

    table = pd.DataFrame(rows)
    table["model"] = pd.Categorical(table["model"], categories=ordered_models, ordered=True)
    table["regime"] = pd.Categorical(table["regime"], categories=ordered_regimes, ordered=True)
    table = table.sort_values(["model", "regime"]).copy()
    table["model"] = table["model"].astype(str).map(MODEL_LABELS)
    table = format_metric_tables(table)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_sh5_model_performance_by_vix_regime.csv", index=False)


def save_table_4_turnover_framework_comparison() -> None:
    table = build_framework_comparison_test_window_frame()
    table = table[table["model"].isin(MAIN_FRAMEWORK_COMPARISON_MODELS)].copy()
    if table["model"].eq("equal_weight_benchmark").any():
        raise ValueError("Table 4 should exclude B0 from the execution-framework comparison.")
    table["model"] = table["model"].astype(str).map(MODEL_LABELS)
    table["framework"] = table["framework"].astype(str).map(FRAMEWORK_LABELS)
    table = table[
        [
            "framework",
            "model",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "turnover",
            "net_return_after_costs",
            "average_transaction_cost_rate",
        ]
    ].copy()
    table = format_metric_tables(table)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_4_turnover_framework_comparison.csv", index=False)


def save_table_5_holding_buffer_sensitivity() -> None:
    table = build_holding_buffer_sensitivity_test_window_frame()
    table = table[table["model"].isin(MAIN_FRAMEWORK_COMPARISON_MODELS)].copy()
    if table["model"].eq("equal_weight_benchmark").any():
        raise ValueError("Table 5 should exclude B0 from the holding-buffer sensitivity comparison.")
    table["model"] = table["model"].astype(str).map(MODEL_LABELS)
    table["framework"] = table["framework"].astype(str).map(FRAMEWORK_LABELS)
    table["holding_buffer_rank"] = table["holding_buffer_rank"].astype(int).map(HOLDING_BUFFER_RANK_LABELS)
    table = table[
        [
            "framework",
            "model",
            "holding_buffer_rank",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "turnover",
            "net_return_after_costs",
            "average_transaction_cost_rate",
        ]
    ].copy()
    table = format_metric_tables(table)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_5_holding_buffer_sensitivity.csv", index=False)


def save_table_d1_etf_summary_statistics() -> None:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    records: list[dict[str, object]] = []

    for ticker in ETF_TICKERS:
        etf = monthly_panel.loc[monthly_panel["symbol"] == ticker, ["Date", "Close"]].copy()
        etf = etf.sort_values("Date").reset_index(drop=True)
        etf["monthly_return"] = etf["Close"].pct_change()
        raw = load_and_clean_price_csv(RAW_DIR / f"{ticker}.csv")
        raw["dollar_volume"] = raw["Close"] * raw["Volume"]
        monthly_dollar_volume = (
            raw.set_index("Date")["dollar_volume"]
            .resample("ME")
            .mean()
            .dropna()
        )

        valid_returns = etf["monthly_return"].dropna()
        mean_monthly_return = float(valid_returns.mean()) if not valid_returns.empty else 0.0
        annualized_return = float((1.0 + valid_returns).prod() ** (12.0 / len(valid_returns)) - 1.0) if not valid_returns.empty else 0.0
        annualized_volatility = float(valid_returns.std(ddof=1) * (12.0**0.5)) if len(valid_returns) > 1 else 0.0
        max_drawdown = compute_drawdown_from_returns(valid_returns)
        avg_monthly_dollar_volume = float(monthly_dollar_volume.mean()) if not monthly_dollar_volume.empty else None

        records.append(
            {
                "ticker": ticker,
                "sample_start_date": etf["Date"].min().date().isoformat(),
                "sample_end_date": etf["Date"].max().date().isoformat(),
                "number_of_monthly_observations": int(etf["Close"].notna().sum()),
                "mean_monthly_return": mean_monthly_return,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "maximum_drawdown": max_drawdown,
                "average_monthly_dollar_volume": avg_monthly_dollar_volume,
            }
        )

    table = pd.DataFrame(records)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_d1_etf_summary_statistics.csv", index=False)


def save_table_d2_model_family_entry_timing() -> None:
    panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    core_columns = ["momentum_score_z", "liquidity_1m_z", "volatility_score_z"]
    first_core_month = (
        panel.dropna(subset=core_columns)
        .sort_values("Date")["Date"]
        .iloc[0]
    )

    common_window_start = pd.read_csv(
        BACKTEST_DIR / main_result_backtest_name("fixed_weight", "portfolio_returns"),
        parse_dates=["Date"],
    )["Date"].min()

    def first_non_neutral_weight_month(filename: str, neutral_weight: float = 1.0 / 3.0) -> pd.Timestamp:
        weights = pd.read_csv(BACKTEST_DIR / filename, parse_dates=["Date"])
        weight_columns = [column for column in weights.columns if column.endswith("_weight")]
        non_neutral = weights.loc[
            ~((weights[weight_columns] - neutral_weight).abs() < 1e-9).all(axis=1),
            "Date",
        ]
        return non_neutral.iloc[0]

    def fmt_date(ts: pd.Timestamp) -> str:
        return ts.strftime("%d/%m/%Y")

    rows = [
        {
            "Model / specification": "B0 benchmark",
            "Common evaluation start": fmt_date(common_window_start),
            "First model-driven month": fmt_date(common_window_start),
        },
        {
            "Model / specification": "S1 static retained-dimension model",
            "Common evaluation start": fmt_date(common_window_start),
            "First model-driven month": fmt_date(first_core_month),
        },
        {
            "Model / specification": "A1 rolling-IC adaptive model",
            "Common evaluation start": fmt_date(common_window_start),
            "First model-driven month": fmt_date(first_non_neutral_weight_month(main_result_backtest_name("rolling_ic", "weight_history"))),
        },
        {
            "Model / specification": "L1-L3 linear ML IC models",
            "Common evaluation start": fmt_date(common_window_start),
            "First model-driven month": fmt_date(first_non_neutral_weight_month(main_result_backtest_name("ridge_ic", "weight_history"))),
        },
        {
            "Model / specification": "T1 Random Forest IC model",
            "Common evaluation start": fmt_date(common_window_start),
            "First model-driven month": fmt_date(first_non_neutral_weight_month(main_result_backtest_name("random_forest_ic", "weight_history"))),
        },
        {
            "Model / specification": "T2 XGBoost IC model",
            "Common evaluation start": fmt_date(common_window_start),
            "First model-driven month": fmt_date(first_non_neutral_weight_month(main_result_backtest_name("xgboost_ic", "weight_history"))),
        },
    ]
    pd.DataFrame(rows).to_csv(TABLES_DIR / "table_d2_model_family_entry_timing.csv", index=False)


def make_figure_d1_etf_return_correlation_heatmap() -> None:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    returns = (
        monthly_panel[["Date", "symbol", "Close"]]
        .sort_values(["symbol", "Date"])
        .assign(monthly_return=lambda df: df.groupby("symbol")["Close"].pct_change())
        .pivot(index="Date", columns="symbol", values="monthly_return")
    )
    common_returns = returns[ETF_TICKERS].dropna(how="any")
    corr = common_returns.corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("ETF Return Correlation Matrix")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(FIGURES_DIR / "figure_d1_etf_return_correlation_heatmap.png")
    plt.close()


def make_figure_d2_factor_score_correlation_heatmap() -> None:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv")
    factor_frame = monthly_panel[
        [
            "momentum_score_z",
            "liquidity_1m_z",
            "volatility_score_z",
        ]
    ].dropna()
    factor_frame = factor_frame.rename(
        columns={
            "momentum_score_z": "Market-adjusted\nrelative perf.",
            "liquidity_1m_z": "Implementation\nfriction",
            "volatility_score_z": "Sector-specific\nuncertainty",
        }
    )
    corr = factor_frame.corr()

    plt.figure(figsize=(7.2, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Core Dimension Proxy Correlation")
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(FIGURES_DIR / "figure_d2_factor_score_correlation_heatmap.png")
    plt.close()


def make_figure_d3a_vix_level_over_time() -> None:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    vix = monthly_panel[["Date", "vix_close"]].drop_duplicates(subset=["Date"]).dropna().sort_values("Date")
    threshold = float(vix["vix_close"].quantile(0.75))

    plt.figure(figsize=(11, 6.5))
    plt.plot(vix["Date"], vix["vix_close"], color="#355070", linewidth=2.0)
    plt.axhline(threshold, color="#c1121f", linestyle="--", linewidth=1.2)
    plt.text(vix["Date"].iloc[-1], threshold, f"75th percentile = {threshold:.2f}", va="bottom", ha="right", fontsize=10)
    plt.title("VIX Level and High-Volatility Threshold")
    plt.xlabel("Date")
    plt.ylabel("VIX level")
    plt.savefig(FIGURES_DIR / "figure_d3a_vix_level_threshold.png")
    plt.close()


def make_figure_d3b_vix_stress_regime_distribution() -> None:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    vix = monthly_panel[["Date", "vix_close"]].drop_duplicates(subset=["Date"]).dropna().sort_values("Date")
    threshold = float(vix["vix_close"].quantile(0.75))
    vix["regime"] = vix["vix_close"].apply(lambda value: "High VIX" if value > threshold else "Normal VIX")
    counts = vix["regime"].value_counts().reindex(["Normal VIX", "High VIX"]).fillna(0)

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=["#8ecae6", "#d62828"], legend=False)
    plt.title("Distribution of VIX Stress Regimes")
    plt.xlabel("")
    plt.ylabel("Number of months")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)
    plt.savefig(FIGURES_DIR / "figure_d3b_vix_stress_regime_distribution.png")
    plt.close()


def make_figure_d4_etf_risk_return_scatter() -> None:
    table = pd.read_csv(TABLES_DIR / "table_d1_etf_summary_statistics.csv")
    plot_frame = table.dropna(subset=["Annualised return", "Annualised volatility"]).copy()
    plot_frame["annualised_return_pct"] = plot_frame["Annualised return"] * 100.0
    plot_frame["annualised_volatility_pct"] = plot_frame["Annualised volatility"] * 100.0

    plt.figure(figsize=(8.5, 6.5))
    plt.scatter(
        plot_frame["annualised_volatility_pct"],
        plot_frame["annualised_return_pct"],
        color="#2a9d8f",
        s=70,
        alpha=0.9,
    )
    for _, row in plot_frame.iterrows():
        plt.text(
            row["annualised_volatility_pct"] + 0.12,
            row["annualised_return_pct"] + 0.05,
            row["Ticker"],
            fontsize=9,
        )
    plt.title("Sector ETF Annualised Return and Volatility")
    plt.xlabel("Annualised volatility (%)")
    plt.ylabel("Annualised return (%)")
    plt.savefig(FIGURES_DIR / "figure_d4_etf_risk_return_scatter.png")
    plt.close()


def save_table_dd1_sector_panel_characteristics() -> None:
    returns = load_monthly_return_panel()
    rows: list[dict[str, object]] = []
    for ticker in ETF_TICKERS:
        sample = (
            returns.loc[returns["symbol"] == ticker, ["Date", "monthly_return"]]
            .sort_values("Date")["monthly_return"]
            .dropna()
        )
        if sample.empty:
            continue
        rows.append(
            {
                "ticker": ticker,
                "mean_monthly_return": float(sample.mean()),
                "monthly_volatility": float(sample.std(ddof=1)) if len(sample) > 1 else None,
                "return_skewness": float(sample.skew()) if len(sample) > 2 else None,
                "excess_kurtosis": float(sample.kurt()) if len(sample) > 3 else None,
                "return_p05": float(sample.quantile(0.05)),
                "expected_shortfall_5": float(sample.loc[sample <= sample.quantile(0.05)].mean()),
                "maximum_drawdown": compute_drawdown_from_returns(sample),
            }
        )
    table = pd.DataFrame(rows)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_dd1_sector_panel_characteristics.csv", index=False)


def make_figure_dd1_sector_return_distribution_boxplot() -> None:
    returns = load_monthly_return_panel().dropna(subset=["monthly_return"]).copy()
    returns["monthly_return_pct"] = returns["monthly_return"] * 100.0
    plt.figure(figsize=(10.5, 6.2))
    sns.boxplot(
        data=returns,
        x="symbol",
        y="monthly_return_pct",
        hue="symbol",
        palette="Set2",
        legend=False,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Monthly Return Distributions Across Sector ETFs")
    plt.xlabel("ETF")
    plt.ylabel("Monthly return (%)")
    plt.savefig(FIGURES_DIR / "figure_dd1_sector_return_distribution_boxplot.png")
    plt.close()


def save_table_dd2_cross_sectional_opportunity_set() -> None:
    opportunity = build_cross_sectional_opportunity_frame()
    metric_map = {
        "cross_sectional_std": ("Cross-sectional standard deviation (%)", 100.0),
        "cross_sectional_iqr": ("Cross-sectional interquartile range (%)", 100.0),
        "top_minus_bottom_spread": ("Top-minus-bottom spread (%)", 100.0),
    }
    rows: list[dict[str, object]] = []
    for column, (label, scale) in metric_map.items():
        sample = opportunity[column].dropna() * scale
        rows.append(
            {
                "metric": label,
                "mean": float(sample.mean()),
                "median": float(sample.median()),
                "p10": float(sample.quantile(0.10)),
                "p90": float(sample.quantile(0.90)),
                "min": float(sample.min()),
                "max": float(sample.max()),
            }
        )
    rename_display_columns(pd.DataFrame(rows)).to_csv(TABLES_DIR / "table_dd2_cross_sectional_opportunity_set.csv", index=False)


def make_figure_dd2_cross_sectional_opportunity_timeseries() -> None:
    opportunity = build_cross_sectional_opportunity_frame().copy()
    plot_specs = [
        ("cross_sectional_std", "Cross-sectional std"),
        ("cross_sectional_iqr", "Cross-sectional IQR"),
        ("top_minus_bottom_spread", "Top-bottom spread"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.8), sharex=True)
    colors = ["#355070", "#6d597a", "#e76f51"]
    event_windows = [
        (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30")),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-31")),
    ]
    for ax, (column, label), color in zip(axes, plot_specs, colors):
        ax.plot(opportunity["Date"], opportunity[column] * 100.0, color=color, linewidth=1.8)
        ax.set_ylabel(f"{label} (%)")
        for start, end in event_windows:
            ax.axvspan(start, end, color="#adb5bd", alpha=0.18, zorder=0)
        ax.grid(alpha=0.25)
    axes[0].set_title("Cross-Sectional Opportunity Set Over Time")
    axes[-1].set_xlabel("Date")
    fig.savefig(FIGURES_DIR / "figure_dd2_cross_sectional_opportunity_timeseries.png")
    plt.close(fig)


def save_table_dd3_common_component_and_market_dependence() -> None:
    table = build_market_dependence_table().copy()
    rename_display_columns(table).to_csv(TABLES_DIR / "table_dd3_common_component_market_dependence.csv", index=False)


def make_figure_dd3_market_dependence_bar() -> None:
    table = build_market_dependence_table().copy()
    plot_frame = table.copy()
    plot_frame["r_squared_pct"] = plot_frame["r_squared_to_spy"] * 100.0
    plot_frame = plot_frame.sort_values("r_squared_pct", ascending=False)

    plt.figure(figsize=(9.5, 6.2))
    ax = sns.barplot(data=plot_frame, x="ticker", y="r_squared_pct", hue="ticker", palette="crest", legend=False)
    plt.title("Sector ETF Dependence on SPY")
    plt.xlabel("ETF")
    plt.ylabel("Monthly return variance explained by SPY (%)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)
    plt.savefig(FIGURES_DIR / "figure_dd3_market_dependence_bar.png")
    plt.close()


def save_table_dd4_regime_dependence() -> None:
    regime_frame = build_regime_diagnostic_frame()
    metric_map = {
        "cross_sectional_std": ("Cross-sectional standard deviation (%)", 100.0),
        "cross_sectional_iqr": ("Cross-sectional interquartile range (%)", 100.0),
        "top_minus_bottom_spread": ("Top-minus-bottom spread (%)", 100.0),
        "avg_pairwise_corr": ("Average pairwise correlation", 1.0),
        "first_pc_variance_share": ("First principal component variance share (%)", 100.0),
        "top_3_membership_turnover": ("Top-3 membership turnover (%)", 100.0),
    }
    rows: list[dict[str, object]] = []
    for regime in ["Normal VIX", "High VIX"]:
        sample = regime_frame.loc[regime_frame["regime"] == regime]
        for column, (label, scale) in metric_map.items():
            values = sample[column].dropna() * scale
            if values.empty:
                continue
            rows.append(
                {
                    "sample": regime,
                    "metric": label,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    rename_display_columns(pd.DataFrame(rows)).to_csv(TABLES_DIR / "table_dd4_regime_dependence.csv", index=False)


def save_table_dd4_regime_mean_difference_summary() -> None:
    regime_frame = build_regime_diagnostic_frame()
    metric_map = {
        "cross_sectional_std": ("Cross-sectional standard deviation (%)", 100.0),
        "top_minus_bottom_spread": ("Top-minus-bottom spread (%)", 100.0),
        "avg_pairwise_corr": ("Average pairwise correlation", 1.0),
        "first_pc_variance_share": ("First principal component variance share (%)", 100.0),
        "top_3_membership_turnover": ("Top-3 membership turnover (%)", 100.0),
    }
    normal = regime_frame.loc[regime_frame["regime"] == "Normal VIX"]
    high = regime_frame.loc[regime_frame["regime"] == "High VIX"]
    rows: list[dict[str, object]] = []
    for column, (label, scale) in metric_map.items():
        normal_values = normal[column].dropna() * scale
        high_values = high[column].dropna() * scale
        if normal_values.empty or high_values.empty:
            continue
        rows.append(
            {
                "metric": label,
                "normal_vix_months": int(normal_values.shape[0]),
                "high_vix_months": int(high_values.shape[0]),
                "normal_vix_mean": float(normal_values.mean()),
                "high_vix_mean": float(high_values.mean()),
                "high_minus_normal": float(high_values.mean() - normal_values.mean()),
            }
        )
    rename_display_columns(pd.DataFrame(rows)).to_csv(TABLES_DIR / "table_dd4_regime_mean_difference_summary.csv", index=False)


def make_figure_dd4_regime_dependence_comparison() -> None:
    regime_frame = build_regime_diagnostic_frame()
    regimes = ["Normal VIX", "High VIX"]
    palette = {"Normal VIX": "#8ecae6", "High VIX": "#d62828"}

    def summarise(metric: str, scale: float = 1.0) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for regime in regimes:
            values = regime_frame.loc[regime_frame["regime"] == regime, metric].dropna() * scale
            rows.append(
                {
                    "regime": regime,
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
            )
        return pd.DataFrame(rows)

    opportunity_metrics = [
        ("Cross-sectional std of returns", summarise("cross_sectional_std", 100.0)),
        ("Top-bottom spread", summarise("top_minus_bottom_spread", 100.0)),
    ]
    commonality = summarise("first_pc_variance_share", 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8), gridspec_kw={"width_ratios": [1.35, 0.95]})

    x = np.arange(len(opportunity_metrics))
    width = 0.34
    for idx, regime in enumerate(regimes):
        offsets = x + (-width / 2 if idx == 0 else width / 2)
        medians = [frame.loc[frame["regime"] == regime, "median"].iloc[0] for _, frame in opportunity_metrics]
        lower = [m - frame.loc[frame["regime"] == regime, "q25"].iloc[0] for m, (_, frame) in zip(medians, opportunity_metrics)]
        upper = [frame.loc[frame["regime"] == regime, "q75"].iloc[0] - m for m, (_, frame) in zip(medians, opportunity_metrics)]
        axes[0].bar(offsets, medians, width=width, color=palette[regime], label=regime)
        axes[0].errorbar(offsets, medians, yerr=[lower, upper], fmt="none", ecolor="#444444", capsize=4, lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([label.replace(" of returns", "") for label, _ in opportunity_metrics], rotation=0)
    axes[0].set_title("Opportunity set")
    axes[0].set_ylabel("Median monthly value (%)")

    commonality_x = np.arange(1)
    commonality_width = 0.44
    for idx, regime in enumerate(regimes):
        row = commonality.loc[commonality["regime"] == regime].iloc[0]
        offset = commonality_x + (-commonality_width / 2 if idx == 0 else commonality_width / 2)
        axes[1].bar(offset, row["median"], width=commonality_width, color=palette[regime], label=regime)
        axes[1].errorbar(
            offset,
            row["median"],
            yerr=[[row["median"] - row["q25"]], [row["q75"] - row["median"]]],
            fmt="none",
            ecolor="#444444",
            capsize=4,
            lw=1,
        )
    axes[1].set_xticks(commonality_x)
    axes[1].set_xticklabels(["PC1 variance share"])
    axes[1].set_title("Commonality")
    axes[1].set_ylabel("Median monthly share (%)")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    handles = [Patch(facecolor=palette[regime], label=regime) for regime in regimes]
    fig.legend(handles=handles, labels=regimes, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Ranking Environment by VIX Regime", y=1.03, fontsize=18)
    fig.text(
        0.5,
        -0.02,
        "Bars show regime medians; whiskers show interquartile ranges. Commonality is measured as the share of cross-sector return variance explained by the first principal component.",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_dd4_regime_dependence_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_table_dd5_persistence_and_implementation_frictions() -> None:
    persistence = build_persistence_frame()
    metric_map = {
        "top_1_persistence": "Top-1 persistence",
        "top_3_retention_rate": "Top-3 retention rate",
        "bottom_3_retention_rate": "Bottom-3 retention rate",
        "top_3_membership_turnover": "Top-3 membership turnover",
        "rank_autocorrelation": "Rank autocorrelation",
    }
    rows: list[dict[str, object]] = []
    for column, label in metric_map.items():
        values = persistence[column].dropna()
        rows.append(
            {
                "metric": label,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p10": float(values.quantile(0.10)),
                "p90": float(values.quantile(0.90)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    rename_display_columns(pd.DataFrame(rows)).to_csv(TABLES_DIR / "table_dd5_persistence_implementation_frictions.csv", index=False)


def save_table_dd6_transition_matrix() -> None:
    transitions = build_leadership_transition_frame()
    if transitions.empty:
        return
    transition = (
        transitions.groupby(["from_bucket", "to_bucket"])
        .size()
        .div(transitions.groupby("from_bucket").size(), level="from_bucket")
        .reset_index(name="transition_probability")
    )
    transition["from_bucket"] = pd.Categorical(
        transition["from_bucket"],
        categories=["Top tercile", "Middle tercile", "Bottom tercile"],
        ordered=True,
    )
    transition["to_bucket"] = pd.Categorical(
        transition["to_bucket"],
        categories=["Top tercile", "Middle tercile", "Bottom tercile"],
        ordered=True,
    )
    transition = transition.sort_values(["from_bucket", "to_bucket"]).copy()
    rename_display_columns(transition).to_csv(TABLES_DIR / "table_dd6_transition_matrix.csv", index=False)


def save_table_ld1_sector_liquidity_distribution() -> None:
    liquidity = build_monthly_liquidity_diagnostic_panel()
    rows: list[dict[str, object]] = []
    for ticker in ETF_TICKERS:
        sample = liquidity.loc[liquidity["symbol"] == ticker].copy()
        dollar = sample["monthly_dollar_volume"].dropna()
        amihud = sample["amihud_illiquidity"].dropna()
        log10_amihud = np.log10(amihud.clip(lower=1e-12)) if not amihud.empty else amihud
        spread = sample["monthly_spread"].dropna()
        rows.append(
            {
                "ticker": ticker,
                "median_monthly_dollar_volume": float(dollar.median()) if not dollar.empty else None,
                "p10_monthly_dollar_volume": float(dollar.quantile(0.10)) if not dollar.empty else None,
                "p90_monthly_dollar_volume": float(dollar.quantile(0.90)) if not dollar.empty else None,
                "mean_amihud_illiquidity": float(log10_amihud.mean()) if not amihud.empty else None,
                "median_amihud_illiquidity": float(log10_amihud.median()) if not amihud.empty else None,
                "p90_amihud_illiquidity": float(log10_amihud.quantile(0.90)) if not amihud.empty else None,
                "mean_spread": float(spread.mean()) if not spread.empty else None,
            }
        )
    table = pd.DataFrame(rows)
    rename_display_columns(table).to_csv(TABLES_DIR / "table_ld1_sector_liquidity_distribution.csv", index=False)


def make_figure_ld1_sector_dollar_volume_distribution() -> None:
    liquidity = build_monthly_liquidity_diagnostic_panel().dropna(subset=["monthly_dollar_volume"]).copy()
    liquidity["log_monthly_dollar_volume"] = np.log10(liquidity["monthly_dollar_volume"])
    plt.figure(figsize=(10.5, 6.2))
    sns.boxplot(
        data=liquidity,
        x="symbol",
        y="log_monthly_dollar_volume",
        hue="symbol",
        palette="Set2",
        legend=False,
    )
    plt.title("Sector-Level Monthly Dollar Volume Distributions")
    plt.xlabel("ETF")
    plt.ylabel("log10(monthly average daily dollar volume, USD)")
    plt.savefig(FIGURES_DIR / "figure_ld1_sector_dollar_volume_distribution.png")
    plt.close()


def make_figure_ld2_sector_amihud_distribution() -> None:
    liquidity = build_monthly_liquidity_diagnostic_panel().dropna(subset=["amihud_illiquidity"]).copy()
    liquidity["log_amihud_illiquidity"] = np.log10(liquidity["amihud_illiquidity"].clip(lower=1e-12))
    plt.figure(figsize=(10.5, 6.2))
    sns.boxplot(
        data=liquidity,
        x="symbol",
        y="log_amihud_illiquidity",
        hue="symbol",
        palette="flare",
        legend=False,
    )
    plt.title("Sector-Level Amihud Illiquidity Distributions")
    plt.xlabel("ETF")
    plt.ylabel("log10(Amihud illiquidity)")
    plt.savefig(FIGURES_DIR / "figure_ld2_sector_amihud_distribution.png")
    plt.close()


def build_liquidity_state_frame() -> pd.DataFrame:
    liquidity = build_monthly_liquidity_diagnostic_panel()
    opportunity = build_cross_sectional_opportunity_frame()
    regime = build_regime_diagnostic_frame()[["Date", "avg_pairwise_corr", "first_pc_variance_share"]].drop_duplicates(subset=["Date"])
    rows: list[dict[str, object]] = []
    for date, group in liquidity.groupby("Date"):
        clean = group.dropna(subset=["monthly_dollar_volume", "amihud_illiquidity", "monthly_spread"])
        if len(clean) < 2:
            continue
        rows.append(
            {
                "Date": date,
                "avg_log_dollar_volume": float(clean["log_monthly_dollar_volume"].mean()),
                "cross_sectional_std_dollar_volume": float(clean["log_monthly_dollar_volume"].std(ddof=1)),
                "average_amihud_illiquidity": float(clean["amihud_illiquidity"].mean()),
                "cross_sectional_std_amihud": float(clean["amihud_illiquidity"].std(ddof=1)),
                "average_corwin_schultz_spread": float(clean["monthly_spread"].mean()),
                "cross_sectional_std_spread": float(clean["monthly_spread"].std(ddof=1)),
                "regime": clean["regime"].iloc[0],
                "vix_close": clean["vix_close"].iloc[0],
            }
        )
    state = pd.DataFrame(rows)
    state = state.merge(opportunity, on="Date", how="left")
    state = state.merge(regime, on="Date", how="left")
    return state.sort_values("Date").reset_index(drop=True)


def save_table_ld2_liquidity_by_vix_regime() -> None:
    state = build_liquidity_state_frame()
    metric_map = {
        "avg_log_dollar_volume": ("Average ln dollar volume", None),
        "cross_sectional_std_dollar_volume": ("Cross-sectional std of ln dollar volume", None),
        "average_amihud_illiquidity": ("log10 average Amihud illiquidity", "log10"),
        "cross_sectional_std_amihud": ("log10 cross-sectional std of Amihud illiquidity", "log10"),
        "average_corwin_schultz_spread": "Average Corwin-Schultz spread",
        "cross_sectional_std_spread": "Cross-sectional std of Corwin-Schultz spread",
    }
    rows: list[dict[str, object]] = []
    for regime in ["Normal VIX", "High VIX"]:
        sample = state.loc[state["regime"] == regime]
        for column, spec in metric_map.items():
            if isinstance(spec, tuple):
                label, transform = spec
            else:
                label, transform = spec, None
            values = sample[column].dropna()
            if transform == "log10":
                values = np.log10(values.clip(lower=1e-12))
            if values.empty:
                continue
            rows.append(
                {
                    "sample": regime,
                    "metric": label,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p10": float(values.quantile(0.10)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    rename_display_columns(pd.DataFrame(rows)).to_csv(TABLES_DIR / "table_ld2_liquidity_by_vix_regime.csv", index=False)


def make_figure_ld3_liquidity_stress_over_time() -> None:
    state = build_liquidity_state_frame()
    plot_frame = state.copy()
    plot_frame["log_average_amihud"] = np.log10(plot_frame["average_amihud_illiquidity"].clip(lower=1e-12))
    plot_frame["log_cross_sectional_std_amihud"] = np.log10(plot_frame["cross_sectional_std_amihud"].clip(lower=1e-12))
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.8), sharex=True)
    axes[0].plot(plot_frame["Date"], plot_frame["log_average_amihud"], color="#9d0208", linewidth=1.8)
    axes[0].set_ylabel("log10 Avg Amihud")
    axes[0].set_title("Liquidity Stress Over Time")
    axes[1].plot(plot_frame["Date"], plot_frame["average_corwin_schultz_spread"], color="#355070", linewidth=1.8)
    axes[1].set_ylabel("Avg spread")
    axes[2].plot(plot_frame["Date"], plot_frame["log_cross_sectional_std_amihud"], color="#6d597a", linewidth=1.8)
    axes[2].set_ylabel("log10 Cross-sec std Amihud")
    axes[2].set_xlabel("Date")

    event_windows = [
        (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-06-30")),
        (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-05-31")),
    ]
    for ax in axes:
        for start, end in event_windows:
            ax.axvspan(start, end, color="#adb5bd", alpha=0.18, zorder=0)
        ax.grid(alpha=0.25)
    fig.text(
        0.01,
        0.01,
        "Note: Shaded bands mark September 2008 to June 2009 and February 2020 to May 2020.",
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES_DIR / "figure_ld3_liquidity_stress_over_time.png", bbox_inches="tight")
    plt.close(fig)


def make_figure_ld4_liquidity_regime_comparison() -> None:
    state = build_liquidity_state_frame()
    plot_frame = state.copy()
    metric_columns = {
        "average_amihud_illiquidity": "Avg Amihud",
        "cross_sectional_std_amihud": "Cross-sec std Amihud",
        "average_corwin_schultz_spread": "Avg spread",
        "cross_sectional_std_dollar_volume": "Cross-sec std dollar volume",
    }
    for column in metric_columns:
        series = plot_frame[column]
        std = series.std(ddof=1)
        plot_frame[column] = 0.0 if pd.isna(std) or std == 0.0 else (series - series.mean()) / std

    summary = pd.DataFrame(
        [
            {"regime": regime, **{
                label: plot_frame.loc[plot_frame["regime"] == regime, column].mean()
                for column, label in metric_columns.items()
            }}
            for regime in ["Normal VIX", "High VIX"]
        ]
    )
    long = summary.melt(id_vars="regime", var_name="metric", value_name="value")
    plt.figure(figsize=(11, 6.2))
    sns.barplot(data=long, x="metric", y="value", hue="regime", palette=["#8ecae6", "#d62828"])
    plt.title("Liquidity Conditions by VIX Regime")
    plt.xlabel("")
    plt.ylabel("Mean standardized value")
    plt.legend(title="")
    plt.savefig(FIGURES_DIR / "figure_ld4_liquidity_regime_comparison.png")
    plt.close()


def save_table_ld3_liquidity_opportunity_commonality_comovement() -> None:
    state = build_liquidity_state_frame()
    metric_map = {
        "average_amihud_illiquidity": "Average Amihud illiquidity",
        "cross_sectional_std_amihud": "Cross-sectional std of Amihud illiquidity",
        "average_corwin_schultz_spread": "Average Corwin-Schultz spread",
        "cross_sectional_std_dollar_volume": "Cross-sectional std of dollar volume",
    }
    rows: list[dict[str, object]] = []
    for column, label in metric_map.items():
        rows.append(
            {
                "metric": label,
                "corr_with_cross_sectional_std": float(state[column].corr(state["cross_sectional_std"])) if state[column].notna().sum() > 2 else None,
                "corr_with_top_minus_bottom_spread": float(state[column].corr(state["top_minus_bottom_spread"])) if state[column].notna().sum() > 2 else None,
                "corr_with_avg_pairwise_corr": float(state[column].corr(state["avg_pairwise_corr"])) if state[column].notna().sum() > 2 else None,
                "corr_with_first_pc_share": float(state[column].corr(state["first_pc_variance_share"])) if state[column].notna().sum() > 2 else None,
            }
        )
    rename_display_columns(pd.DataFrame(rows)).to_csv(TABLES_DIR / "table_ld3_liquidity_opportunity_commonality_comovement.csv", index=False)


def make_figure_ld5_liquidity_dispersion_commonality_scatter() -> None:
    state = build_liquidity_state_frame()
    plot_frame = state.copy()
    plot_frame["log_average_amihud"] = np.log10(plot_frame["average_amihud_illiquidity"].clip(lower=1e-12))
    palette = {"Normal VIX": "#8ecae6", "High VIX": "#d62828"}
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    sns.scatterplot(
        data=plot_frame,
        x="log_average_amihud",
        y="cross_sectional_std",
        hue="regime",
        palette=palette,
        alpha=0.8,
        s=40,
        ax=axes[0],
    )
    axes[0].set_title("Liquidity Stress vs Opportunity Set")
    axes[0].set_xlabel("log10(Average Amihud illiquidity)")
    axes[0].set_ylabel("Cross-sectional std of returns")
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()

    sns.scatterplot(
        data=plot_frame,
        x="average_corwin_schultz_spread",
        y="avg_pairwise_corr",
        hue="regime",
        palette=palette,
        alpha=0.8,
        s=40,
        ax=axes[1],
        legend=True,
    )
    axes[1].set_title("Trading Friction vs Commonality")
    axes[1].set_xlabel("Average Corwin-Schultz spread")
    axes[1].set_ylabel("Average pairwise correlation")
    axes[1].legend(title="Regime", loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_ld5_liquidity_dispersion_commonality_scatter.png")
    plt.close(fig)


def make_figure_dd5_transition_matrix_heatmap() -> None:
    transitions = build_leadership_transition_frame()
    if transitions.empty:
        return
    transition = (
        transitions.groupby(["from_bucket", "to_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=["Top tercile", "Middle tercile", "Bottom tercile"], columns=["Top tercile", "Middle tercile", "Bottom tercile"])
    )
    transition = transition.div(transition.sum(axis=1), axis=0)

    plt.figure(figsize=(7.8, 6.4))
    sns.heatmap(transition, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"shrink": 0.8})
    plt.title("Return-Rank Tercile Transition Matrix")
    plt.xlabel("Next month")
    plt.ylabel("Current month")
    plt.savefig(FIGURES_DIR / "figure_dd5_transition_matrix_heatmap.png")
    plt.close()


def make_figure_dd6_top3_retention_turnover_timeseries() -> None:
    persistence = build_persistence_frame().copy()
    if persistence.empty:
        return
    persistence = persistence.sort_values("Date").copy()
    persistence["rolling_top3_retention_pct"] = persistence["top_3_retention_rate"].rolling(12, min_periods=6).mean() * 100.0
    persistence["rolling_top3_replaced_pct"] = 100.0 - persistence["rolling_top3_retention_pct"]
    persistence["rolling_top1_persistence_pct"] = persistence["top_1_persistence"].rolling(12, min_periods=6).mean() * 100.0
    persistence["rolling_rank_autocorr_scaled"] = persistence["rank_autocorrelation"].rolling(12, min_periods=6).mean() * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.6), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.15]})

    axes[0].fill_between(
        persistence["Date"],
        0,
        persistence["rolling_top3_retention_pct"],
        color="#2a9d8f",
        alpha=0.82,
        label="Retained from previous top-3",
    )
    axes[0].fill_between(
        persistence["Date"],
        persistence["rolling_top3_retention_pct"],
        100,
        color="#e76f51",
        alpha=0.74,
        label="Replaced in current top-3",
    )
    axes[0].plot(persistence["Date"], persistence["rolling_top3_retention_pct"], color="#1f7f73", linewidth=1.3)
    axes[0].set_title("Rolling Replacement Pressure in Realised Sector Leaders")
    axes[0].set_ylabel("12m avg share (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=True, loc="upper right")
    axes[0].grid(axis="y", alpha=0.22)

    axes[1].plot(
        persistence["Date"],
        persistence["rolling_top1_persistence_pct"],
        color="#355070",
        linewidth=1.8,
        label="Top-1 persistence (12m avg, %)",
    )
    axes[1].plot(
        persistence["Date"],
        persistence["rolling_rank_autocorr_scaled"],
        color="#6d597a",
        linewidth=1.8,
        label="Rank autocorr. (12m avg, x100)",
    )
    axes[1].axhline(0.0, color="#495057", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Stability")
    axes[1].set_xlabel("Year")
    axes[1].legend(frameon=True, loc="upper right")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_dd6_top3_retention_turnover_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_figure_m1_factor_ic_distribution_boxplot() -> None:
    long_ic, _ = build_factor_ic_regime_long_frame()

    plt.figure(figsize=(9, 6.2))
    sns.boxplot(
        data=long_ic,
        x="factor",
        y="ic",
        hue="factor",
        palette=["#4c78a8", "#f58518", "#c44e52"],
        legend=False,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Distribution of Monthly Core-Dimension Information Coefficients")
    plt.xlabel("")
    plt.ylabel("Monthly dimension IC")
    plt.savefig(FIGURES_DIR / "figure_m1_factor_ic_distribution_boxplot.png")
    plt.close()


def save_table_m1_factor_ic_by_vix_regime() -> None:
    long_ic, _ = build_factor_ic_regime_long_frame()
    factor_order = [
        "Market-adjusted relative performance",
        "Implementation friction",
        "Sector-specific uncertainty",
    ]
    sample_order = ["Overall", "Normal VIX", "High VIX"]
    rows: list[dict[str, object]] = []

    for factor in factor_order:
        overall = long_ic.loc[long_ic["factor"] == factor, "ic"].dropna()
        rows.append(
            {
                "factor": factor,
                "sample": "Overall",
                "n_months": int(len(overall)),
                "mean_ic": float(overall.mean()),
                "median_ic": float(overall.median()),
                "std_ic": float(overall.std(ddof=1)),
                "p25": float(overall.quantile(0.25)),
                "p75": float(overall.quantile(0.75)),
                "share_positive": float((overall > 0).mean()),
            }
        )
        for regime in ["Normal VIX", "High VIX"]:
            sample = long_ic.loc[(long_ic["factor"] == factor) & (long_ic["regime"] == regime), "ic"].dropna()
            rows.append(
                {
                    "factor": factor,
                    "sample": regime,
                    "n_months": int(len(sample)),
                    "mean_ic": float(sample.mean()),
                    "median_ic": float(sample.median()),
                    "std_ic": float(sample.std(ddof=1)),
                    "p25": float(sample.quantile(0.25)),
                    "p75": float(sample.quantile(0.75)),
                    "share_positive": float((sample > 0).mean()),
                }
            )

    table = pd.DataFrame(rows)
    table["factor"] = pd.Categorical(table["factor"], categories=factor_order, ordered=True)
    table["sample"] = pd.Categorical(table["sample"], categories=sample_order, ordered=True)
    table = table.sort_values(["factor", "sample"]).copy()
    table["mean_ic"] = table["mean_ic"].round(4)
    table["median_ic"] = table["median_ic"].round(4)
    table["std_ic"] = table["std_ic"].round(4)
    table["p25"] = table["p25"].round(4)
    table["p75"] = table["p75"].round(4)
    table["share_positive"] = format_percentage(table["share_positive"], decimals=2)
    renamed = rename_display_columns(table)
    renamed.to_csv(TABLES_DIR / "table_m1_factor_ic_by_vix_regime.csv", index=False)


def make_figure_m3_factor_ic_by_vix_regime_boxplot() -> None:
    long_ic, threshold = build_factor_ic_regime_long_frame()
    long_ic["regime"] = pd.Categorical(long_ic["regime"], categories=["Normal VIX", "High VIX"], ordered=True)

    plt.figure(figsize=(10.5, 6.4))
    sns.boxplot(
        data=long_ic,
        x="factor",
        y="ic",
        hue="regime",
        palette=["#7aa6c2", "#d17c5b"],
        width=0.72,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
    plt.title("Core-Dimension IC Distributions by VIX Regime")
    plt.xlabel("")
    plt.ylabel("Monthly dimension IC")
    plt.legend(title="", frameon=True, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2)
    plt.figtext(
        0.01,
        -0.03,
        f"Note: High VIX months are defined as months with VIX above the 75th-percentile threshold ({threshold:.2f}).",
        ha="left",
        fontsize=10,
    )
    plt.savefig(FIGURES_DIR / "figure_m3_factor_ic_by_vix_regime_boxplot.png")
    plt.close()


def make_figure_m2_rolling_12m_factor_ic_heatmap() -> None:
    ic = build_factor_ic_history().sort_values("Date")
    factor_map = {
        "momentum_score_z_ic": "Market-adjusted relative performance",
        "liquidity_1m_z_ic": "Implementation friction",
        "volatility_score_z_ic": "Sector-specific uncertainty",
    }
    rolling = ic.copy()
    for column in factor_map:
        rolling[column] = rolling[column].rolling(12, min_periods=6).mean()
    heatmap_frame = (
        rolling.set_index("Date")[list(factor_map)]
        .rename(columns=factor_map)
        .T
    )

    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#e9eef5")
    sns.heatmap(
        heatmap_frame,
        cmap=cmap,
        center=0,
        cbar_kws={"label": "Rolling 12-month dimension IC"},
        ax=ax,
    )
    x_positions = np.arange(len(heatmap_frame.columns)) + 0.5
    date_index = pd.Index(heatmap_frame.columns)
    year_mask = date_index.month == 12
    tick_positions = x_positions[year_mask]
    tick_labels = [str(date.year) for date in date_index[year_mask]]
    if len(tick_positions) > 12:
        tick_positions = tick_positions[::2]
        tick_labels = tick_labels[::2]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)

    ax.set_title("Rolling 12-Month Core-Dimension IC Heatmap")
    ax.set_xlabel("Date")
    ax.set_ylabel("")
    plt.figtext(
        0.01,
        -0.03,
        "Note: Colors show 12-month rolling averages of monthly dimension ICs. Light-grey cells indicate periods in which the retained proxy is unavailable because it requires a formation window.",
        ha="left",
        fontsize=10,
    )
    plt.savefig(FIGURES_DIR / "figure_m2_rolling_12m_factor_ic_heatmap.png")
    plt.close()



def make_figure_1_cumulative_wealth() -> None:
    summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv").iloc[0]
    test_start_str, test_end_str = [part.strip() for part in str(summary["test_period"]).split(" to ")]
    test_start = pd.Timestamp(test_start_str)
    test_end = pd.Timestamp(test_end_str)

    plt.figure(figsize=(11, 6.5))
    for file_name, model_key in [
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_equal_weight_benchmark_portfolio_returns.csv", "equal_weight_benchmark"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_fixed_weight_portfolio_returns.csv", "fixed_weight"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_rolling_ic_portfolio_returns.csv", "rolling_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_ridge_ic_portfolio_returns.csv", "ridge_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_lasso_ic_portfolio_returns.csv", "lasso_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_elastic_net_ic_portfolio_returns.csv", "elastic_net_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_random_forest_ic_portfolio_returns.csv", "random_forest_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_xgboost_ic_portfolio_returns.csv", "xgboost_ic"),
    ]:
        frame = load_csv(file_name, parse_dates=["Date"])
        frame = frame[(frame["Date"] >= test_start) & (frame["Date"] <= test_end)].copy()
        frame = build_net_cumulative_wealth(frame)
        plt.plot(
            frame["Date"],
            frame["wealth"],
            linewidth=2.2,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )

    plt.title("Net Cumulative Portfolio Wealth (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative wealth of $1")
    plt.legend(frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.figtext(
        0.01,
        -0.02,
        f"Note: Portfolio wealth series are net of time-varying transaction costs and shown for the final test period only ({test_start_str} to {test_end_str}).",
        ha="left",
        fontsize=10,
    )
    plt.savefig(FIGURES_DIR / "figure_1_cumulative_portfolio_wealth.png")
    plt.close()


def make_figure_1_full_window_cumulative_wealth() -> None:
    summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv").iloc[0]
    common_start = pd.Timestamp(summary["common_window_start"])
    common_end = pd.Timestamp(summary["common_window_end"])
    train_start_str, train_end_str = [part.strip() for part in str(summary["train_period"]).split(" to ")]
    validation_start_str, validation_end_str = [part.strip() for part in str(summary["validation_period"]).split(" to ")]
    test_start_str, test_end_str = [part.strip() for part in str(summary["test_period"]).split(" to ")]
    train_start = pd.Timestamp(train_start_str)
    train_end = pd.Timestamp(train_end_str)
    validation_start = pd.Timestamp(validation_start_str)
    validation_end = pd.Timestamp(validation_end_str)
    test_start = pd.Timestamp(test_start_str)
    test_end = pd.Timestamp(test_end_str)

    plt.figure(figsize=(11.5, 6.8))
    ax = plt.gca()
    ax.axvspan(train_start, train_end, color="#dceaf7", alpha=0.35)
    ax.axvspan(validation_start, validation_end, color="#fef3c7", alpha=0.35)
    ax.axvspan(test_start, test_end, color="#fee2e2", alpha=0.35)

    for file_name, model_key in [
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_equal_weight_benchmark_portfolio_returns.csv", "equal_weight_benchmark"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_fixed_weight_portfolio_returns.csv", "fixed_weight"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_rolling_ic_portfolio_returns.csv", "rolling_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_ridge_ic_portfolio_returns.csv", "ridge_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_lasso_ic_portfolio_returns.csv", "lasso_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_elastic_net_ic_portfolio_returns.csv", "elastic_net_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_random_forest_ic_portfolio_returns.csv", "random_forest_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_xgboost_ic_portfolio_returns.csv", "xgboost_ic"),
    ]:
        frame = load_csv(file_name, parse_dates=["Date"])
        frame = frame[(frame["Date"] >= common_start) & (frame["Date"] <= common_end)].copy()
        frame = build_net_cumulative_wealth(frame)
        plt.plot(
            frame["Date"],
            frame["wealth"],
            linewidth=2.0,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )

    handles = [
        Patch(facecolor="#dceaf7", edgecolor="none", alpha=0.6, label="Training"),
        Patch(facecolor="#fef3c7", edgecolor="none", alpha=0.6, label="Validation"),
        Patch(facecolor="#fee2e2", edgecolor="none", alpha=0.6, label="Test"),
    ]
    line_handles, line_labels = ax.get_legend_handles_labels()
    ax.legend(line_handles + handles, line_labels + [h.get_label() for h in handles], frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.title("Net Cumulative Portfolio Wealth (Full Window)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative wealth of $1")
    plt.figtext(
        0.01,
        -0.02,
        "Note: Portfolio wealth series are net of time-varying transaction costs. Shaded regions mark the training, validation, and test windows used in common shrinkage selection.",
        ha="left",
        fontsize=10,
    )
    plt.savefig(FIGURES_DIR / "figure_1_full_window_cumulative_portfolio_wealth.png")
    plt.close()


def make_figure_a1_full_window_and_test_rebased() -> None:
    summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv").iloc[0]
    common_start = pd.Timestamp(summary["common_window_start"])
    common_end = pd.Timestamp(summary["common_window_end"])
    train_start_str, train_end_str = [part.strip() for part in str(summary["train_period"]).split(" to ")]
    validation_start_str, validation_end_str = [part.strip() for part in str(summary["validation_period"]).split(" to ")]
    test_start_str, test_end_str = [part.strip() for part in str(summary["test_period"]).split(" to ")]
    train_start = pd.Timestamp(train_start_str)
    train_end = pd.Timestamp(train_end_str)
    validation_start = pd.Timestamp(validation_start_str)
    validation_end = pd.Timestamp(validation_end_str)
    test_start = pd.Timestamp(test_start_str)
    test_end = pd.Timestamp(test_end_str)

    model_specs = [
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_equal_weight_benchmark_portfolio_returns.csv", "equal_weight_benchmark"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_fixed_weight_portfolio_returns.csv", "fixed_weight"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_rolling_ic_portfolio_returns.csv", "rolling_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_ridge_ic_portfolio_returns.csv", "ridge_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_lasso_ic_portfolio_returns.csv", "lasso_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_elastic_net_ic_portfolio_returns.csv", "elastic_net_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_random_forest_ic_portfolio_returns.csv", "random_forest_ic"),
        (f"common_shrinkage_{MAIN_HOLDING_BUFFER_FRAMEWORK}_xgboost_ic_portfolio_returns.csv", "xgboost_ic"),
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.4, 9.6),
        sharex=False,
        gridspec_kw={"height_ratios": [0.82, 1.18], "hspace": 0.20},
    )
    top_ax, bottom_ax = axes

    top_ax.axvspan(train_start, train_end, color="#dceaf7", alpha=0.35)
    top_ax.axvspan(validation_start, validation_end, color="#fef3c7", alpha=0.35)
    top_ax.axvspan(test_start, test_end, color="#fee2e2", alpha=0.35)

    for file_name, model_key in model_specs:
        frame = load_csv(file_name, parse_dates=["Date"])
        frame = frame[(frame["Date"] >= common_start) & (frame["Date"] <= common_end)].copy()
        frame = build_net_cumulative_wealth(frame)
        top_ax.plot(
            frame["Date"],
            frame["wealth"],
            linewidth=2.0,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )

        full_frame = load_csv(file_name, parse_dates=["Date"])
        full_frame = full_frame[(full_frame["Date"] >= common_start) & (full_frame["Date"] <= common_end)].copy()
        full_frame["wealth"] = (1.0 + full_frame["net_portfolio_return"].fillna(0.0)).cumprod()
        pre_test = full_frame[full_frame["Date"] < test_start].copy()
        test_frame = full_frame[(full_frame["Date"] >= test_start) & (full_frame["Date"] <= test_end)].copy()
        anchor = float(pre_test["wealth"].iloc[-1]) if not pre_test.empty else 1.0
        test_frame["rebased_wealth"] = test_frame["wealth"] / anchor
        bottom_ax.plot(
            test_frame["Date"],
            test_frame["rebased_wealth"],
            linewidth=2.2,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )

    top_ax.set_title("Panel A: Full common window")
    top_ax.set_xlabel("")
    top_ax.set_ylabel("Cumulative wealth of $1")
    top_ax.tick_params(axis="x", labelbottom=True)

    bottom_ax.set_title("Panel B: Same test segment, rebased at test start")
    bottom_ax.set_xlabel("Date")
    bottom_ax.set_ylabel("Rebased cumulative wealth")
    bottom_ax.set_xlim(test_start, test_end)

    for ax in (top_ax, bottom_ax):
        ax.grid(True, alpha=0.25, linewidth=0.8)
        ax.set_axisbelow(True)

    regime_handles = [
        Patch(facecolor="#dceaf7", edgecolor="none", alpha=0.6, label="Training"),
        Patch(facecolor="#fef3c7", edgecolor="none", alpha=0.6, label="Validation"),
        Patch(facecolor="#fee2e2", edgecolor="none", alpha=0.6, label="Test"),
    ]
    line_handles, line_labels = top_ax.get_legend_handles_labels()

    top_ymin, top_ymax = top_ax.get_ylim()
    rect = Rectangle(
        (mdates.date2num(test_start), top_ymin),
        mdates.date2num(test_end) - mdates.date2num(test_start),
        top_ymax - top_ymin,
        fill=False,
        edgecolor="#c1121f",
        linewidth=0.9,
        linestyle="-",
        alpha=0.4,
        transform=top_ax.transData,
    )
    top_ax.add_patch(rect)

    _, bottom_ymax = bottom_ax.get_ylim()
    for x in [test_start, test_end]:
        connector = ConnectionPatch(
            xyA=(mdates.date2num(x), top_ymin),
            coordsA=top_ax.transData,
            xyB=(mdates.date2num(x), bottom_ymax),
            coordsB=bottom_ax.transData,
            color="#9b2226",
            linewidth=0.8,
            alpha=0.3,
        )
        fig.add_artist(connector)

    fig.suptitle("Full-Window Wealth Paths and Rebased Test-Period Zoom", fontsize=16, y=0.975)
    fig.legend(
        line_handles + regime_handles,
        line_labels + [h.get_label() for h in regime_handles],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        frameon=False,
    )
    fig.text(
        0.02,
        0.012,
        "Notes: Panel A reports cumulative net wealth over the full common window. The red box marks the final test period.\n"
        "Panel B presents the same test segment rebased to 1 at the test start, allowing within-test performance to be compared without pre-test wealth effects.",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=1.000, top=0.92, bottom=0.18)
    plt.savefig(FIGURES_DIR / "figure_a1_full_window_and_test_rebased.png")
    plt.close()


def make_figure_2_rolling_weight_evolution() -> None:
    summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv").iloc[0]
    common_start = pd.Timestamp(summary["common_window_start"])
    common_end = pd.Timestamp(summary["common_window_end"])
    weights = load_csv(main_result_backtest_name("rolling_ic", "weight_history"), parse_dates=["Date"])
    weights = weights[(weights["Date"] >= common_start) & (weights["Date"] <= common_end)].copy()
    smoothed = weights.copy()
    weight_columns = list(FACTOR_LABELS.keys())
    display_labels = [FACTOR_LABELS[column] for column in weight_columns]
    color_map = ["#4E79A7", "#F28E2B", "#59A14F"]
    for column in weight_columns:
        smoothed[column] = smoothed[column].rolling(12, min_periods=12).mean()

    fig, ax = plt.subplots(figsize=(11.4, 5.0))
    ax.stackplot(
        smoothed["Date"],
        [smoothed[column] for column in weight_columns],
        labels=display_labels,
        colors=color_map,
        alpha=0.62,
        linewidth=0.0,
        edgecolor="none",
    )
    ax.set_title("Time-Varying Core Dimension Weights under the Rolling IC Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, alpha=0.18, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    fig.text(
        0.01,
        0.015,
        "Notes: The figure reports twelve-month rolling averages of monthly dimension weights for visual clarity. Monthly dimension weights sum to one by construction.",
        ha="left",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=0.74, top=0.88, bottom=0.17)
    plt.savefig(FIGURES_DIR / "figure_2_rolling_ic_weight_evolution.png")
    plt.close()


def make_figure_3_xgboost_concentration() -> None:
    summary = load_table_csv("table_sh4_common_shrinkage_selection_conclusion.csv").iloc[0]
    window_start = pd.Timestamp(summary["common_window_start"])
    window_end = pd.Timestamp(summary["common_window_end"])
    baseline_weight = float(summary["selected_baseline_weight"])
    model_weight = float(summary["selected_ic_weight"])
    weights = load_csv(main_result_backtest_name("xgboost_ic", "weight_history"), parse_dates=["Date"])
    weights = weights[(weights["Date"] >= window_start) & (weights["Date"] <= window_end)].copy()
    weight_columns = list(FACTOR_LABELS)
    weights["max_dimension_weight"] = weights[weight_columns].max(axis=1)
    mean_max = weights["max_dimension_weight"].mean()
    pct_gt_50 = (weights["max_dimension_weight"] > 0.50).mean()
    pct_gt_60 = (weights["max_dimension_weight"] > 0.60).mean()
    moderate_reference = 0.50
    theoretical_cap = model_weight + baseline_weight / 3.0

    plt.figure(figsize=(11, 6.5))
    plt.plot(weights["Date"], weights["max_dimension_weight"], color=MODEL_COLORS["xgboost_ic"], linewidth=2.2)
    plt.axhline(moderate_reference, color="#6c757d", linestyle="--", linewidth=1.2)
    plt.axhline(theoretical_cap, color="#c1121f", linestyle="--", linewidth=1.2)
    plt.text(
        weights["Date"].iloc[-1],
        moderate_reference + 0.005,
        "",
        va="bottom",
        ha="right",
        fontsize=10,
    )
    plt.text(
        weights["Date"].iloc[-1],
        theoretical_cap + 0.005,
        "",
        va="bottom",
        ha="right",
        fontsize=10,
    )
    annotation = (
        f"Mean maximum dimension weight: {mean_max:.2f}\n"
        f"Months with max weight > 0.50: {pct_gt_50 * 100:.1f}%\n"
        f"Months with max weight > 0.60: {pct_gt_60 * 100:.1f}%\n"
        f"Theoretical post-shrinkage cap: {theoretical_cap:.4f}"
    )
    plt.text(
        0.02,
        0.98,
        annotation,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    plt.title("XGBoost Dimension-Weight Concentration Over Time")
    plt.xlabel("Date")
    plt.ylabel("Maximum dimension weight")
    plt.ylim(0, 1)
    plt.savefig(FIGURES_DIR / "figure_3_xgboost_weight_concentration.png")
    plt.close()
    pd.DataFrame(
        [
            {
                "mean_maximum_factor_weight": mean_max,
                "share_of_months_max_weight_gt_0_50": pct_gt_50,
                "share_of_months_max_weight_gt_0_60": pct_gt_60,
            }
        ]
    ).pipe(rename_display_columns).to_csv(TABLES_DIR / "table_xgboost_weight_concentration_diagnostics.csv", index=False)


def make_figure_4_gross_vs_net_returns() -> None:
    main = load_table_csv("table_sh3_common_shrinkage_test_comparison.csv")
    ordered = [
        "equal_weight_benchmark",
        "fixed_weight",
        "rolling_ic",
        "ridge_ic",
        "lasso_ic",
        "elastic_net_ic",
        "random_forest_ic",
        "xgboost_ic",
    ]
    main["model"] = pd.Categorical(main["model"], categories=ordered, ordered=True)
    main = main.sort_values("model").copy()
    plot_frame = main[["model", "annualized_return", "net_return_after_costs"]].copy()
    plot_frame["model_code"] = plot_frame["model"].astype(str).map(MODEL_CODE_LABELS)
    plot_frame = plot_frame.melt(
        id_vars=["model", "model_code"],
        value_vars=["annualized_return", "net_return_after_costs"],
        var_name="return_type",
        value_name="value",
    )
    plot_frame["return_type"] = plot_frame["return_type"].map(
        {
            "annualized_return": "Gross annualised return",
            "net_return_after_costs": "Net annualised return",
        }
    )
    plot_frame["value_pct"] = plot_frame["value"] * 100.0
    plot_frame["model_code"] = plot_frame["model"].astype(str).map(MODEL_CODE_LABELS)

    plt.figure(figsize=(11, 6.5))
    sns.barplot(data=plot_frame, x="model_code", y="value_pct", hue="return_type", palette=["#4c78a8", "#f58518"])
    plt.title("Gross and Net Annualised Returns")
    plt.xlabel("Model")
    plt.ylabel("Annualised return (%)")
    plt.legend(
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    plt.xticks(rotation=0)
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(FIGURES_DIR / "figure_4_gross_versus_net_returns.png")
    plt.close()


def make_figure_6_turnover_framework_comparison() -> None:
    plot_frame = build_framework_comparison_test_window_frame()
    ordered_frameworks = ["baseline", "pta", MAIN_HOLDING_BUFFER_FRAMEWORK]
    plot_frame = plot_frame[
        plot_frame["model"].isin(MAIN_FRAMEWORK_COMPARISON_MODELS) & plot_frame["framework"].isin(ordered_frameworks)
    ].copy()
    if plot_frame["model"].eq("equal_weight_benchmark").any():
        raise ValueError("Figure 6 should exclude B0 from the execution-framework comparison.")
    plot_frame["model"] = pd.Categorical(
        plot_frame["model"], categories=MAIN_FRAMEWORK_COMPARISON_MODELS, ordered=True
    )
    plot_frame["framework"] = pd.Categorical(plot_frame["framework"], categories=ordered_frameworks, ordered=True)
    plot_frame = plot_frame.sort_values(["model", "framework"]).copy()
    ordered_model_codes = [MODEL_CODE_LABELS[m] for m in MAIN_FRAMEWORK_COMPARISON_MODELS]
    plot_frame["model_code"] = pd.Categorical(
        plot_frame["model"].astype(str).map(MODEL_CODE_LABELS),
        categories=ordered_model_codes,
        ordered=True,
    )
    plot_frame["framework_label"] = plot_frame["framework"].astype(str).map(FRAMEWORK_SHORT_LABELS)
    plot_frame["net_return_pct"] = plot_frame["net_return_after_costs"] * 100.0

    plt.figure(figsize=(12.5, 6.8))
    ax = sns.barplot(
        data=plot_frame,
        x="model_code",
        y="net_return_pct",
        hue="framework_label",
        palette=["#7f8c8d", "#c97b63", "#2a9d8f"],
    )
    plt.title("Net Annualised Returns by Turnover Control Framework")
    plt.xlabel("Model")
    plt.ylabel("Net annualised return (%)")
    plt.xticks(rotation=0)
    plt.legend(
        title="",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    plt.figtext(
        0.01,
        -0.055,
        "Note: Displayed model codes are S1, A1, L1, L2, L3, T1, and T2. Metrics are recomputed on the retained final test window under F1 (no control), F2 (PTA), and F3 (holding-buffer rule with a top-6 threshold).",
        ha="left",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(FIGURES_DIR / "figure_6_turnover_framework_comparison.png")
    plt.close()


def make_figure_7_holding_buffer_sensitivity() -> None:
    plot_frame = build_holding_buffer_sensitivity_test_window_frame()
    plot_frame = plot_frame[plot_frame["model"].isin(MAIN_FRAMEWORK_COMPARISON_MODELS)].copy()
    plot_frame["model"] = pd.Categorical(
        plot_frame["model"], categories=MAIN_FRAMEWORK_COMPARISON_MODELS, ordered=True
    )
    plot_frame = plot_frame.sort_values(["model", "holding_buffer_rank"]).copy()

    net_return_pivot = (
        plot_frame.assign(net_return_pct=lambda df: df["net_return_after_costs"] * 100.0)
        .pivot(index="model", columns="holding_buffer_rank", values="net_return_pct")
        .reindex(MAIN_FRAMEWORK_COMPARISON_MODELS)
    )
    turnover_pivot = (
        plot_frame.assign(turnover_pct=lambda df: df["turnover"] * 100.0)
        .pivot(index="model", columns="holding_buffer_rank", values="turnover_pct")
        .reindex(MAIN_FRAMEWORK_COMPARISON_MODELS)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.7), constrained_layout=False)
    sns.heatmap(
        net_return_pivot.rename(index=MODEL_SHORT_LABELS),
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        cbar_kws={"label": "Net annualised return (%)"},
        ax=axes[0],
    )
    axes[0].set_title("Net Return Sensitivity")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("")

    sns.heatmap(
        turnover_pivot.rename(index=MODEL_SHORT_LABELS),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Turnover (%)"},
        ax=axes[1],
    )
    axes[1].set_title("Turnover Sensitivity")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("")

    fig.suptitle("Holding-buffer Sensitivity Across Buffer Widths", fontsize=14)
    note = (
        "Note: Rows are S1, A1, L1, L2, L3, T1, and T2. Metrics are recomputed on the "
        "retained final test window. F3 denotes the holding-buffer rule. The retained "
        "execution rule is a top-3 portfolio with a 2-rank holding buffer, operationalised "
        "as a Top 6 threshold, while Top 4 and Top 5 are kept as sensitivity checks."
    )
    fig.text(
        0.5,
        0.02,
        "\n".join(textwrap.wrap(note, width=120)),
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.savefig(FIGURES_DIR / "figure_7_holding_buffer_sensitivity.png")
    plt.close()


def make_figure_5_turnover_vs_net_return_scatter() -> None:
    table = load_table_csv("table_sh3_common_shrinkage_test_comparison.csv")
    plot_frame = table[table["model"] != "equal_weight_benchmark"].copy()
    plot_frame["turnover_pct"] = plot_frame["turnover"] * 100.0
    plot_frame["net_return_pct"] = plot_frame["net_return_after_costs"] * 100.0
    plot_frame["model_code"] = plot_frame["model"].map(MODEL_CODE_LABELS)

    plt.figure(figsize=(8.8, 6.5))
    plt.scatter(plot_frame["turnover_pct"], plot_frame["net_return_pct"], color="#2a9d8f", s=85, alpha=0.9)
    for _, row in plot_frame.iterrows():
        plt.text(row["turnover_pct"] + 0.7, row["net_return_pct"] + 0.04, row["model_code"], fontsize=9)
    plt.title("Turnover and Net Return Trade-off")
    plt.xlabel("Turnover (%)")
    plt.ylabel("Net annualised return (%)")
    plt.figtext(
        0.01,
        -0.03,
        "Note: B0 is excluded because it holds all 9 ETFs and is not directly comparable to the top-3 selection models.",
        ha="left",
        fontsize=10,
    )
    plt.savefig(FIGURES_DIR / "figure_5_turnover_vs_net_return_tradeoff.png")
    plt.close()


def make_figure_8_gross_to_net_gap_by_model() -> None:
    table = load_table_csv("table_sh3_common_shrinkage_test_comparison.csv")
    plot_frame = table.copy()
    plot_frame["gap_pct"] = (plot_frame["annualized_return"] - plot_frame["net_return_after_costs"]) * 100.0
    plot_frame["model_code"] = plot_frame["model"].map(MODEL_CODE_LABELS)
    plot_frame = plot_frame.sort_values("gap_pct", ascending=False)

    plt.figure(figsize=(9.2, 6.3))
    ax = sns.barplot(data=plot_frame, x="model_code", y="gap_pct", color="#7c9a92")
    plt.title("Gross-to-Net Return Gap by Model")
    plt.xlabel("Model")
    plt.ylabel("Gross-to-net gap (percentage points)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    plt.savefig(FIGURES_DIR / "figure_8_gross_to_net_return_gap.png")
    plt.close()


def make_figure_9_framework_turnover_comparison() -> None:
    plot_frame = build_framework_comparison_test_window_frame()
    ordered_frameworks = ["baseline", "pta", MAIN_HOLDING_BUFFER_FRAMEWORK]
    plot_frame = plot_frame[
        plot_frame["model"].isin(MAIN_FRAMEWORK_COMPARISON_MODELS) & plot_frame["framework"].isin(ordered_frameworks)
    ].copy()
    if plot_frame["model"].eq("equal_weight_benchmark").any():
        raise ValueError("Figure 9 should exclude B0 from the execution-framework comparison.")
    ordered_model_codes = [MODEL_CODE_LABELS[m] for m in MAIN_FRAMEWORK_COMPARISON_MODELS]
    plot_frame["model_code"] = pd.Categorical(
        plot_frame["model"].astype(str).map(MODEL_CODE_LABELS),
        categories=ordered_model_codes,
        ordered=True,
    )
    plot_frame["framework_label"] = plot_frame["framework"].map(FRAMEWORK_SHORT_LABELS)
    plot_frame["turnover_pct"] = plot_frame["turnover"] * 100.0

    plt.figure(figsize=(12.5, 6.8))
    ax = sns.barplot(
        data=plot_frame,
        x="model_code",
        y="turnover_pct",
        hue="framework_label",
        palette=["#7f8c8d", "#c97b63", "#2a9d8f"],
    )
    plt.title("Turnover by Turnover-Control Framework", pad=12)
    plt.xlabel("Model")
    plt.ylabel("Turnover (%)")
    plt.legend(
        title="",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=8)
    plt.figtext(
        0.01,
        -0.03,
        "Note: Displayed model codes are S1, A1, L1, L2, L3, T1, and T2. Turnover is recomputed on the retained final test window under F1 (no control), F2 (PTA), and F3 (holding-buffer rule with a top-6 threshold).",
        ha="left",
        fontsize=10,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(FIGURES_DIR / "figure_9_framework_turnover_comparison.png")
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    save_table_1_model_specification()
    save_table_2_main_comparison()
    save_table_sh5_model_performance_by_vix_regime()
    save_table_4_turnover_framework_comparison()
    save_table_5_holding_buffer_sensitivity()
    save_table_d1_etf_summary_statistics()
    save_table_d2_model_family_entry_timing()
    save_table_dd1_sector_panel_characteristics()
    save_table_dd2_cross_sectional_opportunity_set()
    save_table_dd3_common_component_and_market_dependence()
    save_table_dd4_regime_dependence()
    save_table_dd4_regime_mean_difference_summary()
    save_table_dd5_persistence_and_implementation_frictions()
    save_table_dd6_transition_matrix()
    save_table_ld1_sector_liquidity_distribution()
    save_table_ld2_liquidity_by_vix_regime()
    save_table_ld3_liquidity_opportunity_commonality_comovement()
    save_table_m1_factor_ic_by_vix_regime()

    make_figure_1_cumulative_wealth()
    make_figure_1_full_window_cumulative_wealth()
    make_figure_a1_full_window_and_test_rebased()
    make_figure_2_rolling_weight_evolution()
    make_figure_3_xgboost_concentration()
    make_figure_4_gross_vs_net_returns()
    make_figure_5_turnover_vs_net_return_scatter()
    make_figure_6_turnover_framework_comparison()
    make_figure_7_holding_buffer_sensitivity()
    make_figure_8_gross_to_net_gap_by_model()
    make_figure_9_framework_turnover_comparison()
    make_figure_d1_etf_return_correlation_heatmap()
    make_figure_d2_factor_score_correlation_heatmap()
    make_figure_d3a_vix_level_over_time()
    make_figure_d3b_vix_stress_regime_distribution()
    make_figure_d4_etf_risk_return_scatter()
    make_figure_dd1_sector_return_distribution_boxplot()
    make_figure_dd2_cross_sectional_opportunity_timeseries()
    make_figure_dd3_market_dependence_bar()
    make_figure_dd4_regime_dependence_comparison()
    make_figure_dd5_transition_matrix_heatmap()
    make_figure_dd6_top3_retention_turnover_timeseries()
    make_figure_ld1_sector_dollar_volume_distribution()
    make_figure_ld2_sector_amihud_distribution()
    make_figure_ld3_liquidity_stress_over_time()
    make_figure_ld4_liquidity_regime_comparison()
    make_figure_ld5_liquidity_dispersion_commonality_scatter()
    make_figure_m1_factor_ic_distribution_boxplot()
    make_figure_m2_rolling_12m_factor_ic_heatmap()
    make_figure_m3_factor_ic_by_vix_regime_boxplot()
    apply_publication_table_formatting()

    print(f"Saved dissertation-ready figures to {FIGURES_DIR.relative_to(ROOT)}")
    print(f"Saved dissertation-ready tables to {TABLES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
