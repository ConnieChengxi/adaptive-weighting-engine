from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "logs" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


BACKTEST_DIR = ROOT / "outputs" / "backtests"
FIGURES_DIR = ROOT / "outputs" / "figures"
TABLES_DIR = ROOT / "outputs" / "tables"
ARCHIVE_DIR = ROOT / "outputs" / "archive_legacy_assets"

MODEL_LABELS = {
    "fixed_weight": "Fixed-weight model",
    "rolling_ic_80_20": "Rolling IC model",
    "xgboost_ic": "XGBoost IC model",
    "rolling_ic_no_shrinkage": "Rolling IC (No shrinkage)",
    "rolling_ic_60_40": "Rolling IC (60/40 shrinkage)",
}

MODEL_COLORS = {
    "fixed_weight": "#264653",
    "rolling_ic_80_20": "#e76f51",
    "xgboost_ic": "#2a9d8f",
    "rolling_ic_no_shrinkage": "#457b9d",
    "rolling_ic_60_40": "#e9c46a",
}

FACTOR_LABELS = {
    "momentum_score_z_weight": "Momentum",
    "liquidity_1m_z_weight": "Liquidity",
    "downside_risk_score_z_weight": "Downside Risk",
    "volatility_score_z_weight": "Volatility",
}

PERCENT_COLUMNS = [
    "annualized_return",
    "annualized_volatility",
    "max_drawdown",
    "turnover",
    "net_return_after_costs",
]
ETF_TICKERS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLU", "XLP", "XLB"]


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


def format_percentage(series: pd.Series, decimals: int = 2) -> pd.Series:
    return (series * 100.0).map(lambda value: f"{value:.{decimals}f}%")


def format_metric_tables(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in PERCENT_COLUMNS:
        if column in formatted.columns:
            formatted[column] = format_percentage(formatted[column])
    if "sharpe_ratio" in formatted.columns:
        formatted["sharpe_ratio"] = formatted["sharpe_ratio"].map(lambda value: f"{value:.3f}")
    if "calmar_ratio" in formatted.columns:
        formatted["calmar_ratio"] = formatted["calmar_ratio"].map(lambda value: f"{value:.3f}")
    return formatted


def build_cumulative_wealth(df: pd.DataFrame) -> pd.DataFrame:
    wealth = df.copy()
    wealth["wealth"] = (1.0 + wealth["portfolio_return"].fillna(0.0)).cumprod()
    return wealth


def compute_drawdown_from_returns(return_series: pd.Series) -> float:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def first_adaptive_date(weights: pd.DataFrame, baseline_weights: list[float]) -> pd.Timestamp | None:
    cols = list(FACTOR_LABELS)
    mask = (weights[cols].round(10) != baseline_weights).any(axis=1)
    if not mask.any():
        return None
    return pd.to_datetime(weights.loc[mask, "Date"].iloc[0])


def save_table_1_model_specification() -> None:
    table = pd.DataFrame(
        [
            {
                "model_name": "Model 0",
                "weighting_logic": "Equal-weight benchmark across selected ETF universe",
                "adaptive_or_not": "No",
                "ml_or_not": "No",
                "shrinkage_rule": "Not applicable",
            },
            {
                "model_name": "Model 1",
                "weighting_logic": "Fixed multi-factor weights: Momentum 0.35, Liquidity 0.15, Downside Risk 0.25, Volatility 0.25",
                "adaptive_or_not": "No",
                "ml_or_not": "No",
                "shrinkage_rule": "Not applicable",
            },
            {
                "model_name": "Model 2",
                "weighting_logic": "Rolling Information Coefficient weighting",
                "adaptive_or_not": "Yes",
                "ml_or_not": "No",
                "shrinkage_rule": "80/20 shrinkage toward fixed-weight baseline",
            },
            {
                "model_name": "Model 3",
                "weighting_logic": "XGBoost-predicted factor IC weighting",
                "adaptive_or_not": "Yes",
                "ml_or_not": "Yes",
                "shrinkage_rule": "80/20 shrinkage toward fixed-weight baseline",
            },
        ]
    )
    table.to_csv(TABLES_DIR / "table_1_model_specification_summary.csv", index=False)


def save_table_2_main_comparison() -> None:
    table = load_csv("full_model_comparison_metrics.csv")
    table["model"] = table["model"].map(MODEL_LABELS)
    table = format_metric_tables(table)
    table.to_csv(TABLES_DIR / "table_2_main_model_performance_comparison.csv", index=False)


def save_table_3_rolling_robustness() -> None:
    table = load_csv("rolling_ic_robustness_comparison.csv")
    ordered = ["fixed_weight", "rolling_ic_no_shrinkage", "rolling_ic_80_20", "rolling_ic_60_40"]
    table["model"] = pd.Categorical(table["model"], categories=ordered, ordered=True)
    table = table.sort_values("model").copy()
    table["model"] = table["model"].astype(str).map(MODEL_LABELS)
    table = format_metric_tables(table)
    table.to_csv(TABLES_DIR / "table_3_rolling_ic_robustness_comparison.csv", index=False)


def save_table_4_transaction_cost_sensitivity() -> None:
    table = load_csv("xgboost_ic_transaction_cost_sensitivity.csv")
    table = table[table["transaction_cost_bps"].isin([0, 4, 5, 10, 25, 50])].copy()
    for column in [
        "fixed_weight_net_annualized_return",
        "xgboost_ic_net_annualized_return",
        "xgboost_ic_minus_fixed_weight",
    ]:
        table[column] = format_percentage(table[column])
    table.to_csv(TABLES_DIR / "table_4_transaction_cost_sensitivity_summary.csv", index=False)


def save_table_d1_etf_summary_statistics() -> None:
    monthly_panel = pd.read_csv(ROOT / "data" / "processed" / "monthly_factor_panel.csv", parse_dates=["Date"])
    records: list[dict[str, object]] = []

    for ticker in ETF_TICKERS:
        etf = monthly_panel.loc[monthly_panel["symbol"] == ticker, ["Date", "Close", "liquidity_1m"]].copy()
        etf = etf.sort_values("Date").reset_index(drop=True)
        etf["monthly_return"] = etf["Close"].pct_change()

        valid_returns = etf["monthly_return"].dropna()
        mean_monthly_return = float(valid_returns.mean()) if not valid_returns.empty else 0.0
        annualized_return = float((1.0 + valid_returns).prod() ** (12.0 / len(valid_returns)) - 1.0) if not valid_returns.empty else 0.0
        annualized_volatility = float(valid_returns.std(ddof=1) * (12.0**0.5)) if len(valid_returns) > 1 else 0.0
        max_drawdown = compute_drawdown_from_returns(valid_returns)
        if etf["liquidity_1m"].dropna().empty:
            avg_monthly_dollar_volume = None
        else:
            avg_monthly_dollar_volume = float(np.exp(etf["liquidity_1m"].dropna()).mean())

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
    table.to_csv(TABLES_DIR / "table_d1_etf_summary_statistics.csv", index=False)


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
    monthly_panel["momentum_score_z"] = monthly_panel[["momentum_3m_z", "momentum_6m_z"]].mean(axis=1, skipna=True)
    factor_frame = monthly_panel[
        [
            "momentum_score_z",
            "liquidity_1m_z",
            "downside_risk_score_z",
            "volatility_score_z",
        ]
    ].dropna()
    factor_frame = factor_frame.rename(
        columns={
            "momentum_score_z": "Momentum",
            "liquidity_1m_z": "Liquidity",
            "downside_risk_score_z": "Downside Risk",
            "volatility_score_z": "Volatility",
        }
    )
    corr = factor_frame.corr()

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Factor Score Correlation Matrix")
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



def make_figure_1_cumulative_wealth() -> None:
    plt.figure(figsize=(11, 6.5))
    for file_name, model_key in [
        ("fixed_weight_portfolio_returns.csv", "fixed_weight"),
        ("rolling_ic_portfolio_returns.csv", "rolling_ic_80_20"),
        ("xgboost_ic_portfolio_returns.csv", "xgboost_ic"),
    ]:
        frame = build_cumulative_wealth(load_csv(file_name, parse_dates=["Date"]))
        plt.plot(
            frame["Date"],
            frame["wealth"],
            linewidth=2.2,
            color=MODEL_COLORS[model_key],
            label=MODEL_LABELS[model_key],
        )

    plt.title("Cumulative Portfolio Wealth")
    plt.xlabel("Date")
    plt.ylabel("Cumulative wealth of $1")
    plt.legend(frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.figtext(0.01, -0.02, "Note: Portfolio wealth series are gross of transaction costs.", ha="left", fontsize=10)
    plt.savefig(FIGURES_DIR / "figure_1_cumulative_portfolio_wealth.png")
    plt.close()


def make_figure_2_rolling_weight_evolution() -> None:
    weights = load_csv("rolling_ic_weight_history.csv", parse_dates=["Date"])
    adaptive_date = first_adaptive_date(weights, [0.35, 0.15, 0.25, 0.25])
    smoothed = weights.copy()
    for column in FACTOR_LABELS:
        smoothed[column] = smoothed[column].rolling(3, min_periods=1).mean()

    plt.figure(figsize=(11, 6.5))
    for column, label in FACTOR_LABELS.items():
        plt.plot(smoothed["Date"], smoothed[column], linewidth=2.0, label=label)
    if adaptive_date is not None:
        plt.axvline(adaptive_date, color="black", linestyle="--", linewidth=1.2, label="Adaptive period begins")
    plt.title("Factor Weight Evolution: Rolling IC Model")
    plt.xlabel("Date")
    plt.ylabel("Factor weight")
    plt.ylim(0, 1)
    plt.legend(frameon=True, ncol=2)
    plt.figtext(0.01, -0.02, "Note: Weights are smoothed for visualization only.", ha="left", fontsize=10)
    plt.savefig(FIGURES_DIR / "figure_2_rolling_ic_weight_evolution.png")
    plt.close()


def make_figure_3_xgboost_concentration() -> None:
    weights = load_csv("xgboost_ic_weight_history.csv", parse_dates=["Date"])
    weight_columns = list(FACTOR_LABELS)
    weights["max_factor_weight"] = weights[weight_columns].max(axis=1)
    mean_max = weights["max_factor_weight"].mean()
    pct_gt_70 = (weights["max_factor_weight"] > 0.70).mean()
    pct_gt_80 = (weights["max_factor_weight"] > 0.80).mean()

    plt.figure(figsize=(11, 6.5))
    plt.plot(weights["Date"], weights["max_factor_weight"], color=MODEL_COLORS["xgboost_ic"], linewidth=2.2)
    plt.axhline(0.70, color="#6c757d", linestyle="--", linewidth=1.2)
    plt.axhline(0.80, color="#c1121f", linestyle="--", linewidth=1.2)
    plt.text(weights["Date"].iloc[-1], 0.705, "0.70", va="bottom", ha="right", fontsize=10)
    plt.text(weights["Date"].iloc[-1], 0.805, "0.80", va="bottom", ha="right", fontsize=10)
    annotation = (
        f"Mean maximum factor weight: {mean_max:.2f}\n"
        f"Months with max weight > 0.70: {pct_gt_70 * 100:.1f}%\n"
        f"Months with max weight > 0.80: {pct_gt_80 * 100:.1f}%"
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
    plt.title("XGBoost Weight Concentration Over Time")
    plt.xlabel("Date")
    plt.ylabel("Maximum factor weight")
    plt.ylim(0, 1)
    plt.savefig(FIGURES_DIR / "figure_3_xgboost_weight_concentration.png")
    plt.close()
    pd.DataFrame(
        [
            {
                "mean_maximum_factor_weight": mean_max,
                "share_of_months_max_weight_gt_0_70": pct_gt_70,
                "share_of_months_max_weight_gt_0_80": pct_gt_80,
            }
        ]
    ).to_csv(TABLES_DIR / "table_xgboost_weight_concentration_diagnostics.csv", index=False)


def make_figure_4_gross_vs_net_returns() -> None:
    main = load_csv("full_model_comparison_metrics.csv")
    plot_frame = main[["model", "annualized_return", "net_return_after_costs"]].copy()
    plot_frame["model"] = plot_frame["model"].map(MODEL_LABELS)
    plot_frame = plot_frame.melt(id_vars="model", var_name="return_type", value_name="value")
    plot_frame["return_type"] = plot_frame["return_type"].map(
        {
            "annualized_return": "Gross annualized return",
            "net_return_after_costs": "Net annualized return",
        }
    )
    plot_frame["value_pct"] = plot_frame["value"] * 100.0

    plt.figure(figsize=(11, 6.5))
    sns.barplot(data=plot_frame, x="model", y="value_pct", hue="return_type", palette=["#4c78a8", "#f58518"])
    plt.title("Gross and Net Annualized Returns")
    plt.xlabel("")
    plt.ylabel("Annualized return (%)")
    plt.legend(frameon=True)
    plt.xticks(rotation=0)
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
    plt.savefig(FIGURES_DIR / "figure_4_gross_versus_net_returns.png")
    plt.close()


def make_figure_5_transaction_cost_break_even() -> None:
    tc = load_csv("xgboost_ic_transaction_cost_sensitivity.csv")
    tc["difference_pct_points"] = tc["xgboost_ic_minus_fixed_weight"] * 100.0
    negative = tc[tc["xgboost_ic_minus_fixed_weight"] < 0]
    break_even_bps = int(negative["transaction_cost_bps"].iloc[0]) if not negative.empty else None
    break_even_value = None
    if break_even_bps is not None:
        break_even_value = tc.loc[tc["transaction_cost_bps"] == break_even_bps, "difference_pct_points"].iloc[0]
    zoom_tc = tc[tc["transaction_cost_bps"] <= 10].copy()

    plt.figure(figsize=(11, 6.5))
    plt.plot(zoom_tc["transaction_cost_bps"], zoom_tc["difference_pct_points"], color="#c1121f", linewidth=2.2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    if break_even_bps is not None:
        plt.scatter([break_even_bps], [break_even_value], color="#1d3557", zorder=5)
        plt.annotate(
            f"Break-even near {break_even_bps} bps",
            xy=(break_even_bps, break_even_value),
            xytext=(break_even_bps + 4, break_even_value + 0.15),
            arrowprops={"arrowstyle": "->", "lw": 1.0},
            fontsize=10,
        )
    plt.title("Transaction Cost Break-even: XGBoost vs Fixed-weight")
    plt.xlabel("Transaction cost in bps")
    plt.ylabel("XGBoost net return minus fixed-weight net return (percentage points)")
    plt.savefig(FIGURES_DIR / "figure_5_transaction_cost_break_even.png")
    plt.close()

    plt.figure(figsize=(11, 6.5))
    plt.plot(tc["transaction_cost_bps"], tc["difference_pct_points"], color="#c1121f", linewidth=2.2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    if break_even_bps is not None:
        plt.scatter([break_even_bps], [break_even_value], color="#1d3557", zorder=5)
        plt.annotate(
            f"Break-even near {break_even_bps} bps",
            xy=(break_even_bps, break_even_value),
            xytext=(break_even_bps + 6, break_even_value + 0.15),
            arrowprops={"arrowstyle": "->", "lw": 1.0},
            fontsize=10,
        )
    plt.title("Appendix Figure: Transaction Cost Break-even up to 50 bps")
    plt.xlabel("Transaction cost in bps")
    plt.ylabel("XGBoost net return minus fixed-weight net return (percentage points)")
    plt.savefig(FIGURES_DIR / "appendix_figure_transaction_cost_break_even_0_50bps.png")
    plt.close()


def make_appendix_xgboost_weight_history() -> None:
    weights = load_csv("xgboost_ic_weight_history.csv", parse_dates=["Date"])
    weight_columns = list(FACTOR_LABELS)

    plt.figure(figsize=(11, 6.5))
    plt.stackplot(
        weights["Date"],
        [weights[column] for column in weight_columns],
        labels=[FACTOR_LABELS[column] for column in weight_columns],
        alpha=0.9,
    )
    plt.title("Appendix Figure: XGBoost IC Factor Weight History")
    plt.xlabel("Date")
    plt.ylabel("Factor weight")
    plt.ylim(0, 1)
    plt.legend(frameon=True, ncol=2)
    plt.savefig(FIGURES_DIR / "appendix_figure_xgboost_weight_history.png")
    plt.close()


def save_report_summary() -> None:
    summary = """# Chapter 4 Figure and Table Package

Tables generated:
- Table 1: Model specification summary
- Table 2: Main model performance comparison
- Table 3: Rolling IC robustness comparison
- Table 4: Transaction cost sensitivity summary

Figures generated:
- Figure 1: Cumulative portfolio wealth by weighting model
- Figure 2: Factor weight evolution under the Rolling IC model
- Figure 3: Factor weight concentration under the XGBoost IC model
- Figure 4: Gross versus net annualized returns
- Figure 5: Transaction cost break-even analysis

Appendix:
- XGBoost raw stacked factor weight history
- XGBoost transaction cost break-even up to 50 bps
- VIX level with 75th percentile threshold
- VIX stress regime distribution
"""
    (TABLES_DIR / "chapter4_key_findings.md").write_text(summary, encoding="utf-8")


def archive_legacy_outputs() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    legacy_files = [
        FIGURES_DIR / "fig_main_cumulative_wealth.png",
        FIGURES_DIR / "fig_rolling_ic_80_20_weights.png",
        FIGURES_DIR / "fig_rolling_ic_robustness.png",
        FIGURES_DIR / "fig_xgboost_ic_weights.png",
        FIGURES_DIR / "fig_xgboost_transaction_cost_sensitivity.png",
        TABLES_DIR / "table_main_model_comparison.csv",
        TABLES_DIR / "table_rolling_ic_robustness.csv",
        TABLES_DIR / "table_xgboost_transaction_cost_threshold.csv",
        FIGURES_DIR / "figure_d3_vix_regime_distribution.png",
    ]
    for path in legacy_files:
        if path.exists():
            path.replace(ARCHIVE_DIR / path.name)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    archive_legacy_outputs()
    setup_plot_style()

    save_table_1_model_specification()
    save_table_2_main_comparison()
    save_table_3_rolling_robustness()
    save_table_4_transaction_cost_sensitivity()
    save_table_d1_etf_summary_statistics()

    make_figure_1_cumulative_wealth()
    make_figure_2_rolling_weight_evolution()
    make_figure_3_xgboost_concentration()
    make_figure_4_gross_vs_net_returns()
    make_figure_5_transaction_cost_break_even()
    make_figure_d1_etf_return_correlation_heatmap()
    make_figure_d2_factor_score_correlation_heatmap()
    make_figure_d3a_vix_level_over_time()
    make_figure_d3b_vix_stress_regime_distribution()
    make_appendix_xgboost_weight_history()
    save_report_summary()

    print(f"Saved dissertation-ready figures to {FIGURES_DIR.relative_to(ROOT)}")
    print(f"Saved dissertation-ready tables to {TABLES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
