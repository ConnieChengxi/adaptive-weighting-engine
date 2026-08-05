from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "logs" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BACKTEST_DIR = ROOT / "outputs" / "backtests"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
MAIN_FRAMEWORK = "holding_buffer_top6"
MAIN_RESULT_PREFIX = f"common_shrinkage_{MAIN_FRAMEWORK}"

MAIN_MODELS = ["fixed_weight", "rolling_ic", "ridge_ic", "lasso_ic", "elastic_net_ic", "random_forest_ic", "xgboost_ic"]
MODEL_LABELS = {
    "fixed_weight": "S1",
    "rolling_ic": "A1",
    "ridge_ic": "L1",
    "lasso_ic": "L2",
    "elastic_net_ic": "L3",
    "random_forest_ic": "T1",
    "xgboost_ic": "T2",
}
MODEL_ORDER = ["S1", "A1", "L1", "L2", "L3", "T1", "T2"]
FACTOR_COLUMNS = {
    "Market-adjusted relative performance": ("momentum_score_z", "momentum_score_z_weight"),
    "Implementation friction": ("liquidity_1m_z", "liquidity_1m_z_weight"),
    "Sector-specific uncertainty": ("volatility_score_z", "volatility_score_z_weight"),
}
STATIC_WEIGHTS = {
    "momentum_score_z_weight": 1.0 / 3.0,
    "liquidity_1m_z_weight": 1.0 / 3.0,
    "volatility_score_z_weight": 1.0 / 3.0,
}
FACTOR_COLORS = {
    "Market-adjusted relative performance": "#4c78a8",
    "Implementation friction": "#72b7b2",
    "Sector-specific uncertainty": "#e45756",
}
TIED_COLOR = "#9aa0a6"


def setup_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10


def main_result_backtest_name(model_name: str, artifact: str) -> str:
    return f"{MAIN_RESULT_PREFIX}_{model_name}_{artifact}.csv"


def load_selection_frame(model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(BACKTEST_DIR / main_result_backtest_name(model_name, "selections"), parse_dates=["Date"])
    if model_name == "fixed_weight":
        for weight_column, value in STATIC_WEIGHTS.items():
            frame[weight_column] = value
    return frame


def load_weight_history_frame(model_name: str) -> pd.DataFrame:
    if model_name == "fixed_weight":
        dates = pd.read_csv(BACKTEST_DIR / main_result_backtest_name("fixed_weight", "selections"), usecols=["Date"]).drop_duplicates()
        for weight_column, value in STATIC_WEIGHTS.items():
            dates[weight_column] = value
        return dates
    return pd.read_csv(BACKTEST_DIR / main_result_backtest_name(model_name, "weight_history"), parse_dates=["Date"])


def build_score_contribution_long_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MAIN_MODELS:
        selection = load_selection_frame(model_name).copy()
        for factor_name, (score_column, weight_column) in FACTOR_COLUMNS.items():
            selection[f"{factor_name}_contribution"] = (
                selection[score_column] * selection[weight_column] * selection["portfolio_weight"]
            )

        contribution_columns = [f"{factor_name}_contribution" for factor_name in FACTOR_COLUMNS]
        monthly = (
            selection.groupby("Date", as_index=False)[contribution_columns]
            .sum()
            .assign(model=MODEL_LABELS[model_name])
        )
        long_monthly = monthly.melt(id_vars=["Date", "model"], var_name="factor", value_name="portfolio_score_contribution")
        long_monthly["factor"] = long_monthly["factor"].str.replace("_contribution", "", regex=False)
        rows.append(long_monthly)
    return pd.concat(rows, ignore_index=True)


def build_score_contribution_summary(score_long: pd.DataFrame) -> pd.DataFrame:
    score_long = score_long.copy()
    score_long["absolute_contribution"] = score_long["portfolio_score_contribution"].abs()
    totals = (
        score_long.groupby(["model", "Date"], as_index=False)["absolute_contribution"]
        .sum()
        .rename(columns={"absolute_contribution": "total_absolute_contribution"})
    )
    score_long = score_long.merge(totals, on=["model", "Date"], how="left")
    score_long["absolute_share"] = score_long["absolute_contribution"] / score_long["total_absolute_contribution"].where(
        score_long["total_absolute_contribution"] > 0
    )

    summary = (
        score_long.groupby(["model", "factor"], as_index=False)
        .agg(
            mean_portfolio_score_contribution=("portfolio_score_contribution", "mean"),
            mean_absolute_score_contribution=("absolute_contribution", "mean"),
            share_of_positive_months=("portfolio_score_contribution", lambda s: (s > 0).mean()),
            mean_share_of_absolute_score=("absolute_share", "mean"),
        )
    )
    return summary


def build_weight_contribution_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dynamic_models = [model for model in MAIN_MODELS if model != "fixed_weight"]
    for model_name in dynamic_models:
        weight_history = load_weight_history_frame(model_name).copy()
        weight_columns = [weight_column for _, weight_column in FACTOR_COLUMNS.values()]
        weight_frame = weight_history[weight_columns]
        row_max = weight_frame.max(axis=1)
        is_max = weight_frame.eq(row_max, axis=0)
        unique_largest = is_max.sum(axis=1) == 1
        for factor_name, (_, weight_column) in FACTOR_COLUMNS.items():
            factor_weights = weight_history[weight_column]
            rows.append(
                {
                    "model": MODEL_LABELS[model_name],
                    "factor": factor_name,
                    "mean_weight": factor_weights.mean(),
                    "weight_std_dev": factor_weights.std(ddof=1),
                    "minimum_weight": factor_weights.min(),
                    "maximum_weight": factor_weights.max(),
                    "share_of_months_as_unique_largest_weight": (unique_largest & is_max[weight_column]).mean(),
                }
            )
    return pd.DataFrame(rows)


def build_weight_dominance_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dynamic_models = [model for model in MAIN_MODELS if model != "fixed_weight"]
    for model_name in dynamic_models:
        weight_history = load_weight_history_frame(model_name).copy()
        weight_columns = [weight_column for _, weight_column in FACTOR_COLUMNS.values()]
        weight_frame = weight_history[weight_columns]
        row_max = weight_frame.max(axis=1)
        is_max = weight_frame.eq(row_max, axis=0)
        unique_largest = is_max.sum(axis=1) == 1

        for factor_name, (_, weight_column) in FACTOR_COLUMNS.items():
            rows.append(
                {
                    "model": MODEL_LABELS[model_name],
                    "factor": factor_name,
                    "share_of_months": (unique_largest & is_max[weight_column]).mean(),
                }
            )

        rows.append(
            {
                "model": MODEL_LABELS[model_name],
                "factor": "Tied maximum",
                "share_of_months": (~unique_largest).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_hhi_concentration_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dynamic_models = [model for model in MAIN_MODELS if model != "fixed_weight"]
    weight_columns = [weight_column for _, weight_column in FACTOR_COLUMNS.values()]
    for model_name in dynamic_models:
        weight_history = load_weight_history_frame(model_name).copy()
        weight_history["hhi"] = weight_history[weight_columns].pow(2).sum(axis=1)
        weight_history["effective_number_of_factors"] = 1.0 / weight_history["hhi"]
        rows.append(
            {
                "model": MODEL_LABELS[model_name],
                "mean_hhi": weight_history["hhi"].mean(),
                "median_hhi": weight_history["hhi"].median(),
                "hhi_std_dev": weight_history["hhi"].std(ddof=1),
                "minimum_hhi": weight_history["hhi"].min(),
                "maximum_hhi": weight_history["hhi"].max(),
                "mean_effective_number_of_dimensions": weight_history["effective_number_of_factors"].mean(),
            }
        )
    return pd.DataFrame(rows)


def save_tables(score_summary: pd.DataFrame, weight_summary: pd.DataFrame, hhi_summary: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    score_table = score_summary.copy()
    for column in [
        "mean_portfolio_score_contribution",
        "mean_absolute_score_contribution",
        "mean_share_of_absolute_score",
        "share_of_positive_months",
    ]:
        if "share" in column:
            score_table[column] = (score_table[column] * 100.0).map(lambda x: f"{x:.2f}%")
        else:
            score_table[column] = score_table[column].map(lambda x: f"{x:.4f}")
    score_table = score_table.rename(
        columns={
            "model": "Model",
            "factor": "Dimension",
            "mean_portfolio_score_contribution": "Mean portfolio-weighted score contribution",
            "mean_absolute_score_contribution": "Mean absolute portfolio-weighted score contribution",
            "share_of_positive_months": "Share of positive-contribution months",
            "mean_share_of_absolute_score": "Mean share of absolute portfolio-weighted score contribution",
        }
    )
    score_table.to_csv(TABLES_DIR / "table_fc1_score_contribution_summary.csv", index=False)

    weight_table = weight_summary.copy()
    for column in [
        "mean_weight",
        "weight_std_dev",
        "share_of_months_as_unique_largest_weight",
    ]:
        if "share" in column:
            weight_table[column] = (weight_table[column] * 100.0).map(lambda x: f"{x:.2f}%")
        else:
            weight_table[column] = weight_table[column].map(lambda x: f"{x:.4f}")
    weight_table = weight_table.rename(
        columns={
            "model": "Model",
            "factor": "Dimension",
            "mean_weight": "Mean weight",
            "weight_std_dev": "Weight standard deviation",
            "share_of_months_as_unique_largest_weight": "Share of months as unique largest weight",
        }
    )
    weight_table = weight_table[
        [
            "Model",
            "Dimension",
            "Mean weight",
            "Weight standard deviation",
            "Share of months as unique largest weight",
        ]
    ]
    weight_table.to_csv(TABLES_DIR / "table_fc2_weight_contribution_summary.csv", index=False)

    hhi_table = hhi_summary.copy()
    for column in [
        "mean_hhi",
        "median_hhi",
        "hhi_std_dev",
        "mean_effective_number_of_dimensions",
    ]:
        hhi_table[column] = hhi_table[column].map(lambda x: f"{x:.4f}")
    hhi_table = hhi_table.rename(
        columns={
            "model": "Model",
            "mean_hhi": "Mean HHI",
            "median_hhi": "Median HHI",
            "hhi_std_dev": "HHI standard deviation",
            "mean_effective_number_of_dimensions": "Mean effective number of dimensions",
        }
    )
    hhi_table = hhi_table[
        [
            "Model",
            "Mean HHI",
            "Median HHI",
            "HHI standard deviation",
            "Mean effective number of dimensions",
        ]
    ]
    hhi_table.to_csv(TABLES_DIR / "table_fc3_hhi_concentration_summary.csv", index=False)


def make_score_contribution_figure(score_summary: pd.DataFrame) -> None:
    plot_frame = score_summary.copy()
    plot_frame["mean_share_pct"] = plot_frame["mean_share_of_absolute_score"] * 100.0
    plot_frame["model"] = pd.Categorical(plot_frame["model"], categories=MODEL_ORDER, ordered=True)
    plot_frame = plot_frame.sort_values(["model", "factor"])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=plot_frame,
        x="model",
        y="mean_share_pct",
        hue="factor",
        palette=FACTOR_COLORS,
        ax=ax,
    )
    ax.set_title("Dimension Portfolio-Weighted Score Contribution by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean share of absolute portfolio-weighted score contribution (%)")
    ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES_DIR / "figure_fc1_score_contribution_by_model.png")
    plt.close(fig)


def make_weight_contribution_figure(dominance_summary: pd.DataFrame) -> None:
    plot_frame = dominance_summary.copy()
    plot_frame["largest_share_pct"] = plot_frame["share_of_months"] * 100.0
    plot_frame["model"] = pd.Categorical(plot_frame["model"], categories=MODEL_ORDER[1:], ordered=True)
    plot_frame = plot_frame.sort_values(["model", "factor"])
    palette = {
        **FACTOR_COLORS,
        "Tied maximum": TIED_COLOR,
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sns.barplot(
        data=plot_frame,
        x="model",
        y="largest_share_pct",
        hue="factor",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Unique Largest-Weight Frequency Across Dynamic Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Share of months as unique largest dimension weight or tied maximum (%)")
    ax.legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=4)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES_DIR / "figure_fc2_weight_contribution_by_model.png")
    plt.close(fig)


def main() -> None:
    setup_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    score_long = build_score_contribution_long_frame()
    score_summary = build_score_contribution_summary(score_long)
    weight_summary = build_weight_contribution_summary()
    dominance_summary = build_weight_dominance_summary()
    hhi_summary = build_hhi_concentration_summary()

    save_tables(score_summary, weight_summary, hhi_summary)
    make_score_contribution_figure(score_summary)
    make_weight_contribution_figure(dominance_summary)

    print("Saved outputs/tables/table_fc1_score_contribution_summary.csv")
    print("Saved outputs/tables/table_fc2_weight_contribution_summary.csv")
    print("Saved outputs/tables/table_fc3_hhi_concentration_summary.csv")
    print("Saved outputs/figures/figure_fc1_score_contribution_by_model.png")
    print("Saved outputs/figures/figure_fc2_weight_contribution_by_model.png")


if __name__ == "__main__":
    main()
