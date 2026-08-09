from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "logs" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_liquidity_diagnostics import (
    build_transaction_cost_linkage_frame_for_source,
    build_traded_leg_linkage_frame_for_source,
)


TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

MODEL_PALETTE = {
    "S1": "#4E79A7",
    "A1": "#F28E2B",
    "L1": "#59A14F",
    "L2": "#E15759",
    "L3": "#76B7B2",
    "T1": "#B07AA1",
    "T2": "#EDC948",
}


def safe_log10(series: pd.Series, floor: float = 1e-16) -> pd.Series:
    return np.log10(series.clip(lower=floor))


def prepare_frame() -> pd.DataFrame:
    frame = build_transaction_cost_linkage_frame_for_source("baseline").copy()
    frame["log10_avg_selected_winsorised_amihud"] = safe_log10(frame["avg_selected_winsorised_amihud"])
    frame["effective_cost_per_unit_turnover_bps"] = frame["effective_cost_per_unit_turnover"] * 10000.0
    frame["transaction_cost_rate_bps"] = frame["transaction_cost_rate"] * 10000.0
    return frame


def fit_ols_with_clustered_se(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    cluster_col: str,
) -> tuple[pd.Series, pd.Series, float, int, int]:
    sample = data[[cluster_col, y_col, *x_cols]].dropna().copy()
    y = sample[y_col].to_numpy(dtype=float)
    x = sample[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(sample)), x])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fitted = x @ beta
    resid = y - fitted
    n_obs, n_params = x.shape
    df_resid = n_obs - n_params

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    xtx_inv = np.linalg.pinv(x.T @ x)
    clusters = sample[cluster_col]
    unique_clusters = pd.Index(clusters.dropna().unique())
    n_clusters = len(unique_clusters)

    if n_clusters >= 2:
        meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
        for cluster_value in unique_clusters:
            cluster_mask = clusters.eq(cluster_value).to_numpy()
            x_g = x[cluster_mask]
            resid_g = resid[cluster_mask]
            score_g = x_g.T @ resid_g
            meat += np.outer(score_g, score_g)

        finite_sample_scale = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
        cov = finite_sample_scale * xtx_inv @ meat @ xtx_inv
        dof = n_clusters - 1
    else:
        sigma2 = rss / df_resid
        cov = sigma2 * xtx_inv
        dof = df_resid

    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se
    p_vals = 2.0 * (1.0 - student_t.cdf(np.abs(t_stats), df=dof))

    coef_index = ["Intercept", *x_cols]
    return (
        pd.Series(beta, index=coef_index),
        pd.Series(p_vals, index=coef_index),
        float(r2),
        int(n_obs),
        int(n_clusters),
    )


def build_regression_summary(frame: pd.DataFrame) -> pd.DataFrame:
    specs = [
        {
            "outcome": "Effective cost per unit turnover (bps)",
            "y_col": "effective_cost_per_unit_turnover_bps",
            "specification": "Log-linear",
            "x_cols": ["log10_avg_selected_winsorised_amihud"],
        },
        {
            "outcome": "Transaction cost rate (bps)",
            "y_col": "transaction_cost_rate_bps",
            "specification": "Log-linear",
            "x_cols": ["log10_avg_selected_winsorised_amihud"],
        },
        {
            "outcome": "Transaction cost rate (bps)",
            "y_col": "transaction_cost_rate_bps",
            "specification": "Log-linear + turnover control",
            "x_cols": ["log10_avg_selected_winsorised_amihud", "turnover"],
        },
    ]

    working = frame.copy()
    working["log10_avg_selected_winsorised_amihud_sq"] = (
        working["log10_avg_selected_winsorised_amihud"] ** 2
    )

    rows: list[dict[str, object]] = []
    for spec in specs:
        coefs, p_vals, r2, n_obs, n_clusters = fit_ols_with_clustered_se(
            working,
            spec["y_col"],
            spec["x_cols"],
            cluster_col="Date",
        )
        row = {
            "Outcome": spec["outcome"],
            "Specification": spec["specification"],
            "Observations": n_obs,
            "Month clusters": n_clusters,
            "R-squared": r2,
            "Intercept (bps)": coefs.get("Intercept", np.nan),
            "Intercept p-value": p_vals.get("Intercept", np.nan),
            "Log-Amihud coefficient": coefs.get("log10_avg_selected_winsorised_amihud", np.nan),
            "Log-Amihud p-value": p_vals.get("log10_avg_selected_winsorised_amihud", np.nan),
            "Turnover coefficient": coefs.get("turnover", np.nan),
            "Turnover p-value": p_vals.get("turnover", np.nan),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    for column in [
        "R-squared",
        "Intercept (bps)",
        "Log-Amihud coefficient",
        "Turnover coefficient",
    ]:
        summary[column] = summary[column].round(3)
    for column in [
        "Intercept p-value",
        "Log-Amihud p-value",
        "Turnover p-value",
    ]:
        summary[column] = summary[column].map(
            lambda value: "" if pd.isna(value) else ("<0.001" if value < 0.001 else f"{value:.3f}")
        )
    return summary


def build_quantile_summary(
    frame: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
    sample = frame.dropna(
        subset=[
            "log10_avg_selected_winsorised_amihud",
            "effective_cost_per_unit_turnover_bps",
        ]
    ).copy()
    sample["Log-Amihud quintile"] = pd.qcut(
        sample["log10_avg_selected_winsorised_amihud"],
        q=n_bins,
        labels=[f"Q{i}" for i in range(1, n_bins + 1)],
        duplicates="drop",
    )
    summary = (
        sample.groupby("Log-Amihud quintile", observed=False)
        .agg(
            observations=("effective_cost_per_unit_turnover_bps", "size"),
            mean_log10_amihud=("log10_avg_selected_winsorised_amihud", "mean"),
            mean_cost_per_turnover_bps=("effective_cost_per_unit_turnover_bps", "mean"),
            mean_transaction_cost_rate_bps=("transaction_cost_rate_bps", "mean"),
            mean_turnover=("turnover", "mean"),
            sd_cost_per_turnover_bps=("effective_cost_per_unit_turnover_bps", "std"),
        )
        .reset_index()
    )
    summary = summary.rename(
        columns={
            "observations": "Observations",
            "mean_log10_amihud": "Mean log10(Amihud)",
            "mean_cost_per_turnover_bps": "Mean effective cost per unit turnover (bps)",
            "mean_transaction_cost_rate_bps": "Mean transaction cost rate (bps)",
            "mean_turnover": "Mean turnover",
            "sd_cost_per_turnover_bps": "SD of effective cost per unit turnover (bps)",
        }
    )
    for column in [
        "Mean log10(Amihud)",
        "Mean effective cost per unit turnover (bps)",
        "Mean transaction cost rate (bps)",
        "Mean turnover",
        "SD of effective cost per unit turnover (bps)",
    ]:
        summary[column] = summary[column].round(3)
    return summary


def build_retained_traded_leg_summary() -> pd.DataFrame:
    frame = build_traded_leg_linkage_frame_for_source("main_result")
    rows: list[dict[str, object]] = []
    for model_name, sample in [("Pooled", frame)] + list(frame.groupby("model", sort=True)):
        sample = sample.dropna(subset=["effective_cost_per_unit_turnover"]).copy()
        rows.append(
            {
                "Model": model_name,
                "Observations": int(len(sample)),
                "Spearman corr with traded-leg Amihud": (
                    float(
                        sample["avg_traded_leg_winsorised_amihud"].corr(
                            sample["effective_cost_per_unit_turnover"],
                            method="spearman",
                        )
                    )
                    if not sample.empty
                    else np.nan
                ),
                "Spearman corr with traded-leg spread": (
                    float(
                        sample["avg_traded_leg_spread"].corr(
                            sample["effective_cost_per_unit_turnover"],
                            method="spearman",
                        )
                    )
                    if not sample.empty
                    else np.nan
                ),
                "Mean effective cost per turnover (bps)": (
                    float(sample["effective_cost_per_unit_turnover"].mean())
                    * 10000.0 if not sample.empty else np.nan
                ),
            }
        )
    summary = pd.DataFrame(rows)
    for column in [
        "Spearman corr with traded-leg Amihud",
        "Spearman corr with traded-leg spread",
        "Mean effective cost per turnover (bps)",
    ]:
        summary[column] = summary[column].round(3)
    return summary


def format_observations(series: pd.Series) -> pd.Series:
    return series.map(lambda value: f"{int(value):,}")


def build_appendix_ready_regression_table(regression_summary: pd.DataFrame) -> pd.DataFrame:
    keep_mask = (
        (
            regression_summary["Outcome"].eq("Effective cost per unit turnover (bps)")
            & regression_summary["Specification"].eq("Log-linear")
        )
        | (
            regression_summary["Outcome"].eq("Transaction cost rate (bps)")
            & regression_summary["Specification"].eq("Log-linear + turnover control")
        )
    )
    filtered = regression_summary.loc[keep_mask].copy()
    filtered = filtered[
        [
            "Outcome",
            "Specification",
            "Log-Amihud coefficient",
            "Log-Amihud p-value",
            "Turnover coefficient",
            "Turnover p-value",
            "Observations",
            "Month clusters",
            "R-squared",
        ]
    ]
    filtered["Observations"] = format_observations(filtered["Observations"])
    for column in ["Log-Amihud coefficient", "Turnover coefficient", "R-squared"]:
        filtered[column] = filtered[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
        )
    for column in ["Log-Amihud p-value", "Turnover p-value"]:
        filtered[column] = filtered[column].fillna("")
    filtered = filtered.reset_index(drop=True).fillna("")

    spec1 = filtered.iloc[0]
    spec2 = filtered.iloc[1]
    wide_rows = [
        {
            "Item": r"log10(average selected winsorised Amihud)",
            "(1) Effective cost per unit turnover (bps)": spec1["Log-Amihud coefficient"],
            "(2) Transaction cost rate (bps)": spec2["Log-Amihud coefficient"],
        },
        {
            "Item": "",
            "(1) Effective cost per unit turnover (bps)": f"({spec1['Log-Amihud p-value']})",
            "(2) Transaction cost rate (bps)": f"({spec2['Log-Amihud p-value']})",
        },
        {
            "Item": "Turnover",
            "(1) Effective cost per unit turnover (bps)": "-",
            "(2) Transaction cost rate (bps)": spec2["Turnover coefficient"],
        },
        {
            "Item": "",
            "(1) Effective cost per unit turnover (bps)": "",
            "(2) Transaction cost rate (bps)": f"({spec2['Turnover p-value']})",
        },
        {
            "Item": "Observations",
            "(1) Effective cost per unit turnover (bps)": spec1["Observations"],
            "(2) Transaction cost rate (bps)": spec2["Observations"],
        },
        {
            "Item": "Month clusters",
            "(1) Effective cost per unit turnover (bps)": f"{int(spec1['Month clusters'])}",
            "(2) Transaction cost rate (bps)": f"{int(spec2['Month clusters'])}",
        },
        {
            "Item": r"$R^2$",
            "(1) Effective cost per unit turnover (bps)": spec1["R-squared"],
            "(2) Transaction cost rate (bps)": spec2["R-squared"],
        },
    ]
    return pd.DataFrame(wide_rows)


def make_log_scatter(frame: pd.DataFrame) -> None:
    sample = frame.dropna(
        subset=[
            "log10_avg_selected_winsorised_amihud",
            "effective_cost_per_unit_turnover_bps",
        ]
    ).copy()
    coefs, _, _, _, _ = fit_ols_with_clustered_se(
        sample,
        "effective_cost_per_unit_turnover_bps",
        ["log10_avg_selected_winsorised_amihud"],
        cluster_col="Date",
    )

    x_grid = np.linspace(
        sample["log10_avg_selected_winsorised_amihud"].min(),
        sample["log10_avg_selected_winsorised_amihud"].max(),
        200,
    )
    y_grid = (
        coefs["Intercept"]
        + coefs["log10_avg_selected_winsorised_amihud"] * x_grid
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    sns.scatterplot(
        data=sample,
        x="log10_avg_selected_winsorised_amihud",
        y="effective_cost_per_unit_turnover_bps",
        hue="model",
        palette=MODEL_PALETTE,
        alpha=0.72,
        s=40,
        ax=ax,
    )
    ax.plot(
        x_grid,
        y_grid,
        color="#1D3557",
        linewidth=2.1,
        linestyle="--",
        label="Pooled log-linear fit",
    )
    ax.set_title("Baseline Log Amihud and Effective Cost per Unit Turnover")
    ax.set_xlabel("log10(Average selected winsorised Amihud)")
    ax.set_ylabel("Effective cost per unit turnover (bps)")
    ax.legend(frameon=True, loc="best", title="Model")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "figure_i1_log_amihud_vs_cost_per_turnover.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_quantile_plot(quantile_summary: pd.DataFrame) -> None:
    plot_df = quantile_summary.copy()
    plot_df["se_cost_per_turnover_bps"] = (
        plot_df["SD of effective cost per unit turnover (bps)"] / np.sqrt(plot_df["Observations"])
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    bars = ax.bar(
        plot_df["Log-Amihud quintile"],
        plot_df["Mean effective cost per unit turnover (bps)"],
        yerr=1.96 * plot_df["se_cost_per_turnover_bps"],
        color="#4E79A7",
        alpha=0.85,
        capsize=4,
    )
    error_heights = 1.96 * plot_df["se_cost_per_turnover_bps"]
    for bar, value, err in zip(
        bars,
        plot_df["Mean effective cost per unit turnover (bps)"],
        error_heights,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + err + 0.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Mean Effective Cost per Unit Turnover by Log-Amihud Quintile")
    ax.set_xlabel("Pooled log-Amihud quintile")
    ax.set_ylabel("Mean effective cost per unit turnover (bps)")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "figure_i2_log_amihud_quintile_cost_per_turnover.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    frame = prepare_frame()
    regression_summary = build_regression_summary(frame)
    quantile_summary = build_quantile_summary(frame)
    retained_summary = build_retained_traded_leg_summary()
    appendix_regression = build_appendix_ready_regression_table(regression_summary)

    regression_summary.to_csv(
        TABLES_DIR / "table_i1_liquidity_cost_exploratory_regressions.csv",
        index=False,
    )
    quantile_summary.to_csv(
        TABLES_DIR / "table_i2_log_amihud_cost_per_turnover_quintiles.csv",
        index=False,
    )
    retained_summary.to_csv(
        TABLES_DIR / "table_i3_retained_traded_leg_linkage_summary.csv",
        index=False,
    )
    appendix_regression.to_csv(
        TABLES_DIR / "table_i1_baseline_liquidity_cost_linkage_appendix.csv",
        index=False,
    )

    make_log_scatter(frame)
    make_quantile_plot(quantile_summary)

    print("Saved outputs/tables/table_i1_liquidity_cost_exploratory_regressions.csv")
    print("Saved outputs/tables/table_i2_log_amihud_cost_per_turnover_quintiles.csv")
    print("Saved outputs/tables/table_i3_retained_traded_leg_linkage_summary.csv")
    print("Saved outputs/tables/table_i1_baseline_liquidity_cost_linkage_appendix.csv")
    print("Saved outputs/figures/figure_i1_log_amihud_vs_cost_per_turnover.png")
    print("Saved outputs/figures/figure_i2_log_amihud_quintile_cost_per_turnover.png")


if __name__ == "__main__":
    main()
