from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "logs" / "mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_liquidity_diagnostics import (  # noqa: E402
    APPENDIX_I_MODEL_ORDER,
    build_transaction_cost_linkage_frame_for_source,
    build_traded_leg_linkage_frame_for_source,
)


TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

TABLE_I1_PATH = TABLES_DIR / "table_i1_execution_setting_liquidity_linkage.csv"
TABLE_I2_PATH = TABLES_DIR / "table_i2_traded_leg_model_summary.csv"
FIGURE_I1_PATH = FIGURES_DIR / "figure_i1_amihud_vs_effective_cost_by_setting.png"
FIGURE_I2_PATH = FIGURES_DIR / "figure_i2_amihud_quintile_unit_cost_by_setting.png"

LEGACY_OUTPUTS = [
    TABLES_DIR / "table_i1_liquidity_cost_exploratory_regressions.csv",
    TABLES_DIR / "table_i2_log_amihud_cost_per_turnover_quintiles.csv",
    TABLES_DIR / "table_i3_retained_traded_leg_linkage_summary.csv",
    TABLES_DIR / "table_i1_baseline_liquidity_cost_linkage_appendix.csv",
    FIGURES_DIR / "figure_i1_log_amihud_vs_cost_per_turnover.png",
    FIGURES_DIR / "figure_i2_log_amihud_quintile_cost_per_turnover.png",
]

SETTING_LABELS = {
    "F1": "Average selected Amihud",
    "F3": "Average traded-leg Amihud",
}

SETTING_PANEL_TITLES = {
    "F1": "Panel A. F1 selected-set Amihud",
    "F3": "Panel B. F3 traded-leg Amihud",
}

SETTING_COLORS = {
    "F1": "#4E79A7",
    "F3": "#E15759",
}

MODEL_ORDER_WITH_POOLED = ["Pooled", *APPENDIX_I_MODEL_ORDER]


def fit_ols_with_clustered_se(
    data: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    cluster_col: str,
) -> tuple[pd.Series, pd.Series, pd.Series, float, int, int]:
    sample = data[[cluster_col, y_col, *x_cols]].dropna().copy()
    if sample.empty:
        raise ValueError(f"No valid rows for clustered regression on {y_col}")

    y = sample[y_col].to_numpy(dtype=float)
    x = sample[x_cols].to_numpy(dtype=float) if x_cols else np.empty((len(sample), 0))
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
        pd.Series(se, index=coef_index),
        pd.Series(p_vals, index=coef_index),
        float(r2),
        int(n_obs),
        int(n_clusters),
    )


def clustered_mean_ci(
    data: pd.DataFrame,
    value_col: str,
    cluster_col: str,
    confidence: float = 0.95,
) -> tuple[float, float, float, int, int]:
    coefs, ses, _, _, n_obs, n_clusters = fit_ols_with_clustered_se(
        data,
        y_col=value_col,
        x_cols=[],
        cluster_col=cluster_col,
    )
    mean_value = float(coefs["Intercept"])
    se_value = float(ses["Intercept"])
    dof = max(n_clusters - 1, 1)
    t_crit = float(student_t.ppf((1.0 + confidence) / 2.0, df=dof))
    lower = mean_value - t_crit * se_value
    upper = mean_value + t_crit * se_value
    return mean_value, lower, upper, n_obs, n_clusters


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return ""
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def clean_legacy_outputs() -> None:
    for path in LEGACY_OUTPUTS:
        if path.exists():
            path.unlink()


def build_appendix_i_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    f1 = build_transaction_cost_linkage_frame_for_source("baseline").copy()
    f3 = build_traded_leg_linkage_frame_for_source("main_result").copy()

    f1 = f1.loc[f1["turnover"] > 0].copy()
    f3 = f3.loc[f3["turnover"] > 0].copy()

    required_cols = [
        "Date",
        "model",
        "execution_setting",
        "amihud",
        "log10_amihud",
        "turnover",
        "transaction_cost_rate",
        "effective_cost_bps",
    ]
    for name, frame in [("F1", f1), ("F3", f3)]:
        missing = [col for col in required_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"{name} frame is missing required columns: {missing}")
        if frame[required_cols].drop(columns=["execution_setting"]).isna().any().any():
            na_counts = frame[required_cols].drop(columns=["execution_setting"]).isna().sum()
            raise ValueError(f"{name} frame contains missing values: {na_counts[na_counts > 0].to_dict()}")

    return f1, f3


def build_table_i1(f1: pd.DataFrame, f3: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for setting_code, frame in [("F1", f1), ("F3", f3)]:
        sample = frame.dropna(subset=["log10_amihud", "effective_cost_bps"]).copy()
        spearman_rho = float(spearmanr(sample["amihud"], sample["effective_cost_bps"]).statistic)
        coefs, _, p_vals, r2, n_obs, n_clusters = fit_ols_with_clustered_se(
            sample,
            y_col="effective_cost_bps",
            x_cols=["log10_amihud"],
            cluster_col="Date",
        )
        rows.append(
            {
                "Setting": setting_code,
                "Predictor": SETTING_LABELS[setting_code],
                "N": n_obs,
                "Months": n_clusters,
                "Spearman rho": round(spearman_rho, 3),
                "Beta": round(float(coefs["log10_amihud"]), 3),
                "Clustered p": format_p_value(float(p_vals["log10_amihud"])),
                "R-squared": round(r2, 3),
            }
        )
    return pd.DataFrame(rows)


def build_table_i2(f3: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = [("Pooled", f3)] + [(model, grp.copy()) for model, grp in f3.groupby("model", sort=False)]
    for model_name, frame in grouped:
        sample = frame.dropna(subset=["amihud", "effective_cost_bps"]).copy()
        rho = float(spearmanr(sample["amihud"], sample["effective_cost_bps"]).statistic) if not sample.empty else np.nan
        rows.append(
            {
                "Model": model_name,
                "N": int(len(sample)),
                "Spearman rho": round(rho, 3) if pd.notna(rho) else np.nan,
                "Mean unit cost": round(float(sample["effective_cost_bps"].mean()), 3) if not sample.empty else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["Model"] = pd.Categorical(table["Model"], categories=MODEL_ORDER_WITH_POOLED, ordered=True)
    table = table.sort_values("Model").reset_index(drop=True)
    table["Model"] = table["Model"].astype(str)
    return table


def make_figure_i1(f1: pd.DataFrame, f3: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)

    for ax, (setting_code, frame) in zip(axes, [("F1", f1), ("F3", f3)]):
        sample = frame.dropna(subset=["log10_amihud", "effective_cost_bps"]).copy()
        color = SETTING_COLORS[setting_code]
        ax.scatter(
            sample["log10_amihud"],
            sample["effective_cost_bps"],
            alpha=0.55,
            s=24,
            color=color,
            edgecolor="none",
        )

        coefs, _, p_vals, _, n_obs, n_clusters = fit_ols_with_clustered_se(
            sample,
            y_col="effective_cost_bps",
            x_cols=["log10_amihud"],
            cluster_col="Date",
        )
        x_vals = np.linspace(sample["log10_amihud"].min(), sample["log10_amihud"].max(), 100)
        y_vals = float(coefs["Intercept"]) + float(coefs["log10_amihud"]) * x_vals
        ax.plot(x_vals, y_vals, color="#222222", linewidth=2.0)

        stats_text = (
            f"N: {n_obs}\n"
            f"Months: {n_clusters}\n"
            f"β: {float(coefs['log10_amihud']):.3f}\n"
            f"p: {format_p_value(float(p_vals['log10_amihud']))}"
        )
        ax.text(
            0.03,
            0.97,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
        )
        ax.set_title(SETTING_PANEL_TITLES[setting_code], fontsize=12)
        ax.set_xlabel("log10(Amihud)")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Effective cost (bps per unit turnover)")
    fig.tight_layout()
    fig.savefig(FIGURE_I1_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_quintile_summary(frame: pd.DataFrame, setting_code: str) -> pd.DataFrame:
    sample = frame.dropna(subset=["log10_amihud", "effective_cost_bps"]).copy()
    sample["quintile"] = pd.qcut(
        sample["log10_amihud"],
        q=5,
        labels=[f"Q{i}" for i in range(1, 6)],
        duplicates="drop",
    )

    rows: list[dict[str, object]] = []
    for quintile, group in sample.groupby("quintile", observed=False):
        if group.empty:
            continue
        mean_value, lower, upper, n_obs, n_clusters = clustered_mean_ci(
            group,
            value_col="effective_cost_bps",
            cluster_col="Date",
        )
        rows.append(
            {
                "execution_setting": setting_code,
                "quintile": str(quintile),
                "mean_unit_cost": mean_value,
                "ci_lower": lower,
                "ci_upper": upper,
                "N": n_obs,
                "Months": n_clusters,
            }
        )
    return pd.DataFrame(rows)


def make_figure_i2(f1: pd.DataFrame, f3: pd.DataFrame) -> None:
    summaries = pd.concat(
        [
            build_quintile_summary(f1, "F1"),
            build_quintile_summary(f3, "F3"),
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for ax, setting_code in zip(axes, ["F1", "F3"]):
        plot_df = summaries.loc[summaries["execution_setting"] == setting_code].copy()
        plot_df["quintile"] = pd.Categorical(plot_df["quintile"], categories=[f"Q{i}" for i in range(1, 6)], ordered=True)
        plot_df = plot_df.sort_values("quintile")
        color = SETTING_COLORS[setting_code]

        yerr = np.vstack(
            [
                plot_df["mean_unit_cost"] - plot_df["ci_lower"],
                plot_df["ci_upper"] - plot_df["mean_unit_cost"],
            ]
        )
        ax.bar(
            plot_df["quintile"].astype(str),
            plot_df["mean_unit_cost"],
            color=color,
            alpha=0.85,
            yerr=yerr,
            capsize=4,
            ecolor="#333333",
        )
        ax.set_title(SETTING_PANEL_TITLES[setting_code], fontsize=12)
        ax.set_xlabel("Within-setting Amihud quintile")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Mean effective cost (bps per unit turnover)")
    fig.tight_layout()
    fig.savefig(FIGURE_I2_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_legacy_outputs()

    f1, f3 = build_appendix_i_frames()
    table_i1 = build_table_i1(f1, f3)
    table_i2 = build_table_i2(f3)

    table_i1.to_csv(TABLE_I1_PATH, index=False)
    table_i2.to_csv(TABLE_I2_PATH, index=False)
    make_figure_i1(f1, f3)
    make_figure_i2(f1, f3)

    print(f"Saved {TABLE_I1_PATH.relative_to(ROOT)}")
    print(f"Saved {TABLE_I2_PATH.relative_to(ROOT)}")
    print(f"Saved {FIGURE_I1_PATH.relative_to(ROOT)}")
    print(f"Saved {FIGURE_I2_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
