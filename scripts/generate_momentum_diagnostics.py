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

from adaptive_weighting.factors.standardize import cross_sectional_zscore


MONTHLY_PANEL_PATH = ROOT / "data" / "processed" / "monthly_factor_panel.csv"
FACTORS_CONFIG = ROOT / "config" / "factors.yaml"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
PRE_TEST_SIGNAL_END = pd.Timestamp("2021-05-31")
PRE_TEST_OUTCOME_END = pd.Timestamp("2021-06-30")


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


MOMENTUM_CONFIG = load_yaml(FACTORS_CONFIG)["factors"]["momentum"]
CURRENT_MOMENTUM_LABEL = "Current 3m + 6m momentum"


def residual_candidate_column(lookback_months: int, skip_recent_months: int) -> str:
    return f"residual_momentum_{lookback_months}_{skip_recent_months}"


def residual_candidate_label(lookback_months: int, skip_recent_months: int) -> str:
    return f"{lookback_months}-{skip_recent_months} residual momentum"


MOMENTUM_CANDIDATE_LABELS = {
    "current_momentum_score_z": CURRENT_MOMENTUM_LABEL,
    **{
        f"{residual_candidate_column(lookback, MOMENTUM_CONFIG['skip_recent_months'])}_z":
        residual_candidate_label(lookback, MOMENTUM_CONFIG["skip_recent_months"])
        for lookback in MOMENTUM_CONFIG["candidate_lookbacks_months"]
    },
}

MOMENTUM_REGIME_PLOT_ORDER = [CURRENT_MOMENTUM_LABEL] + [
    residual_candidate_label(lookback, MOMENTUM_CONFIG["skip_recent_months"])
    for lookback in MOMENTUM_CONFIG["candidate_lookbacks_months"]
]


def load_common_vix_threshold() -> float:
    panel = pd.read_csv(MONTHLY_PANEL_PATH, usecols=["Date", "vix_close"], parse_dates=["Date"])
    vix = panel.drop_duplicates(subset=["Date"]).dropna(subset=["vix_close"]).sort_values("Date")
    vix = vix[vix["Date"] <= PRE_TEST_OUTCOME_END]
    return float(vix["vix_close"].quantile(0.75))


def compute_skip_month_compounded_residual_momentum(
    residual_returns: pd.Series,
    lookback_months: int,
    skip_recent_months: int = 1,
) -> pd.Series:
    transformed = np.log1p(residual_returns.clip(lower=-0.95))
    return (
        transformed.shift(skip_recent_months)
        .rolling(lookback_months, min_periods=lookback_months)
        .sum()
        .pipe(np.expm1)
    )


def compute_residual_returns(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window_months: int = 24,
) -> pd.Series:
    mean_x = market_returns.rolling(window_months, min_periods=window_months).mean()
    mean_y = asset_returns.rolling(window_months, min_periods=window_months).mean()
    mean_xy = (asset_returns * market_returns).rolling(window_months, min_periods=window_months).mean()
    mean_x2 = (market_returns * market_returns).rolling(window_months, min_periods=window_months).mean()

    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_x2 - mean_x.pow(2)
    beta = cov_xy / var_x.replace(0.0, np.nan)
    alpha = mean_y - beta * mean_x

    fitted = alpha + beta * market_returns
    return asset_returns - fitted


def build_master_panel() -> pd.DataFrame:
    panel = pd.read_csv(MONTHLY_PANEL_PATH, parse_dates=["Date"]).sort_values(["symbol", "Date"]).copy()
    panel["next_month_return"] = panel.groupby("symbol")["Close"].shift(-1) / panel["Close"] - 1.0
    panel["monthly_return"] = panel.groupby("symbol")["Close"].pct_change()
    panel["spy_monthly_return"] = panel["spy_close"] / panel["spy_close"].shift(1) - 1.0

    symbol_frames: list[pd.DataFrame] = []
    for _, group in panel.groupby("symbol", sort=False):
        enriched = group.copy()
        enriched["residual_monthly_return"] = compute_residual_returns(
            enriched["monthly_return"],
            enriched["spy_monthly_return"],
            window_months=MOMENTUM_CONFIG["residual_window_months"],
        )
        for lookback in MOMENTUM_CONFIG["candidate_lookbacks_months"]:
            enriched[residual_candidate_column(lookback, MOMENTUM_CONFIG["skip_recent_months"])] = (
                compute_skip_month_compounded_residual_momentum(
                    enriched["residual_monthly_return"],
                    lookback,
                    MOMENTUM_CONFIG["skip_recent_months"],
                )
            )
        symbol_frames.append(enriched)

    panel = pd.concat(symbol_frames, ignore_index=True)

    panel = cross_sectional_zscore(
        panel,
        ["current_momentum_score"]
        + [
            residual_candidate_column(lookback, MOMENTUM_CONFIG["skip_recent_months"])
            for lookback in MOMENTUM_CONFIG["candidate_lookbacks_months"]
        ],
    )
    return panel


def compute_monthly_ic_frame(panel: pd.DataFrame, factor_columns: dict[str, str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for factor_name, factor_column in factor_columns.items():
        for date, group in panel.groupby("Date"):
            valid = group[[factor_column, "next_month_return", "vix_close"]].dropna()
            if len(valid) < 2:
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


def classify_candidate(overall_mean_ic: float, sample_name: str, sample_mean_ic: float) -> str:
    return ""


def summarize_ic_by_regime(ic_frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    vix_threshold = load_common_vix_threshold()
    regime_masks = {
        "Overall": pd.Series(True, index=ic_frame.index),
        "Normal VIX": ic_frame["vix_close"] <= vix_threshold,
        "High VIX": ic_frame["vix_close"] > vix_threshold,
    }

    rows: list[dict[str, object]] = []
    status_map = {
        ("12-1 residual momentum", "Overall"): "Retained in decision summary",
        ("12-1 residual momentum", "Normal VIX"): "Limited normal-VIX support",
        ("12-1 residual momentum", "High VIX"): "Strongest high-VIX support",
        ("6-1 residual momentum", "Overall"): "Limited overall support",
        ("6-1 residual momentum", "Normal VIX"): "Limited normal-VIX support",
        ("6-1 residual momentum", "High VIX"): "Moderate high-VIX support",
        ("Current 3m + 6m momentum", "Overall"): "Limited overall support",
        ("Current 3m + 6m momentum", "Normal VIX"): "Limited normal-VIX support",
        ("Current 3m + 6m momentum", "High VIX"): "Limited high-VIX support",
    }
    for factor_name, group in ic_frame.groupby("factor"):
        overall_mean_ic = float(group["ic"].mean())
        for sample_name, mask in regime_masks.items():
            sample = group.loc[mask.loc[group.index], "ic"].dropna()
            if sample.empty:
                continue
            rows.append(
                {
                    "Factor": factor_name,
                    "Sample": sample_name,
                    "Number of months": int(sample.shape[0]),
                    "Mean IC": float(sample.mean()),
                    "Median IC": float(sample.median()),
                    "IC standard deviation": float(sample.std(ddof=1)) if len(sample) > 1 else np.nan,
                    "IC 25th percentile": float(sample.quantile(0.25)),
                    "IC 75th percentile": float(sample.quantile(0.75)),
                    "Share of positive IC months": float((sample > 0).mean()),
                    "Diagnostic status": status_map.get((factor_name, sample_name), ""),
                }
            )
    return pd.DataFrame(rows), vix_threshold


def summarize_overlap(panel: pd.DataFrame, factor_columns: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor_name, factor_column in factor_columns.items():
        comparison = pd.DataFrame(
            {
                "factor_value": panel[factor_column],
                "current_momentum": panel["current_momentum_score_z"],
            }
        ).dropna()
        current_momentum_correlation = (
            1.0
            if factor_column == "current_momentum_score_z"
            else float(comparison["factor_value"].corr(comparison["current_momentum"])) if not comparison.empty else np.nan
        )
        rows.append(
            {
                "Factor": factor_name,
                "Pooled correlation with current 3m + 6m momentum": current_momentum_correlation,
            }
        )
    return pd.DataFrame(rows)


def format_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "Factor",
        "Sample",
        "Number of months",
        "Mean IC",
        "Median IC",
        "IC standard deviation",
        "Share of positive IC months",
    ]
    formatted = df[keep_columns].copy()
    formatted["Share of positive IC months"] = formatted["Share of positive IC months"].map(lambda x: f"{x:.2%}")
    for col in ["Mean IC", "Median IC", "IC standard deviation"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")
    return formatted


def build_market_adjusted_summary_table(summary_df: pd.DataFrame, overlap_df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("Current 3m + 6m momentum", "Raw benchmark"),
        ("6-1 residual momentum", "Shorter residual specification"),
        ("12-1 residual momentum", "Skip-month residual specification"),
    ]
    rows: list[dict[str, object]] = []
    for factor_name, interpretation in candidates:
        overall = summary_df[(summary_df["Factor"] == factor_name) & (summary_df["Sample"] == "Overall")].iloc[0]
        high_vix = summary_df[(summary_df["Factor"] == factor_name) & (summary_df["Sample"] == "High VIX")].iloc[0]
        overlap_row = overlap_df[overlap_df["Factor"] == factor_name].iloc[0]
        if factor_name == "12-1 residual momentum":
            decision = "Retained"
        else:
            decision = "Not retained"
        rows.append(
            {
                "Candidate proxy": factor_name,
                "Specification role": interpretation,
                "Overall mean IC": float(overall["Mean IC"]),
                "High-VIX mean IC": float(high_vix["Mean IC"]),
                "Correlation with raw momentum": float(
                    overlap_row["Pooled correlation with current 3m + 6m momentum"]
                ),
                "Selection outcome": decision,
            }
        )
    table = pd.DataFrame(rows)
    for column in ["Overall mean IC", "High-VIX mean IC", "Correlation with raw momentum"]:
        table[column] = table[column].map(lambda x: f"{x:.4f}")
    return table


def make_momentum_regime_bar_chart(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["Sample"].isin(["Normal VIX", "High VIX"])].copy()
    order = MOMENTUM_REGIME_PLOT_ORDER
    sample_order = ["Normal VIX", "High VIX"]
    x = np.arange(len(order))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    colors = {"Normal VIX": "#4C78A8", "High VIX": "#E07B39"}
    for idx, sample_name in enumerate(sample_order):
        sample = plot_df[plot_df["Sample"] == sample_name].set_index("Factor").reindex(order)
        bars = ax.bar(
            x + (idx - 0.5) * width,
            sample["Mean IC"],
            width=width,
            label=sample_name,
            color=colors[sample_name],
        )
        finite_values = [abs(float(v)) for v in sample["Mean IC"].dropna()]
        offset = max(max(finite_values) * 0.012, 0.002) if finite_values else 0.002
        for bar, value in zip(bars, sample["Mean IC"]):
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
    ax.set_xticklabels(order, rotation=8)
    ax.set_ylabel("Mean monthly IC")
    ax.set_xlabel("Candidate proxy")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    ax.margins(x=0.05)
    upper = float(plot_df["Mean IC"].max())
    lower = float(plot_df["Mean IC"].min())
    ax.set_ylim(lower - 0.01, upper + 0.02)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
    fig.savefig(FIGURES_DIR / "figure_mr2_momentum_candidate_ic_by_regime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_master_panel()
    panel = panel[panel["Date"] <= PRE_TEST_SIGNAL_END].copy()
    factor_columns = {label: column for column, label in MOMENTUM_CANDIDATE_LABELS.items()}
    ic_frame = compute_monthly_ic_frame(panel, factor_columns)
    summary_frame, vix_threshold = summarize_ic_by_regime(ic_frame)
    overlap_frame = summarize_overlap(panel, factor_columns)

    format_summary_table(summary_frame).to_csv(
        TABLES_DIR / "table_mr1_momentum_candidate_summary.csv",
        index=False,
    )

    build_market_adjusted_summary_table(summary_frame, overlap_frame).to_csv(
        TABLES_DIR / "table_mr3_market_adjusted_relative_performance_summary.csv",
        index=False,
    )

    make_momentum_regime_bar_chart(summary_frame)

    print("Saved outputs/tables/table_mr1_momentum_candidate_summary.csv")
    print("Saved outputs/tables/table_mr3_market_adjusted_relative_performance_summary.csv")
    print("Saved outputs/figures/figure_mr2_momentum_candidate_ic_by_regime.png")
    print(f"VIX high-regime threshold (75th percentile): {vix_threshold:.3f}")


if __name__ == "__main__":
    main()
