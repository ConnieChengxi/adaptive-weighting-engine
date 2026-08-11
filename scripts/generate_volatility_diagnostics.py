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
from adaptive_weighting.factors.standardize import cross_sectional_zscore
from adaptive_weighting.factors.volatility import compute_rolling_volatility


BASE_CONFIG = ROOT / "config" / "base.yaml"
FACTORS_CONFIG = ROOT / "config" / "factors.yaml"
MONTHLY_PANEL_PATH = ROOT / "data" / "processed" / "monthly_factor_panel.csv"
RAW_DIR = ROOT / "data" / "raw"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
PRE_TEST_SIGNAL_END = pd.Timestamp("2021-05-31")
PRE_TEST_OUTCOME_END = pd.Timestamp("2021-06-30")


VOLATILITY_CANDIDATE_LABELS = {
    "volatility_current_score_z": "Low current total volatility",
    "idio_volatility_score_z": "Low idiosyncratic volatility to SPY",
    "ewma_volatility_score_z": "Low EWMA volatility",
    "volatility_shock_score_z": "Low volatility shock",
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


def compute_rolling_idiosyncratic_volatility(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window_days: int,
) -> pd.Series:
    valid_window = market_returns.rolling(window_days, min_periods=window_days).count() >= window_days
    mean_x = market_returns.rolling(window_days, min_periods=1).mean()
    mean_y = asset_returns.rolling(window_days, min_periods=1).mean()
    mean_xy = (asset_returns * market_returns).rolling(window_days, min_periods=1).mean()
    mean_x2 = (market_returns * market_returns).rolling(window_days, min_periods=1).mean()
    mean_y2 = (asset_returns * asset_returns).rolling(window_days, min_periods=1).mean()

    cov_xy = mean_xy - (mean_x * mean_y)
    var_x = mean_x2 - mean_x.pow(2)
    var_y = mean_y2 - mean_y.pow(2)

    resid_var = var_y - cov_xy.pow(2) / var_x.replace(0.0, np.nan)
    resid_var = resid_var.clip(lower=0.0)
    return resid_var.where(valid_window).pow(0.5)


def compute_ewma_volatility(daily_returns: pd.Series, decay: float, min_periods: int) -> pd.Series:
    alpha = 1.0 - decay
    ewma_var = daily_returns.pow(2).ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
    return ewma_var.pow(0.5)


def compute_volatility_shock(current_volatility: pd.Series, window_days: int) -> pd.Series:
    baseline = current_volatility.rolling(window_days, min_periods=window_days).mean()
    return current_volatility - baseline


def compute_beta_and_r_squared(asset_returns: pd.Series, market_returns: pd.Series) -> tuple[float, float, float]:
    valid = pd.DataFrame({"asset": asset_returns, "market": market_returns}).dropna()
    if len(valid) < 12:
        return np.nan, np.nan, np.nan
    cov = valid["asset"].cov(valid["market"])
    var_m = valid["market"].var()
    if pd.isna(var_m) or var_m == 0:
        return np.nan, np.nan, np.nan
    beta = cov / var_m
    corr = valid["asset"].corr(valid["market"])
    r_squared = corr**2 if pd.notna(corr) else np.nan
    alpha = valid["asset"].mean() - beta * valid["market"].mean()
    residual = valid["asset"] - (alpha + beta * valid["market"])
    residual_vol = residual.std(ddof=1)
    return float(beta), float(r_squared), float(residual_vol)


def build_market_dependence_summary() -> pd.DataFrame:
    symbols = load_yaml(BASE_CONFIG)["universe"]["etfs"]
    spy = load_and_clean_price_csv(RAW_DIR / "SPY.csv")[["Date", "Close"]].copy()
    spy = spy[spy["Date"] <= PRE_TEST_OUTCOME_END].copy()
    spy = spy.set_index("Date")["Close"].resample("ME").last().pct_change().rename("spy_return")
    rows: list[dict[str, float | str]] = []
    for symbol in symbols:
        asset = load_and_clean_price_csv(RAW_DIR / f"{symbol}.csv")[["Date", "Close"]].copy()
        asset = asset[asset["Date"] <= PRE_TEST_OUTCOME_END].copy()
        asset = asset.set_index("Date")["Close"].resample("ME").last().pct_change().rename("asset_return")
        merged = pd.concat([asset, spy], axis=1).dropna()
        beta, r_squared, residual_vol = compute_beta_and_r_squared(merged["asset_return"], merged["spy_return"])
        rows.append(
            {
                "Ticker": symbol,
                "Beta to SPY": beta,
                "R-squared to SPY": r_squared,
                "Residual volatility": residual_vol,
                "Residual variance share": 1.0 - r_squared if pd.notna(r_squared) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_master_panel() -> pd.DataFrame:
    base_cfg = load_yaml(BASE_CONFIG)
    factor_cfg = load_yaml(FACTORS_CONFIG)
    symbols = base_cfg["universe"]["etfs"]
    volatility_window = factor_cfg["factors"]["volatility"]["window_days"]

    spy = load_and_clean_price_csv(RAW_DIR / "SPY.csv")[["Date", "Close"]].copy()
    spy["spy_daily_return"] = spy["Close"].pct_change()

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        df = load_and_clean_price_csv(RAW_DIR / f"{symbol}.csv")
        df["daily_return"] = df["Close"].pct_change()
        merged = df.merge(spy[["Date", "spy_daily_return"]], on="Date", how="left")

        merged["volatility_current"] = compute_rolling_volatility(merged["daily_return"], volatility_window)
        merged["idio_volatility"] = compute_rolling_idiosyncratic_volatility(
            merged["daily_return"],
            merged["spy_daily_return"],
            volatility_window,
        )
        merged["ewma_volatility"] = compute_ewma_volatility(
            merged["daily_return"],
            decay=0.94,
            min_periods=volatility_window,
        )
        merged["volatility_shock"] = compute_volatility_shock(merged["volatility_current"], volatility_window)

        monthly = month_end_snapshot(
            merged,
            value_columns=[
                "Close",
                "volatility_current",
                "idio_volatility",
                "ewma_volatility",
                "volatility_shock",
            ],
        )
        monthly["symbol"] = symbol
        frames.append(monthly)

    panel = pd.concat(frames, ignore_index=True).sort_values(["Date", "symbol"]).reset_index(drop=True)
    panel["next_month_return"] = panel.groupby("symbol")["Close"].shift(-1) / panel["Close"] - 1.0

    panel["volatility_current_score"] = -panel["volatility_current"]
    panel["idio_volatility_score"] = -panel["idio_volatility"]
    panel["ewma_volatility_score"] = -panel["ewma_volatility"]
    panel["volatility_shock_score"] = -panel["volatility_shock"]
    panel = cross_sectional_zscore(
        panel,
        [
            "volatility_current_score",
            "idio_volatility_score",
            "ewma_volatility_score",
            "volatility_shock_score",
        ],
    )

    ref_panel = pd.read_csv(MONTHLY_PANEL_PATH, parse_dates=["Date"])[["Date", "symbol", "vix_close"]].drop_duplicates()
    panel = panel.merge(ref_panel, on=["Date", "symbol"], how="left")
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


def summarize_ic_by_regime(ic_frame: pd.DataFrame) -> pd.DataFrame:
    vix_threshold = load_common_vix_threshold()
    regime_masks = {
        "Overall": pd.Series(True, index=ic_frame.index),
        "Normal VIX": ic_frame["vix_close"] <= vix_threshold,
        "High VIX": ic_frame["vix_close"] > vix_threshold,
    }

    rows: list[dict[str, object]] = []
    for factor_name, group in ic_frame.groupby("factor"):
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
                }
            )
    return pd.DataFrame(rows)


def save_tables(summary_frame: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    keep_columns = [
        "Factor",
        "Sample",
        "Number of months",
        "Mean IC",
        "Median IC",
        "IC standard deviation",
        "Share of positive IC months",
    ]
    formatted = summary_frame[keep_columns].copy()
    formatted["Share of positive IC months"] = formatted["Share of positive IC months"].map(lambda x: f"{x:.2%}")
    for col in ["Mean IC", "Median IC", "IC standard deviation"]:
        formatted[col] = formatted[col].round(4)
    formatted.to_csv(TABLES_DIR / "table_vr1_volatility_candidate_summary.csv", index=False)


def save_dimension_justification_table(summary_frame: pd.DataFrame) -> None:
    dependence = build_market_dependence_summary()
    total_overall = summary_frame[(summary_frame["Factor"] == "Low current total volatility") & (summary_frame["Sample"] == "Overall")].iloc[0]
    total_high_vix = summary_frame[(summary_frame["Factor"] == "Low current total volatility") & (summary_frame["Sample"] == "High VIX")].iloc[0]
    idio_overall = summary_frame[(summary_frame["Factor"] == "Low idiosyncratic volatility to SPY") & (summary_frame["Sample"] == "Overall")].iloc[0]
    idio_high_vix = summary_frame[(summary_frame["Factor"] == "Low idiosyncratic volatility to SPY") & (summary_frame["Sample"] == "High VIX")].iloc[0]

    rows = [
        {
            "Evidence block": "Market contamination",
            "Metric": "Average SPY R-squared across sectors",
            "Value": dependence["R-squared to SPY"].mean(),
            "Why it matters": "A large share of sector ETF variance is explained by the common market component, so total volatility is mechanically a mixed object.",
        },
        {
            "Evidence block": "Market contamination",
            "Metric": "Range of SPY R-squared across sectors",
            "Value": f"{dependence['R-squared to SPY'].min():.3f} to {dependence['R-squared to SPY'].max():.3f}",
            "Why it matters": "The degree of market contamination differs materially across sectors, so total volatility does not isolate sector-specific uncertainty consistently.",
        },
        {
            "Evidence block": "Market contamination",
            "Metric": "Average residual variance share across sectors",
            "Value": dependence["Residual variance share"].mean(),
            "Why it matters": "Residual uncertainty is the unexplained component after removing SPY; this is conceptually closer to the sector-specific uncertainty dimension.",
        },
        {
            "Evidence block": "Predictive comparison",
            "Metric": "Overall mean IC: low current total volatility",
            "Value": float(total_overall["Mean IC"]),
            "Why it matters": "The total-volatility candidate does not provide positive cross-sectional ranking value under the retained specification.",
        },
        {
            "Evidence block": "Predictive comparison",
            "Metric": "Overall mean IC: low idiosyncratic volatility to SPY",
            "Value": float(idio_overall["Mean IC"]),
            "Why it matters": "The idiosyncratic-volatility candidate better matches the residual-uncertainty concept and delivers a stronger overall signal.",
        },
        {
            "Evidence block": "Predictive comparison",
            "Metric": "High-VIX mean IC: low current total volatility",
            "Value": float(total_high_vix["Mean IC"]),
            "Why it matters": "Even in stressed states, total volatility does not separate sector-specific uncertainty from market-wide stress cleanly.",
        },
        {
            "Evidence block": "Predictive comparison",
            "Metric": "High-VIX mean IC: low idiosyncratic volatility to SPY",
            "Value": float(idio_high_vix["Mean IC"]),
            "Why it matters": "In stressed states, the residual-uncertainty proxy becomes materially more informative for next-month ranking.",
        },
    ]
    table = pd.DataFrame(rows)
    table["Value"] = table["Value"].map(lambda x: f"{x:.4f}" if isinstance(x, float) and pd.notna(x) else x)
    table.to_csv(TABLES_DIR / "table_vr2_volatility_dimension_justification.csv", index=False)


def save_ranking_disagreement_table(panel: pd.DataFrame) -> None:
    vix_threshold = load_common_vix_threshold()
    month_rows: list[dict[str, object]] = []

    for date, group in panel.groupby("Date"):
        valid = group[["symbol", "volatility_current", "idio_volatility", "vix_close"]].dropna().copy()
        if len(valid) < 3:
            continue

        valid["total_rank"] = valid["volatility_current"].rank(method="first", ascending=True)
        valid["idio_rank"] = valid["idio_volatility"].rank(method="first", ascending=True)
        valid["abs_rank_diff"] = (valid["total_rank"] - valid["idio_rank"]).abs()

        top1_total = valid.nsmallest(1, "volatility_current")["symbol"].tolist()
        top1_idio = valid.nsmallest(1, "idio_volatility")["symbol"].tolist()
        top3_total = set(valid.nsmallest(3, "volatility_current")["symbol"].tolist())
        top3_idio = set(valid.nsmallest(3, "idio_volatility")["symbol"].tolist())

        month_rows.append(
            {
                "Date": date,
                "sample": "High VIX" if float(valid["vix_close"].iloc[0]) > vix_threshold else "Normal VIX",
                "mean_abs_rank_difference": float(valid["abs_rank_diff"].mean()),
                "top_sector_differs": float(top1_total != top1_idio),
                "top3_overlap_share": float(len(top3_total & top3_idio) / 3.0),
                "top3_set_differs": float(top3_total != top3_idio),
            }
        )

    month_frame = pd.DataFrame(month_rows)
    rows: list[dict[str, object]] = []
    for sample_name, sample in {
        "Overall": month_frame,
        "Normal VIX": month_frame[month_frame["sample"] == "Normal VIX"],
        "High VIX": month_frame[month_frame["sample"] == "High VIX"],
    }.items():
        if sample.empty:
            continue
        rows.append(
            {
                "Sample": sample_name,
                "Number of months": int(sample.shape[0]),
                "Mean absolute rank difference": float(sample["mean_abs_rank_difference"].mean()),
                "Median absolute rank difference": float(sample["mean_abs_rank_difference"].median()),
                "Share of months with different top sector": float(sample["top_sector_differs"].mean()),
                "Average top-3 overlap share": float(sample["top3_overlap_share"].mean()),
                "Share of months with different top-3 set": float(sample["top3_set_differs"].mean()),
            }
        )

    table = pd.DataFrame(rows)
    for col in [
        "Mean absolute rank difference",
        "Median absolute rank difference",
        "Share of months with different top sector",
        "Average top-3 overlap share",
        "Share of months with different top-3 set",
    ]:
        if "Share" in col:
            table[col] = table[col].map(lambda x: f"{x:.2%}")
        else:
            table[col] = table[col].map(lambda x: f"{x:.2f}")
    table.to_csv(TABLES_DIR / "table_vr3_uncertainty_ranking_disagreement.csv", index=False)


def save_figures(ic_frame: pd.DataFrame, summary_frame: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = list(VOLATILITY_CANDIDATE_LABELS.values())
    display_labels = {
        "Low current total volatility": "Total volatility",
        "Low idiosyncratic volatility to SPY": "Idiosyncratic volatility to SPY",
        "Low EWMA volatility": "EWMA volatility",
        "Low volatility shock": "Volatility shock",
    }

    plot_frame = summary_frame[summary_frame["Sample"].isin(["Normal VIX", "High VIX"])].copy()
    plot_frame["Display factor"] = plot_frame["Factor"].map(display_labels)
    fig, ax = plt.subplots(figsize=(10.5, 6))
    x = np.arange(len(order))
    width = 0.34
    colors = {"Normal VIX": "#7f8c8d", "High VIX": "#c44e52"}
    for idx, sample_name in enumerate(["Normal VIX", "High VIX"]):
        sample = plot_frame[plot_frame["Sample"] == sample_name].set_index("Factor").reindex(order)
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
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([display_labels[label] for label in order], rotation=14, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Mean monthly factor IC")
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.22)
    upper = float(plot_frame["Mean IC"].max())
    lower = float(plot_frame["Mean IC"].min())
    ax.set_ylim(lower - 0.01, upper + 0.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure_vr2_volatility_candidate_ic_by_regime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    panel = build_master_panel()
    panel = panel[panel["Date"] <= PRE_TEST_SIGNAL_END].copy()
    factor_columns = {
        VOLATILITY_CANDIDATE_LABELS["volatility_current_score_z"]: "volatility_current_score_z",
        VOLATILITY_CANDIDATE_LABELS["idio_volatility_score_z"]: "idio_volatility_score_z",
        VOLATILITY_CANDIDATE_LABELS["ewma_volatility_score_z"]: "ewma_volatility_score_z",
        VOLATILITY_CANDIDATE_LABELS["volatility_shock_score_z"]: "volatility_shock_score_z",
    }

    ic_frame = compute_monthly_ic_frame(panel, factor_columns)
    summary_frame = summarize_ic_by_regime(ic_frame)
    save_tables(summary_frame)
    save_dimension_justification_table(summary_frame)
    save_ranking_disagreement_table(panel)
    save_figures(ic_frame, summary_frame)


if __name__ == "__main__":
    main()
