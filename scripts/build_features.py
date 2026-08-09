from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_weighting.data.preprocess import load_and_clean_price_csv
from adaptive_weighting.factors.liquidity import compute_amihud_illiquidity
from adaptive_weighting.factors.momentum import compute_momentum, compute_residual_momentum
from adaptive_weighting.factors.spread import compute_corwin_schultz_spread
from adaptive_weighting.factors.standardize import cross_sectional_zscore
from adaptive_weighting.factors.volatility import compute_rolling_idiosyncratic_volatility


BASE_CONFIG = ROOT / "config" / "base.yaml"
FACTORS_CONFIG = ROOT / "config" / "factors.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MAIN_PANEL_FILENAME = "monthly_factor_panel.csv"
DAILY_SPREAD_FILENAME = "daily_spread_panel.csv"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def month_end_snapshot(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    monthly = (
        df.set_index("Date")[value_columns]
        .resample("ME")
        .last()
        .dropna(how="all")
        .reset_index()
    )
    return monthly


def month_end_average(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    monthly = (
        df.set_index("Date")[value_columns]
        .resample("ME")
        .mean()
        .dropna(how="all")
        .reset_index()
    )
    return monthly


def build_symbol_factor_frame(symbol: str, factor_cfg: dict, spy_daily: pd.DataFrame) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    df = load_and_clean_price_csv(path)
    df["daily_return"] = df["Close"].pct_change()
    df = df.merge(spy_daily, on="Date", how="left")

    momentum_cfg = factor_cfg["factors"]["momentum"]
    liquidity_cfg = factor_cfg["factors"]["liquidity"]
    volatility_cfg = factor_cfg["factors"]["volatility"]

    momentum_windows = momentum_cfg["current_windows_months"]
    liquidity_window = liquidity_cfg["window_days"]
    volatility_window = volatility_cfg["window_days"]
    lower_q = liquidity_cfg["winsorization"]["lower_quantile"]
    upper_q = liquidity_cfg["winsorization"]["upper_quantile"]

    # The active liquidity candidate is month-end log Amihud, computed from an
    # expanding-window winsorised daily series and then aggregated over 21 days.
    df["liquidity_1m"] = compute_amihud_illiquidity(
        df["Close"],
        df["Volume"],
        liquidity_window,
        lower_q=lower_q,
        upper_q=upper_q,
    )
    df["volatility_raw"] = compute_rolling_idiosyncratic_volatility(
        df["daily_return"],
        df["spy_daily_return"],
        volatility_window,
    )
    df["corwin_schultz_spread"] = compute_corwin_schultz_spread(df["High"], df["Low"])

    monthly_levels = month_end_snapshot(
        df,
        value_columns=["Close", "liquidity_1m", "volatility_raw"],
    )
    monthly_spread = month_end_average(df, value_columns=["corwin_schultz_spread"])
    monthly = monthly_levels.merge(monthly_spread, on="Date", how="left")

    for window in momentum_windows:
        monthly[f"momentum_{window}m"] = compute_momentum(monthly["Close"], window)
    monthly["current_momentum_score"] = monthly[
        [f"momentum_{window}m" for window in momentum_windows]
    ].mean(axis=1, skipna=True)
    monthly = monthly.drop(columns=[f"momentum_{window}m" for window in momentum_windows])

    monthly["symbol"] = symbol
    return monthly


def build_daily_spread_frame(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    df = load_and_clean_price_csv(path)
    df["corwin_schultz_spread"] = compute_corwin_schultz_spread(df["High"], df["Low"])
    df["symbol"] = symbol
    return df[["Date", "symbol", "corwin_schultz_spread"]].copy()


def build_market_feature_frame() -> pd.DataFrame:
    spy = load_and_clean_price_csv(RAW_DIR / "SPY.csv")
    vix = load_and_clean_price_csv(RAW_DIR / "VIX.csv")

    spy_monthly = month_end_snapshot(spy, ["Close"]).rename(columns={"Close": "spy_close"})
    vix_monthly = month_end_snapshot(vix, ["Close"]).rename(columns={"Close": "vix_close"})

    spy_monthly["spy_return_1m"] = spy_monthly["spy_close"].pct_change()
    spy_monthly["spy_drawdown"] = spy_monthly["spy_close"] / spy_monthly["spy_close"].cummax() - 1.0

    vix_monthly["vix_change_1m"] = vix_monthly["vix_close"].pct_change()

    return spy_monthly.merge(vix_monthly, on="Date", how="inner")


def build_spy_daily_frame() -> pd.DataFrame:
    spy = load_and_clean_price_csv(RAW_DIR / "SPY.csv")
    spy["spy_daily_return"] = spy["Close"].pct_change()
    return spy[["Date", "spy_daily_return"]].copy()


def main() -> None:
    base_cfg = load_yaml(BASE_CONFIG)
    factor_cfg = load_yaml(FACTORS_CONFIG)
    momentum_cfg = factor_cfg["factors"]["momentum"]
    symbols = base_cfg["universe"]["etfs"]
    spy_daily = build_spy_daily_frame()

    factor_frames = [build_symbol_factor_frame(symbol, factor_cfg, spy_daily) for symbol in symbols]
    panel = pd.concat(factor_frames, ignore_index=True)
    daily_spread_frames = [build_daily_spread_frame(symbol) for symbol in symbols]
    daily_spread_panel = pd.concat(daily_spread_frames, ignore_index=True).sort_values(["Date", "symbol"]).reset_index(drop=True)

    panel["volatility_score"] = -panel["volatility_raw"]

    market_features = build_market_feature_frame()
    final_panel = panel.merge(market_features, on="Date", how="left")

    enriched_frames: list[pd.DataFrame] = []
    for _, group in final_panel.groupby("symbol", sort=False):
        enriched = group.copy()
        enriched["momentum_score"] = compute_residual_momentum(
            monthly_close=enriched["Close"],
            market_close=enriched["spy_close"],
            residual_window_months=momentum_cfg["residual_window_months"],
            lookback_months=momentum_cfg["retained_lookback_months"],
            skip_recent_months=momentum_cfg["skip_recent_months"],
        )
        enriched_frames.append(enriched)

    final_panel = pd.concat(enriched_frames, ignore_index=True)

    score_columns = [
        "current_momentum_score",
        "momentum_score",
        "liquidity_1m",
        "volatility_score",
    ]
    factor_panel = cross_sectional_zscore(final_panel, score_columns)
    factor_panel = factor_panel.sort_values(["Date", "symbol"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    main_output_path = PROCESSED_DIR / MAIN_PANEL_FILENAME
    daily_spread_output_path = PROCESSED_DIR / DAILY_SPREAD_FILENAME

    factor_panel.to_csv(main_output_path, index=False)
    daily_spread_panel.to_csv(daily_spread_output_path, index=False)

    print(f"Saved {main_output_path.relative_to(ROOT)}")
    print(f"Saved {daily_spread_output_path.relative_to(ROOT)}")
    print(f"Rows: {len(factor_panel)}")
    print(f"Date range: {factor_panel['Date'].min().date()} to {factor_panel['Date'].max().date()}")
    print(f"Symbols: {factor_panel['symbol'].nunique()}")


if __name__ == "__main__":
    main()
