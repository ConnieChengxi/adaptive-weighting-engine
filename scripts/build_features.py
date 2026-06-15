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
from adaptive_weighting.factors.downside_risk import compute_downside_deviation
from adaptive_weighting.factors.liquidity import compute_average_dollar_volume
from adaptive_weighting.factors.momentum import compute_momentum
from adaptive_weighting.factors.standardize import cross_sectional_zscore
from adaptive_weighting.factors.volatility import compute_rolling_volatility


BASE_CONFIG = ROOT / "config" / "base.yaml"
FACTORS_CONFIG = ROOT / "config" / "factors.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


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


def build_symbol_factor_frame(symbol: str, factor_cfg: dict) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    df = load_and_clean_price_csv(path)
    df["daily_return"] = df["Close"].pct_change()

    momentum_windows = factor_cfg["factors"]["momentum"]["windows_months"]
    liquidity_window = factor_cfg["factors"]["liquidity"]["window_days"]
    downside_window = factor_cfg["factors"]["downside_risk"]["window_days"]
    volatility_window = factor_cfg["factors"]["volatility"]["window_days"]

    df["liquidity_1m"] = compute_average_dollar_volume(df["Close"], df["Volume"], liquidity_window)
    df["downside_dev_6m"] = compute_downside_deviation(df["daily_return"], downside_window)
    df["volatility_3m"] = compute_rolling_volatility(df["daily_return"], volatility_window)

    monthly = month_end_snapshot(
        df,
        value_columns=["Close", "liquidity_1m", "downside_dev_6m", "volatility_3m"],
    )

    for window in momentum_windows:
        monthly[f"momentum_{window}m"] = compute_momentum(monthly["Close"], window)

    monthly["symbol"] = symbol
    return monthly


def build_market_feature_frame() -> pd.DataFrame:
    spy = load_and_clean_price_csv(RAW_DIR / "SPY.csv")
    vix = load_and_clean_price_csv(RAW_DIR / "VIX.csv")

    spy_monthly = month_end_snapshot(spy, ["Close"]).rename(columns={"Close": "spy_close"})
    vix_monthly = month_end_snapshot(vix, ["Close"]).rename(columns={"Close": "vix_close"})

    spy_monthly["spy_return_1m"] = spy_monthly["spy_close"].pct_change()
    spy_monthly["spy_drawdown"] = spy_monthly["spy_close"] / spy_monthly["spy_close"].cummax() - 1.0

    vix_monthly["vix_change_1m"] = vix_monthly["vix_close"].pct_change()
    vix_monthly["vix_regime"] = pd.qcut(
        vix_monthly["vix_close"],
        q=2,
        labels=["low_vix", "high_vix"],
        duplicates="drop",
    )

    return spy_monthly.merge(vix_monthly, on="Date", how="inner")


def main() -> None:
    base_cfg = load_yaml(BASE_CONFIG)
    factor_cfg = load_yaml(FACTORS_CONFIG)
    symbols = base_cfg["universe"]["etfs"]

    factor_frames = [build_symbol_factor_frame(symbol, factor_cfg) for symbol in symbols]
    panel = pd.concat(factor_frames, ignore_index=True)

    panel["downside_risk_score"] = -panel["downside_dev_6m"]
    panel["volatility_score"] = -panel["volatility_3m"]

    score_columns = [
        "momentum_3m",
        "momentum_6m",
        "liquidity_1m",
        "downside_risk_score",
        "volatility_score",
    ]
    panel = cross_sectional_zscore(panel, score_columns)

    market_features = build_market_feature_frame()
    final_panel = panel.merge(market_features, on="Date", how="left")
    final_panel = final_panel.sort_values(["Date", "symbol"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "monthly_factor_panel.csv"
    final_panel.to_csv(output_path, index=False)

    print(f"Saved {output_path.relative_to(ROOT)}")
    print(f"Rows: {len(final_panel)}")
    print(f"Date range: {final_panel['Date'].min().date()} to {final_panel['Date'].max().date()}")
    print(f"Symbols: {final_panel['symbol'].nunique()}")


if __name__ == "__main__":
    main()
