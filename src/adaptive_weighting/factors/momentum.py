from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum(monthly_close: pd.Series, window_months: int) -> pd.Series:
    """Cumulative return over a monthly lookback window."""
    return monthly_close / monthly_close.shift(window_months) - 1.0


def compute_residual_returns(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window_months: int = 24,
) -> pd.Series:
    """Rolling market-model residual returns using only information available up to each month."""
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


def compute_skip_month_compounded_residual_momentum(
    residual_returns: pd.Series,
    lookback_months: int,
    skip_recent_months: int = 1,
) -> pd.Series:
    """Compounded residual momentum over a skip-month window."""
    transformed = np.log1p(residual_returns.clip(lower=-0.95))
    return (
        transformed.shift(skip_recent_months)
        .rolling(lookback_months, min_periods=lookback_months)
        .sum()
        .pipe(np.expm1)
    )


def compute_residual_momentum(
    monthly_close: pd.Series,
    market_close: pd.Series,
    residual_window_months: int = 24,
    lookback_months: int = 12,
    skip_recent_months: int = 1,
) -> pd.Series:
    """12-1 style residual momentum based on rolling market-model residual returns."""
    asset_returns = monthly_close.pct_change()
    market_returns = market_close.pct_change()
    residual_returns = compute_residual_returns(
        asset_returns=asset_returns,
        market_returns=market_returns,
        window_months=residual_window_months,
    )
    return compute_skip_month_compounded_residual_momentum(
        residual_returns=residual_returns,
        lookback_months=lookback_months,
        skip_recent_months=skip_recent_months,
    )
