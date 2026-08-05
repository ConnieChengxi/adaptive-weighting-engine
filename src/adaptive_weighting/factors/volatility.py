from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rolling_volatility(daily_returns: pd.Series, window_days: int) -> pd.Series:
    return daily_returns.rolling(window_days, min_periods=window_days).std()


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
