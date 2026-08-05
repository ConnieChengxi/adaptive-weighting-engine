from __future__ import annotations

import numpy as np
import pandas as pd


def compute_average_dollar_volume(close: pd.Series, volume: pd.Series, window_days: int) -> pd.Series:
    dollar_volume = close * volume
    rolling_avg = dollar_volume.rolling(window_days, min_periods=window_days).mean()
    return np.log(rolling_avg.replace(0, np.nan))


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if series.dropna().empty:
        return series.copy()
    lower = series.expanding(min_periods=1).quantile(lower_q)
    upper = series.expanding(min_periods=1).quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def compute_amihud_illiquidity(
    close: pd.Series,
    volume: pd.Series,
    window_days: int,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    abs_return = close.pct_change().abs()
    dollar_volume = (close * volume).replace(0, np.nan)
    amihud_daily = abs_return / dollar_volume
    amihud_daily = winsorize_series(amihud_daily, lower_q=lower_q, upper_q=upper_q)
    rolling_mean = amihud_daily.rolling(window_days, min_periods=window_days).mean()
    return np.log1p(rolling_mean)
