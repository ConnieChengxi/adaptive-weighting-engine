from __future__ import annotations

import pandas as pd


def compute_rolling_volatility(daily_returns: pd.Series, window_days: int) -> pd.Series:
    return daily_returns.rolling(window_days, min_periods=window_days).std()
