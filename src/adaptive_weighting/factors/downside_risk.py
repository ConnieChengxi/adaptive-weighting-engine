from __future__ import annotations

import numpy as np
import pandas as pd


def compute_downside_deviation(daily_returns: pd.Series, window_days: int) -> pd.Series:
    downside_returns = daily_returns.where(daily_returns < 0.0, 0.0)
    squared = downside_returns.pow(2)
    mean_squared = squared.rolling(window_days, min_periods=window_days).mean()
    return np.sqrt(mean_squared)
