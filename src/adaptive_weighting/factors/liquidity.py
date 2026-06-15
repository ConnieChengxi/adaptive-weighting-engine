from __future__ import annotations

import numpy as np
import pandas as pd


def compute_average_dollar_volume(close: pd.Series, volume: pd.Series, window_days: int) -> pd.Series:
    dollar_volume = close * volume
    rolling_avg = dollar_volume.rolling(window_days, min_periods=window_days).mean()
    return np.log(rolling_avg.replace(0, np.nan))
