from __future__ import annotations

import pandas as pd


def compute_momentum(monthly_close: pd.Series, window_months: int) -> pd.Series:
    """Cumulative return over a monthly lookback window."""
    return monthly_close / monthly_close.shift(window_months) - 1.0
