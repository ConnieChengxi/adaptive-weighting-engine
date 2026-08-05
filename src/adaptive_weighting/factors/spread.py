from __future__ import annotations

import numpy as np
import pandas as pd


CS_DENOMINATOR = 3.0 - 2.0 * np.sqrt(2.0)


def compute_corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Estimate the proportional bid-ask spread from daily high/low prices.

    The estimator follows Corwin and Schultz (2012) and produces a daily,
    dimensionless spread proxy that can be interpreted as a percentage of price.
    """

    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    valid_prices = (high > 0.0) & (low > 0.0)

    log_hl = pd.Series(np.nan, index=high.index, dtype=float)
    log_hl.loc[valid_prices] = np.log(high.loc[valid_prices] / low.loc[valid_prices])

    beta = log_hl.pow(2).rolling(2, min_periods=2).sum()

    high_lag = high.shift(1)
    low_lag = low.shift(1)
    high_two_day = pd.concat([high, high_lag], axis=1).max(axis=1)
    low_two_day = pd.concat([low, low_lag], axis=1).min(axis=1)

    valid_two_day = (high_two_day > 0.0) & (low_two_day > 0.0)
    gamma = pd.Series(np.nan, index=high.index, dtype=float)
    gamma.loc[valid_two_day] = np.log(high_two_day.loc[valid_two_day] / low_two_day.loc[valid_two_day]).pow(2)

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / CS_DENOMINATOR - np.sqrt(gamma / CS_DENOMINATOR)
    alpha = alpha.clip(lower=0.0)

    exp_alpha = np.exp(alpha)
    spread = 2.0 * (exp_alpha - 1.0) / (1.0 + exp_alpha)
    return spread.rename("corwin_schultz_spread")
