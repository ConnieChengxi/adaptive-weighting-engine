from __future__ import annotations

import pandas as pd

from adaptive_weighting.factors.liquidity import winsorize_series


def test_winsorize_series_is_prefix_invariant() -> None:
    prefix = pd.Series([1.0, 2.0, 3.0, 50.0])
    full = pd.Series([1.0, 2.0, 3.0, 50.0, 4.0, 1000.0])

    prefix_result = winsorize_series(prefix, lower_q=0.25, upper_q=0.75)
    full_result = winsorize_series(full, lower_q=0.25, upper_q=0.75).iloc[: len(prefix)]

    pd.testing.assert_series_equal(prefix_result, full_result)
