from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_zscore(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    standardized = panel.copy()
    for column in columns:
        group_mean = standardized.groupby("Date")[column].transform("mean")
        group_std = standardized.groupby("Date")[column].transform("std")
        zscore = (standardized[column] - group_mean) / group_std
        standardized[f"{column}_z"] = np.where(
            standardized[column].notna(),
            np.where(group_std > 0, zscore, 0.0),
            np.nan,
        )
    return standardized
