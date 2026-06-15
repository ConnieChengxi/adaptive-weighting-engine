from __future__ import annotations

import numpy as np
import pandas as pd


def build_rolling_ic_weights(
    factor_ic_frame: pd.DataFrame,
    factor_columns: list[str],
    lookback_months: int,
    fallback: str,
    negative_ic_weight: float,
    fallback_weights: dict[str, float] | None = None,
    shrinkage_ic_weight: float | None = None,
    shrinkage_baseline_weight: float | None = None,
) -> pd.DataFrame:
    weights_frame = factor_ic_frame[["Date"]].copy()
    weight_columns: list[str] = []

    for factor_column in factor_columns:
        ic_column = f"{factor_column}_ic"
        weight_column = f"{factor_column}_weight"
        rolling_ic = factor_ic_frame[ic_column].rolling(lookback_months, min_periods=lookback_months).mean().shift(1)
        adjusted_ic = rolling_ic.clip(lower=negative_ic_weight)
        weights_frame[weight_column] = adjusted_ic
        weight_columns.append(weight_column)

    raw_weight_sum = weights_frame[weight_columns].sum(axis=1, min_count=len(weight_columns))
    valid_weight_rows = raw_weight_sum > 0

    for weight_column in weight_columns:
        weights_frame[weight_column] = np.where(
            valid_weight_rows,
            weights_frame[weight_column] / raw_weight_sum,
            np.nan,
        )

    if fallback == "equal_factor_weight":
        equal_weight = 1.0 / len(weight_columns)
        for weight_column in weight_columns:
            weights_frame[weight_column] = weights_frame[weight_column].fillna(equal_weight)
    elif fallback == "fixed_weight_baseline":
        if fallback_weights is None:
            raise ValueError("fallback_weights is required when fallback is fixed_weight_baseline")
        for factor_column in factor_columns:
            weight_column = f"{factor_column}_weight"
            if factor_column not in fallback_weights:
                raise ValueError(f"Missing fallback weight for {factor_column}")
            weights_frame[weight_column] = weights_frame[weight_column].fillna(fallback_weights[factor_column])

    if shrinkage_ic_weight is not None or shrinkage_baseline_weight is not None:
        if fallback_weights is None:
            raise ValueError("fallback_weights is required when shrinkage is enabled")
        if shrinkage_ic_weight is None or shrinkage_baseline_weight is None:
            raise ValueError("Both shrinkage weights must be provided")

        for factor_column in factor_columns:
            weight_column = f"{factor_column}_weight"
            baseline_weight = fallback_weights[factor_column]
            weights_frame[weight_column] = (
                shrinkage_ic_weight * weights_frame[weight_column]
                + shrinkage_baseline_weight * baseline_weight
            )

    return weights_frame


def apply_rolling_ic_score(
    panel: pd.DataFrame,
    weights_frame: pd.DataFrame,
    factor_columns: list[str],
) -> pd.DataFrame:
    merged = panel.merge(weights_frame, on="Date", how="left")
    score = 0.0
    for factor_column in factor_columns:
        score = score + merged[factor_column] * merged[f"{factor_column}_weight"]
    merged["rolling_ic_score"] = score
    return merged
