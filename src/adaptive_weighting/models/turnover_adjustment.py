from __future__ import annotations

import numpy as np
import pandas as pd


def project_to_simplex(vector: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""
    if np.all(vector == 0):
        return np.full_like(vector, 1.0 / len(vector), dtype=float)

    sorted_vector = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_vector)
    rho_candidates = sorted_vector - (cumulative - 1.0) / (np.arange(len(vector)) + 1)
    positive = np.where(rho_candidates > 0)[0]
    if len(positive) == 0:
        return np.full_like(vector, 1.0 / len(vector), dtype=float)

    rho = positive[-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    projected = np.maximum(vector - theta, 0.0)
    projected_sum = projected.sum()
    if projected_sum <= 0:
        return np.full_like(vector, 1.0 / len(vector), dtype=float)
    return projected / projected_sum


def apply_post_model_turnover_adjustment(
    weights_frame: pd.DataFrame,
    weight_columns: list[str],
    penalty_lambda: float,
) -> pd.DataFrame:
    """Apply a PTA-style smoothing step to dynamic model weights.

    The implementation uses a soft-thresholding step around the previous adjusted
    weights followed by simplex projection so weights remain non-negative and sum to one.
    """

    adjusted = weights_frame.copy()
    if adjusted.empty:
        return adjusted

    weight_matrix = adjusted[weight_columns].to_numpy(dtype=float)
    finite_rows = np.isfinite(weight_matrix).all(axis=1)
    if not finite_rows.any():
        return adjusted

    first_valid_idx = int(np.where(finite_rows)[0][0])
    previous_weights = adjusted.loc[first_valid_idx, weight_columns].to_numpy(dtype=float)
    previous_weights = project_to_simplex(previous_weights)
    adjusted.loc[first_valid_idx, weight_columns] = previous_weights

    threshold = penalty_lambda / 2.0
    for idx in range(first_valid_idx + 1, len(adjusted)):
        target_weights = adjusted.loc[idx, weight_columns].to_numpy(dtype=float)
        if not np.isfinite(target_weights).all():
            continue
        delta = target_weights - previous_weights
        shrunk_delta = np.sign(delta) * np.maximum(np.abs(delta) - threshold, 0.0)
        candidate = previous_weights + shrunk_delta
        projected = project_to_simplex(candidate)
        adjusted.loc[idx, weight_columns] = projected
        previous_weights = projected

    return adjusted
