from __future__ import annotations

import pandas as pd


def build_fixed_weight_score(panel: pd.DataFrame, factor_weights: dict[str, float]) -> pd.DataFrame:
    scored = panel.copy()
    factor_map = {
        "momentum": "momentum_score_z",
        "liquidity": "liquidity_1m_z",
        "volatility": "volatility_score_z",
    }
    score = 0.0
    for factor_name, factor_column in factor_map.items():
        if factor_name not in factor_weights:
            continue
        score = score + factor_weights[factor_name] * scored[factor_column]
    scored["fixed_weight_score"] = score
    return scored
