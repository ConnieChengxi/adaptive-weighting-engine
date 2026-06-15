from __future__ import annotations

import pandas as pd


def build_fixed_weight_score(panel: pd.DataFrame, factor_weights: dict[str, float]) -> pd.DataFrame:
    scored = panel.copy()
    scored["momentum_score_z"] = scored[["momentum_3m_z", "momentum_6m_z"]].mean(axis=1, skipna=True)

    scored["fixed_weight_score"] = (
        factor_weights["momentum"] * scored["momentum_score_z"]
        + factor_weights["liquidity"] * scored["liquidity_1m_z"]
        + factor_weights["downside_risk"] * scored["downside_risk_score_z"]
        + factor_weights["volatility"] * scored["volatility_score_z"]
    )
    return scored
