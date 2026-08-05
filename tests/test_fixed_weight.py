import pandas as pd

from adaptive_weighting.models.fixed_weight import build_fixed_weight_score


def test_build_fixed_weight_score_uses_expected_columns() -> None:
    panel = pd.DataFrame(
        {
            "momentum_score_z": [2.0],
            "liquidity_1m_z": [2.0],
            "volatility_score_z": [5.0],
        }
    )
    weights = {
        "momentum": 0.35,
        "liquidity": 0.15,
        "volatility": 0.50,
    }

    result = build_fixed_weight_score(panel, weights)

    assert result.loc[0, "momentum_score_z"] == 2.0
    assert result.loc[0, "fixed_weight_score"] == 3.2
