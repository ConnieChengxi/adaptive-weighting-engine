import pandas as pd

from adaptive_weighting.factors.standardize import cross_sectional_zscore


def test_cross_sectional_zscore_adds_expected_columns() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]),
            "symbol": ["A", "B", "A", "B"],
            "current_momentum_score": [1.0, 3.0, 2.0, 4.0],
        }
    )

    result = cross_sectional_zscore(panel, ["current_momentum_score"])

    assert "current_momentum_score_z" in result.columns
    jan_values = result.loc[
        result["Date"] == pd.Timestamp("2024-01-31"),
        "current_momentum_score_z",
    ].round(6).tolist()
    assert jan_values == [-0.707107, 0.707107]
