import pandas as pd

from adaptive_weighting.ic.compute_ic import compute_monthly_factor_ic
from adaptive_weighting.models.rolling_ic_weighting import build_rolling_ic_weights


def test_compute_monthly_factor_ic_returns_expected_columns() -> None:
    panel = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]),
            "factor_a": [1.0, 2.0, 2.0, 1.0],
            "next_month_return": [0.1, 0.2, 0.2, 0.1],
        }
    )

    result = compute_monthly_factor_ic(panel, ["factor_a"], "next_month_return")

    assert list(result.columns) == ["Date", "factor_a_ic"]


def test_build_rolling_ic_weights_falls_back_to_equal_weights() -> None:
    ic_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "factor_a_ic": [-0.1, -0.2],
            "factor_b_ic": [-0.3, -0.4],
        }
    )

    result = build_rolling_ic_weights(
        factor_ic_frame=ic_frame,
        factor_columns=["factor_a", "factor_b"],
        lookback_months=1,
        fallback="equal_factor_weight",
        negative_ic_weight=0.0,
    )

    assert result.loc[0, "factor_a_weight"] == 0.5
    assert result.loc[0, "factor_b_weight"] == 0.5


def test_build_rolling_ic_weights_can_fall_back_to_fixed_baseline() -> None:
    ic_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31"]),
            "factor_a_ic": [float("nan")],
            "factor_b_ic": [float("nan")],
        }
    )

    result = build_rolling_ic_weights(
        factor_ic_frame=ic_frame,
        factor_columns=["factor_a", "factor_b"],
        lookback_months=1,
        fallback="fixed_weight_baseline",
        negative_ic_weight=0.0,
        fallback_weights={"factor_a": 0.7, "factor_b": 0.3},
    )

    assert result.loc[0, "factor_a_weight"] == 0.7
    assert result.loc[0, "factor_b_weight"] == 0.3
