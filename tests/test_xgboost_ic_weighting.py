import pandas as pd

from adaptive_weighting.models.xgboost_ic_weighting import build_predicted_ic_weights


def test_build_predicted_ic_weights_can_fall_back_to_fixed_baseline() -> None:
    prediction_frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31"]),
            "factor_a_predicted_ic": [float("nan")],
            "factor_b_predicted_ic": [float("nan")],
        }
    )

    result = build_predicted_ic_weights(
        prediction_frame=prediction_frame,
        factor_columns=["factor_a", "factor_b"],
        negative_prediction_weight=0.0,
        fallback="fixed_weight_baseline",
        fallback_weights={"factor_a": 0.6, "factor_b": 0.4},
    )

    assert result.loc[0, "factor_a_weight"] == 0.6
    assert result.loc[0, "factor_b_weight"] == 0.4
