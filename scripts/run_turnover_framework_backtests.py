from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_weighting.backtest.engine import run_scored_backtest
from adaptive_weighting.ic.compute_ic import compute_monthly_factor_ic
from adaptive_weighting.models.fixed_weight import build_fixed_weight_score
from adaptive_weighting.models.linear_ic_weighting import predict_factor_ic_with_linear_model
from adaptive_weighting.models.rolling_ic_weighting import apply_rolling_ic_score, build_rolling_ic_weights
from adaptive_weighting.models.turnover_adjustment import apply_post_model_turnover_adjustment
from adaptive_weighting.models.tree_ic_weighting import predict_factor_ic_with_random_forest
from adaptive_weighting.models.xgboost_ic_weighting import (
    apply_predicted_ic_score,
    build_predicted_ic_weights,
    build_xgboost_ic_feature_frame,
    predict_factor_ic_with_xgboost,
)
from scripts.run_common_shrinkage_selection import (
    CORE3_FACTOR_COLUMNS,
    build_trial_panel,
    build_trial_tradable_panel,
)
from scripts.run_backtest import (
    BACKTEST_CONFIG,
    FIXED_WEIGHT_CONFIG,
    ROLLING_IC_CONFIG,
    RIDGE_IC_CONFIG,
    LASSO_IC_CONFIG,
    ELASTIC_NET_IC_CONFIG,
    RANDOM_FOREST_IC_CONFIG,
    XGBOOST_IC_CONFIG,
    OUTPUT_DIR,
    build_tradable_panel,
    load_yaml,
    resolve_shrinkage_config,
    resolve_model_shrinkage_config,
    resolve_prior_weights,
    run_equal_weight_backtest,
    save_backtest_outputs,
)


BASE_FRAMEWORKS = ("baseline", "pta")
HOLDING_BUFFER_RANKS = (4, 5, 6)
HOLDING_BUFFER_FRAMEWORKS = tuple(f"holding_buffer_top{rank}" for rank in HOLDING_BUFFER_RANKS)
FRAMEWORKS = BASE_FRAMEWORKS + HOLDING_BUFFER_FRAMEWORKS


def holding_buffer_rank_for_framework(framework: str) -> int | None:
    if not framework.startswith("holding_buffer_top"):
        return None
    suffix = framework.removeprefix("holding_buffer_top")
    return int(suffix) if suffix.isdigit() else None


def apply_turnover_framework(weights_frame: pd.DataFrame, model_cfg: dict, framework: str) -> pd.DataFrame:
    if framework != "pta":
        return weights_frame.copy()

    turnover_adjustment_cfg = model_cfg["model"].get("turnover_adjustment", {})
    if not turnover_adjustment_cfg.get("enabled", False):
        return weights_frame.copy()

    weight_columns = [column for column in weights_frame.columns if column.endswith("_weight")]
    return apply_post_model_turnover_adjustment(
        weights_frame=weights_frame.copy(),
        weight_columns=weight_columns,
        penalty_lambda=float(turnover_adjustment_cfg["lambda"]),
    )


def prefixed_name(model_name: str, framework: str) -> str:
    return f"{framework}_{model_name}"


def save_framework_long(
    records: list[dict[str, float | str]],
    filename: str,
) -> None:
    pd.DataFrame(records).to_csv(OUTPUT_DIR / filename, index=False)


def run_model_backtest_under_framework(
    tradable: pd.DataFrame,
    score_column: str,
    top_n: int,
    transaction_cost_bps: float,
    framework: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_scored_backtest(
        tradable,
        score_column=score_column,
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=framework,
    )


def main() -> None:
    backtest_cfg = load_yaml(BACKTEST_CONFIG)
    fixed_weight_cfg = load_yaml(FIXED_WEIGHT_CONFIG)
    rolling_ic_cfg = load_yaml(ROLLING_IC_CONFIG)
    ridge_ic_cfg = load_yaml(RIDGE_IC_CONFIG)
    lasso_ic_cfg = load_yaml(LASSO_IC_CONFIG)
    elastic_net_ic_cfg = load_yaml(ELASTIC_NET_IC_CONFIG)
    random_forest_ic_cfg = load_yaml(RANDOM_FOREST_IC_CONFIG)
    xgboost_ic_cfg = load_yaml(XGBOOST_IC_CONFIG)

    top_n = backtest_cfg["portfolio"]["top_n"]
    transaction_cost_bps = backtest_cfg["costs"]["transaction_cost_bps"]
    factor_weights = fixed_weight_cfg["model"]["factors"]

    panel = build_trial_panel()
    scored = panel.copy()
    tradable = build_trial_tradable_panel(scored)
    factor_ic_frame = compute_monthly_factor_ic(scored, CORE3_FACTOR_COLUMNS, "next_month_return")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eq_selected, eq_returns, eq_metrics = run_equal_weight_backtest(tradable, transaction_cost_bps)
    framework_metrics: dict[str, dict[str, pd.DataFrame]] = {framework: {} for framework in FRAMEWORKS}
    for framework in FRAMEWORKS:
        save_backtest_outputs(prefixed_name("equal_weight_benchmark", framework), eq_selected, eq_returns, eq_metrics)
        framework_metrics[framework]["equal_weight_benchmark"] = eq_metrics

        fixed_selected, fixed_returns, fixed_metrics = run_model_backtest_under_framework(
            tradable,
            "fixed_weight_score",
            top_n,
            transaction_cost_bps,
            framework,
        )
        save_backtest_outputs(prefixed_name("fixed_weight", framework), fixed_selected, fixed_returns, fixed_metrics)
        framework_metrics[framework]["fixed_weight"] = fixed_metrics

    rolling_shrinkage_ic_weight, rolling_shrinkage_baseline_weight = resolve_shrinkage_config(rolling_ic_cfg)
    rolling_raw_weights = build_rolling_ic_weights(
        factor_ic_frame=factor_ic_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        lookback_months=rolling_ic_cfg["model"]["lookback_months"],
        fallback=rolling_ic_cfg["model"]["fallback"],
        negative_ic_weight=rolling_ic_cfg["model"]["negative_ic_weight"],
        fallback_weights=resolve_prior_weights(rolling_ic_cfg, factor_weights),
        shrinkage_ic_weight=rolling_shrinkage_ic_weight,
        shrinkage_baseline_weight=rolling_shrinkage_baseline_weight,
    )

    for framework in FRAMEWORKS:
        weights = apply_turnover_framework(rolling_raw_weights, rolling_ic_cfg, framework)
        weights.to_csv(OUTPUT_DIR / f"{prefixed_name('rolling_ic', framework)}_weight_history.csv", index=False)
        model_scored = apply_rolling_ic_score(scored, weights, CORE3_FACTOR_COLUMNS)
        model_tradable = build_trial_tradable_panel(
            model_scored,
            extra_required_columns=[
                "rolling_ic_score",
                "momentum_score_z_weight",
                "liquidity_1m_z_weight",
                "volatility_score_z_weight",
            ],
        )
        selected, portfolio_returns, metrics = run_model_backtest_under_framework(
            model_tradable,
            "rolling_ic_score",
            top_n,
            transaction_cost_bps,
            framework,
        )
        save_backtest_outputs(prefixed_name("rolling_ic", framework), selected, portfolio_returns, metrics)
        framework_metrics[framework]["rolling_ic"] = metrics

    feature_cfg = elastic_net_ic_cfg["model"]["features"]
    feature_frame, feature_columns = build_xgboost_ic_feature_frame(
        panel=scored,
        factor_ic_frame=factor_ic_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        ic_lag_months=feature_cfg["ic_lag_months"],
        ic_rolling_means=feature_cfg["ic_rolling_means"],
    )

    linear_model_cfgs = {
        "ridge_ic": ridge_ic_cfg,
        "lasso_ic": lasso_ic_cfg,
        "elastic_net_ic": elastic_net_ic_cfg,
    }

    for model_name, model_cfg in linear_model_cfgs.items():
        estimator_cfg = model_cfg["model"]["estimator"].copy()
        estimator_type = str(estimator_cfg.pop("type"))
        estimator_cfg.pop("family", None)
        prediction_frame = predict_factor_ic_with_linear_model(
            feature_frame=feature_frame,
            factor_columns=CORE3_FACTOR_COLUMNS,
            feature_columns=feature_columns,
            training_window_months=model_cfg["model"]["training_window_months"],
            min_training_rows=model_cfg["model"]["min_training_rows"],
            estimator_type=estimator_type,
            estimator_params=estimator_cfg,
        )
        raw_weights = build_predicted_ic_weights(
            prediction_frame=prediction_frame,
            factor_columns=CORE3_FACTOR_COLUMNS,
            negative_prediction_weight=model_cfg["model"]["negative_prediction_weight"],
            fallback=model_cfg["model"]["fallback"],
            fallback_weights=resolve_prior_weights(model_cfg, factor_weights),
            shrinkage_ic_weight=resolve_model_shrinkage_config(model_cfg)[0],
            shrinkage_baseline_weight=resolve_model_shrinkage_config(model_cfg)[1],
        )

        for framework in FRAMEWORKS:
            weights = apply_turnover_framework(raw_weights, model_cfg, framework)
            weights.to_csv(OUTPUT_DIR / f"{prefixed_name(model_name, framework)}_weight_history.csv", index=False)
            model_scored = apply_predicted_ic_score(scored, weights, CORE3_FACTOR_COLUMNS)
            model_tradable = build_trial_tradable_panel(
                model_scored,
                extra_required_columns=[
                    "xgboost_ic_score",
                    "momentum_score_z_weight",
                    "liquidity_1m_z_weight",
                    "volatility_score_z_weight",
                ],
            )
            selected, portfolio_returns, metrics = run_model_backtest_under_framework(
                model_tradable,
                "xgboost_ic_score",
                top_n,
                transaction_cost_bps,
                framework,
            )
            save_backtest_outputs(prefixed_name(model_name, framework), selected, portfolio_returns, metrics)
            framework_metrics[framework][model_name] = metrics

    rf_estimator_cfg = random_forest_ic_cfg["model"]["estimator"].copy()
    rf_estimator_cfg.pop("family", None)
    rf_estimator_cfg.pop("type", None)
    rf_prediction_frame = predict_factor_ic_with_random_forest(
        feature_frame=feature_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        feature_columns=feature_columns,
        training_window_months=random_forest_ic_cfg["model"]["training_window_months"],
        min_training_rows=random_forest_ic_cfg["model"]["min_training_rows"],
        estimator_params=rf_estimator_cfg,
    )
    rf_raw_weights = build_predicted_ic_weights(
        prediction_frame=rf_prediction_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        negative_prediction_weight=random_forest_ic_cfg["model"]["negative_prediction_weight"],
        fallback=random_forest_ic_cfg["model"]["fallback"],
        fallback_weights=resolve_prior_weights(random_forest_ic_cfg, factor_weights),
        shrinkage_ic_weight=resolve_model_shrinkage_config(random_forest_ic_cfg)[0],
        shrinkage_baseline_weight=resolve_model_shrinkage_config(random_forest_ic_cfg)[1],
    )

    for framework in FRAMEWORKS:
        weights = apply_turnover_framework(rf_raw_weights, random_forest_ic_cfg, framework)
        weights.to_csv(OUTPUT_DIR / f"{prefixed_name('random_forest_ic', framework)}_weight_history.csv", index=False)
        model_scored = apply_predicted_ic_score(scored, weights, CORE3_FACTOR_COLUMNS)
        model_tradable = build_trial_tradable_panel(
            model_scored,
            extra_required_columns=[
                "xgboost_ic_score",
                "momentum_score_z_weight",
                "liquidity_1m_z_weight",
                "volatility_score_z_weight",
            ],
        )
        selected, portfolio_returns, metrics = run_model_backtest_under_framework(
            model_tradable,
            "xgboost_ic_score",
            top_n,
            transaction_cost_bps,
            framework,
        )
        save_backtest_outputs(prefixed_name("random_forest_ic", framework), selected, portfolio_returns, metrics)
        framework_metrics[framework]["random_forest_ic"] = metrics

    xgb_prediction_frame = predict_factor_ic_with_xgboost(
        feature_frame=feature_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        feature_columns=feature_columns,
        training_window_months=xgboost_ic_cfg["model"]["training_window_months"],
        min_training_rows=xgboost_ic_cfg["model"]["min_training_rows"],
        xgb_params=xgboost_ic_cfg["model"]["xgb_params"],
    )
    xgb_raw_weights = build_predicted_ic_weights(
        prediction_frame=xgb_prediction_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        negative_prediction_weight=xgboost_ic_cfg["model"]["negative_prediction_weight"],
        fallback=xgboost_ic_cfg["model"]["fallback"],
        fallback_weights=resolve_prior_weights(xgboost_ic_cfg, factor_weights),
        shrinkage_ic_weight=resolve_model_shrinkage_config(xgboost_ic_cfg)[0],
        shrinkage_baseline_weight=resolve_model_shrinkage_config(xgboost_ic_cfg)[1],
    )

    for framework in FRAMEWORKS:
        weights = apply_turnover_framework(xgb_raw_weights, xgboost_ic_cfg, framework)
        weights.to_csv(OUTPUT_DIR / f"{prefixed_name('xgboost_ic', framework)}_weight_history.csv", index=False)
        model_scored = apply_predicted_ic_score(scored, weights, CORE3_FACTOR_COLUMNS)
        model_tradable = build_trial_tradable_panel(
            model_scored,
            extra_required_columns=[
                "xgboost_ic_score",
                "momentum_score_z_weight",
                "liquidity_1m_z_weight",
                "volatility_score_z_weight",
            ],
        )
        selected, portfolio_returns, metrics = run_model_backtest_under_framework(
            model_tradable,
            "xgboost_ic_score",
            top_n,
            transaction_cost_bps,
            framework,
        )
        save_backtest_outputs(prefixed_name("xgboost_ic", framework), selected, portfolio_returns, metrics)
        framework_metrics[framework]["xgboost_ic"] = metrics

    framework_orders: dict[str, dict[str, pd.DataFrame]] = {}

    framework_orders["baseline"] = {
        "equal_weight_benchmark": eq_metrics,
        "fixed_weight": framework_metrics["baseline"]["fixed_weight"],
        "rolling_ic": framework_metrics["baseline"]["rolling_ic"],
        "ridge_ic": framework_metrics["baseline"]["ridge_ic"],
        "lasso_ic": framework_metrics["baseline"]["lasso_ic"],
        "elastic_net_ic": framework_metrics["baseline"]["elastic_net_ic"],
        "random_forest_ic": framework_metrics["baseline"]["random_forest_ic"],
        "xgboost_ic": framework_metrics["baseline"]["xgboost_ic"],
    }
    framework_orders["pta"] = {
        "equal_weight_benchmark": eq_metrics,
        "fixed_weight": framework_metrics["pta"]["fixed_weight"],
        "rolling_ic": framework_metrics["pta"]["rolling_ic"],
        "ridge_ic": framework_metrics["pta"]["ridge_ic"],
        "lasso_ic": framework_metrics["pta"]["lasso_ic"],
        "elastic_net_ic": framework_metrics["pta"]["elastic_net_ic"],
        "random_forest_ic": framework_metrics["pta"]["random_forest_ic"],
        "xgboost_ic": framework_metrics["pta"]["xgboost_ic"],
    }
    for framework in HOLDING_BUFFER_FRAMEWORKS:
        framework_orders[framework] = {
            "equal_weight_benchmark": eq_metrics,
            "fixed_weight": framework_metrics[framework]["fixed_weight"],
            "rolling_ic": framework_metrics[framework]["rolling_ic"],
            "ridge_ic": framework_metrics[framework]["ridge_ic"],
            "lasso_ic": framework_metrics[framework]["lasso_ic"],
            "elastic_net_ic": framework_metrics[framework]["elastic_net_ic"],
            "random_forest_ic": framework_metrics[framework]["random_forest_ic"],
            "xgboost_ic": framework_metrics[framework]["xgboost_ic"],
        }

    long_records: list[dict[str, float | str]] = []
    for framework_name, model_map in framework_orders.items():
        for model_name, frame in model_map.items():
            row = frame.iloc[0].to_dict()
            row["framework"] = framework_name
            row["model"] = model_name
            long_records.append(row)
    save_framework_long(long_records, "framework_comparison_metrics.csv")

    sensitivity_records: list[dict[str, float | str]] = []
    for framework_name in HOLDING_BUFFER_FRAMEWORKS:
        hold_buffer_rank = holding_buffer_rank_for_framework(framework_name)
        assert hold_buffer_rank is not None
        for model_name, frame in framework_orders[framework_name].items():
            row = frame.iloc[0].to_dict()
            row["framework"] = framework_name
            row["holding_buffer_rank"] = hold_buffer_rank
            row["model"] = model_name
            sensitivity_records.append(row)
    save_framework_long(sensitivity_records, "holding_buffer_sensitivity_metrics.csv")


if __name__ == "__main__":
    main()
