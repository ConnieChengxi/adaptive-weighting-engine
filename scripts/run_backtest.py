from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_weighting.backtest.engine import (
    build_equal_weight_benchmark_panel,
    build_portfolio_returns,
    compute_time_varying_transaction_costs,
    compute_turnover,
    prepare_panel,
    run_equal_weight_backtest,
    run_scored_backtest,
)
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

BACKTEST_CONFIG = ROOT / "config" / "backtest.yaml"
FIXED_WEIGHT_CONFIG = ROOT / "config" / "models" / "S1_static_equal_factor.yaml"
ROLLING_IC_CONFIG = ROOT / "config" / "models" / "A1_rolling_ic.yaml"
RIDGE_IC_CONFIG = ROOT / "config" / "models" / "L1_ridge_ic.yaml"
LASSO_IC_CONFIG = ROOT / "config" / "models" / "L2_lasso_ic.yaml"
ELASTIC_NET_IC_CONFIG = ROOT / "config" / "models" / "L3_elastic_net_ic.yaml"
RANDOM_FOREST_IC_CONFIG = ROOT / "config" / "models" / "T1_random_forest_ic.yaml"
XGBOOST_IC_CONFIG = ROOT / "config" / "models" / "T2_xgboost_ic.yaml"
PANEL_PATH = ROOT / "data" / "processed" / "monthly_factor_panel.csv"
OUTPUT_DIR = ROOT / "outputs" / "backtests"

FACTOR_COLUMNS = [
    "momentum_score_z",
    "liquidity_1m_z",
    "volatility_score_z",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_scored_panel(panel: pd.DataFrame, factor_weights: dict[str, float]) -> pd.DataFrame:
    scored = build_fixed_weight_score(panel, factor_weights)
    return scored


def build_factor_weight_mapping(factor_weights: dict[str, float]) -> dict[str, float]:
    return {
        "momentum_score_z": factor_weights["momentum"],
        "liquidity_1m_z": factor_weights["liquidity"],
        "volatility_score_z": factor_weights["volatility"],
    }


def resolve_prior_weights(model_cfg: dict, default_factor_weights: dict[str, float]) -> dict[str, float]:
    prior_weights = model_cfg["model"].get("prior_weights")
    if prior_weights:
        return prior_weights
    return build_factor_weight_mapping(default_factor_weights)


def resolve_shrinkage_config(rolling_ic_cfg: dict) -> tuple[float | None, float | None]:
    shrinkage_cfg = rolling_ic_cfg["model"].get("shrinkage", {})
    if not shrinkage_cfg.get("enabled", False):
        return None, None
    return shrinkage_cfg["ic_weight"], shrinkage_cfg["baseline_weight"]


def build_rolling_variant_specs(rolling_ic_cfg: dict) -> list[dict[str, float | str | None]]:
    specs: list[dict[str, float | str | None]] = []
    primary_ic_weight, primary_baseline_weight = resolve_shrinkage_config(rolling_ic_cfg)
    specs.append(
        {
            "name": "rolling_ic",
            "shrinkage_ic_weight": primary_ic_weight,
            "shrinkage_baseline_weight": primary_baseline_weight,
        }
    )

    robustness_cfg = rolling_ic_cfg["model"].get("robustness", {})
    if "no_shrinkage" in robustness_cfg:
        specs.append(
            {
                "name": "rolling_ic_no_shrinkage",
                "shrinkage_ic_weight": robustness_cfg["no_shrinkage"]["ic_weight"],
                "shrinkage_baseline_weight": robustness_cfg["no_shrinkage"]["baseline_weight"],
            }
        )
    if "shrinkage_60_40" in robustness_cfg:
        specs.append(
            {
                "name": "rolling_ic_shrinkage_60_40",
                "shrinkage_ic_weight": robustness_cfg["shrinkage_60_40"]["ic_weight"],
                "shrinkage_baseline_weight": robustness_cfg["shrinkage_60_40"]["baseline_weight"],
            }
        )
    return specs


def resolve_model_shrinkage_config(model_cfg: dict) -> tuple[float | None, float | None]:
    shrinkage_cfg = model_cfg["model"].get("shrinkage", {})
    if not shrinkage_cfg.get("enabled", False):
        return None, None
    return shrinkage_cfg["ic_weight"], shrinkage_cfg["baseline_weight"]


def apply_turnover_adjustment_if_enabled(weights_frame: pd.DataFrame, model_cfg: dict) -> pd.DataFrame:
    turnover_adjustment_cfg = model_cfg["model"].get("turnover_adjustment", {})
    if not turnover_adjustment_cfg.get("enabled", False):
        return weights_frame

    weight_columns = [column for column in weights_frame.columns if column.endswith("_weight")]
    if not weight_columns:
        return weights_frame

    return apply_post_model_turnover_adjustment(
        weights_frame=weights_frame,
        weight_columns=weight_columns,
        penalty_lambda=float(turnover_adjustment_cfg["lambda"]),
    )


def build_tradable_panel(scored: pd.DataFrame, extra_required_columns: list[str] | None = None) -> pd.DataFrame:
    required_columns = [
        "liquidity_1m_z",
        "volatility_score_z",
        "momentum_score_z",
        "next_month_return",
    ]
    if extra_required_columns:
        required_columns.extend(extra_required_columns)
    return scored.dropna(subset=required_columns).copy()


def run_model_backtest(
    tradable: pd.DataFrame,
    score_column: str,
    top_n: int,
    transaction_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return run_scored_backtest(
        tradable,
        score_column=score_column,
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework="top_n",
    )


def save_backtest_outputs(
    prefix: str,
    selected: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    metrics_frame: pd.DataFrame,
) -> None:
    portfolio_path = OUTPUT_DIR / f"{prefix}_portfolio_returns.csv"
    selections_path = OUTPUT_DIR / f"{prefix}_selections.csv"
    metrics_path = OUTPUT_DIR / f"{prefix}_metrics.csv"

    portfolio_returns.to_csv(portfolio_path, index=False)
    selected.to_csv(selections_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)

    print(f"Saved {portfolio_path.relative_to(ROOT)}")
    print(f"Saved {selections_path.relative_to(ROOT)}")
    print(f"Saved {metrics_path.relative_to(ROOT)}")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f'{value:.6f}'))


def save_model_comparison(metrics_by_model: dict[str, pd.DataFrame], filename: str) -> None:
    comparison = pd.concat(
        [frame.assign(model=model_name) for model_name, frame in metrics_by_model.items()],
        ignore_index=True,
    )
    ordered_columns = [
        "model",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "turnover",
        "net_return_after_costs",
    ]
    comparison = comparison[ordered_columns]
    comparison_path = OUTPUT_DIR / filename
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved {comparison_path.relative_to(ROOT)}")


def run_rolling_ic_variant(
    prefix: str,
    scored: pd.DataFrame,
    factor_ic_frame: pd.DataFrame,
    rolling_ic_cfg: dict,
    factor_weights: dict[str, float],
    top_n: int,
    transaction_cost_bps: float,
    shrinkage_ic_weight: float | None,
    shrinkage_baseline_weight: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weights_frame = build_rolling_ic_weights(
        factor_ic_frame=factor_ic_frame,
        factor_columns=FACTOR_COLUMNS,
        lookback_months=rolling_ic_cfg["model"]["lookback_months"],
        fallback=rolling_ic_cfg["model"]["fallback"],
        negative_ic_weight=rolling_ic_cfg["model"]["negative_ic_weight"],
        fallback_weights=resolve_prior_weights(rolling_ic_cfg, factor_weights),
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    weights_frame.to_csv(OUTPUT_DIR / f"{prefix}_raw_weight_history.csv", index=False)
    weights_frame = apply_turnover_adjustment_if_enabled(weights_frame, rolling_ic_cfg)
    weights_frame.to_csv(OUTPUT_DIR / f"{prefix}_weight_history.csv", index=False)

    rolling_scored = apply_rolling_ic_score(scored, weights_frame, FACTOR_COLUMNS)
    rolling_tradable = build_tradable_panel(
        rolling_scored,
        extra_required_columns=[
            "rolling_ic_score",
            "momentum_score_z_weight",
            "liquidity_1m_z_weight",
            "volatility_score_z_weight",
        ],
    )
    selected, portfolio_returns, metrics = run_model_backtest(
        rolling_tradable,
        score_column="rolling_ic_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
    )
    save_backtest_outputs(prefix, selected, portfolio_returns, metrics)
    return selected, portfolio_returns, metrics


def run_legacy_artifact_export() -> None:
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

    panel = prepare_panel(PANEL_PATH)
    scored = prepare_scored_panel(panel, factor_weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed_weight_tradable = build_tradable_panel(scored)
    equal_weight_selected, equal_weight_portfolio_returns, equal_weight_metrics = run_equal_weight_backtest(
        fixed_weight_tradable,
        transaction_cost_bps=transaction_cost_bps,
    )
    save_backtest_outputs(
        "equal_weight_benchmark",
        equal_weight_selected,
        equal_weight_portfolio_returns,
        equal_weight_metrics,
    )

    fixed_selected, fixed_portfolio_returns, fixed_metrics = run_model_backtest(
        fixed_weight_tradable,
        score_column="fixed_weight_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
    )
    save_backtest_outputs("fixed_weight", fixed_selected, fixed_portfolio_returns, fixed_metrics)

    factor_ic_frame = compute_monthly_factor_ic(scored, FACTOR_COLUMNS, "next_month_return")
    factor_ic_frame.to_csv(OUTPUT_DIR / "rolling_ic_factor_history.csv", index=False)

    rolling_metrics_by_model: dict[str, pd.DataFrame] = {}
    for spec in build_rolling_variant_specs(rolling_ic_cfg):
        _, _, metrics = run_rolling_ic_variant(
            prefix=str(spec["name"]),
            scored=scored,
            factor_ic_frame=factor_ic_frame,
            rolling_ic_cfg=rolling_ic_cfg,
            factor_weights=factor_weights,
            top_n=top_n,
            transaction_cost_bps=transaction_cost_bps,
            shrinkage_ic_weight=spec["shrinkage_ic_weight"],
            shrinkage_baseline_weight=spec["shrinkage_baseline_weight"],
        )
        rolling_metrics_by_model[str(spec["name"])] = metrics

    save_model_comparison(
        {
            "equal_weight_benchmark": equal_weight_metrics,
            "fixed_weight": fixed_metrics,
            "rolling_ic": rolling_metrics_by_model["rolling_ic"],
        },
        filename="model_comparison_metrics.csv",
    )
    save_model_comparison(
        {
            "equal_weight_benchmark": equal_weight_metrics,
            "fixed_weight": fixed_metrics,
            "rolling_ic": rolling_metrics_by_model["rolling_ic"],
            "rolling_ic_no_shrinkage": rolling_metrics_by_model["rolling_ic_no_shrinkage"],
            "rolling_ic_60_40": rolling_metrics_by_model["rolling_ic_shrinkage_60_40"],
        },
        filename="rolling_ic_robustness_comparison.csv",
    )

    model_feature_cfg = elastic_net_ic_cfg["model"]["features"]
    xgboost_feature_frame, xgboost_feature_columns = build_xgboost_ic_feature_frame(
        panel=scored,
        factor_ic_frame=factor_ic_frame,
        factor_columns=FACTOR_COLUMNS,
        ic_lag_months=model_feature_cfg["ic_lag_months"],
        ic_rolling_means=model_feature_cfg["ic_rolling_means"],
    )
    xgboost_feature_frame.to_csv(OUTPUT_DIR / "xgboost_ic_feature_frame.csv", index=False)

    linear_model_cfgs = {
        "ridge_ic": ridge_ic_cfg,
        "lasso_ic": lasso_ic_cfg,
        "elastic_net_ic": elastic_net_ic_cfg,
    }
    linear_model_metrics: dict[str, pd.DataFrame] = {}
    for output_name, model_cfg in linear_model_cfgs.items():
        linear_estimator_cfg = model_cfg["model"]["estimator"].copy()
        linear_estimator_type = str(linear_estimator_cfg.pop("type"))
        linear_estimator_cfg.pop("family", None)
        prediction_frame = predict_factor_ic_with_linear_model(
            feature_frame=xgboost_feature_frame,
            factor_columns=FACTOR_COLUMNS,
            feature_columns=xgboost_feature_columns,
            training_window_months=model_cfg["model"]["training_window_months"],
            min_training_rows=model_cfg["model"]["min_training_rows"],
            estimator_type=linear_estimator_type,
            estimator_params=linear_estimator_cfg,
        )
        prediction_frame.to_csv(OUTPUT_DIR / f"{output_name}_predictions.csv", index=False)

        weights = build_predicted_ic_weights(
            prediction_frame=prediction_frame,
            factor_columns=FACTOR_COLUMNS,
            negative_prediction_weight=model_cfg["model"]["negative_prediction_weight"],
            fallback=model_cfg["model"]["fallback"],
            fallback_weights=resolve_prior_weights(model_cfg, factor_weights),
            shrinkage_ic_weight=resolve_model_shrinkage_config(model_cfg)[0],
            shrinkage_baseline_weight=resolve_model_shrinkage_config(model_cfg)[1],
        )
        weights.to_csv(OUTPUT_DIR / f"{output_name}_raw_weight_history.csv", index=False)
        weights = apply_turnover_adjustment_if_enabled(weights, model_cfg)
        weights.to_csv(OUTPUT_DIR / f"{output_name}_weight_history.csv", index=False)

        model_scored = apply_predicted_ic_score(scored, weights, FACTOR_COLUMNS)
        model_tradable = build_tradable_panel(
            model_scored,
            extra_required_columns=[
                "xgboost_ic_score",
                "momentum_score_z_weight",
                "liquidity_1m_z_weight",
                "volatility_score_z_weight",
            ],
        )
        selected, portfolio_returns, metrics = run_model_backtest(
            model_tradable,
            score_column="xgboost_ic_score",
            top_n=top_n,
            transaction_cost_bps=transaction_cost_bps,
        )
        save_backtest_outputs(output_name, selected, portfolio_returns, metrics)
        linear_model_metrics[output_name] = metrics

    random_forest_estimator_cfg = random_forest_ic_cfg["model"]["estimator"].copy()
    random_forest_estimator_cfg.pop("family", None)
    random_forest_estimator_cfg.pop("type", None)
    random_forest_prediction_frame = predict_factor_ic_with_random_forest(
        feature_frame=xgboost_feature_frame,
        factor_columns=FACTOR_COLUMNS,
        feature_columns=xgboost_feature_columns,
        training_window_months=random_forest_ic_cfg["model"]["training_window_months"],
        min_training_rows=random_forest_ic_cfg["model"]["min_training_rows"],
        estimator_params=random_forest_estimator_cfg,
    )
    random_forest_prediction_frame.to_csv(OUTPUT_DIR / "random_forest_ic_predictions.csv", index=False)

    random_forest_weights = build_predicted_ic_weights(
        prediction_frame=random_forest_prediction_frame,
        factor_columns=FACTOR_COLUMNS,
        negative_prediction_weight=random_forest_ic_cfg["model"]["negative_prediction_weight"],
        fallback=random_forest_ic_cfg["model"]["fallback"],
        fallback_weights=resolve_prior_weights(random_forest_ic_cfg, factor_weights),
        shrinkage_ic_weight=resolve_model_shrinkage_config(random_forest_ic_cfg)[0],
        shrinkage_baseline_weight=resolve_model_shrinkage_config(random_forest_ic_cfg)[1],
    )
    random_forest_weights.to_csv(OUTPUT_DIR / "random_forest_ic_raw_weight_history.csv", index=False)
    random_forest_weights = apply_turnover_adjustment_if_enabled(random_forest_weights, random_forest_ic_cfg)
    random_forest_weights.to_csv(OUTPUT_DIR / "random_forest_ic_weight_history.csv", index=False)

    random_forest_scored = apply_predicted_ic_score(scored, random_forest_weights, FACTOR_COLUMNS)
    random_forest_tradable = build_tradable_panel(
        random_forest_scored,
        extra_required_columns=[
            "xgboost_ic_score",
            "momentum_score_z_weight",
            "liquidity_1m_z_weight",
            "volatility_score_z_weight",
        ],
    )
    random_forest_selected, random_forest_portfolio_returns, random_forest_metrics = run_model_backtest(
        random_forest_tradable,
        score_column="xgboost_ic_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
    )
    save_backtest_outputs(
        "random_forest_ic",
        random_forest_selected,
        random_forest_portfolio_returns,
        random_forest_metrics,
    )

    xgboost_prediction_frame = predict_factor_ic_with_xgboost(
        feature_frame=xgboost_feature_frame,
        factor_columns=FACTOR_COLUMNS,
        feature_columns=xgboost_feature_columns,
        training_window_months=xgboost_ic_cfg["model"]["training_window_months"],
        min_training_rows=xgboost_ic_cfg["model"]["min_training_rows"],
        xgb_params=xgboost_ic_cfg["model"]["xgb_params"],
    )
    xgboost_prediction_frame.to_csv(OUTPUT_DIR / "xgboost_ic_predictions.csv", index=False)

    xgboost_weights = build_predicted_ic_weights(
        prediction_frame=xgboost_prediction_frame,
        factor_columns=FACTOR_COLUMNS,
        negative_prediction_weight=xgboost_ic_cfg["model"]["negative_prediction_weight"],
        fallback=xgboost_ic_cfg["model"]["fallback"],
        fallback_weights=resolve_prior_weights(xgboost_ic_cfg, factor_weights),
        shrinkage_ic_weight=resolve_model_shrinkage_config(xgboost_ic_cfg)[0],
        shrinkage_baseline_weight=resolve_model_shrinkage_config(xgboost_ic_cfg)[1],
    )
    xgboost_weights.to_csv(OUTPUT_DIR / "xgboost_ic_raw_weight_history.csv", index=False)
    xgboost_weights = apply_turnover_adjustment_if_enabled(xgboost_weights, xgboost_ic_cfg)
    xgboost_weights.to_csv(OUTPUT_DIR / "xgboost_ic_weight_history.csv", index=False)

    xgboost_scored = apply_predicted_ic_score(scored, xgboost_weights, FACTOR_COLUMNS)
    xgboost_tradable = build_tradable_panel(
        xgboost_scored,
        extra_required_columns=[
            "xgboost_ic_score",
            "momentum_score_z_weight",
            "liquidity_1m_z_weight",
            "volatility_score_z_weight",
        ],
    )
    xgboost_selected, xgboost_portfolio_returns, xgboost_metrics = run_model_backtest(
        xgboost_tradable,
        score_column="xgboost_ic_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
    )
    save_backtest_outputs("xgboost_ic", xgboost_selected, xgboost_portfolio_returns, xgboost_metrics)

    save_model_comparison(
        {
            "equal_weight_benchmark": equal_weight_metrics,
            "fixed_weight": fixed_metrics,
            "rolling_ic": rolling_metrics_by_model["rolling_ic"],
            "ridge_ic": linear_model_metrics["ridge_ic"],
            "lasso_ic": linear_model_metrics["lasso_ic"],
            "elastic_net_ic": linear_model_metrics["elastic_net_ic"],
            "random_forest_ic": random_forest_metrics,
            "xgboost_ic": xgboost_metrics,
        },
        filename="full_model_comparison_metrics.csv",
    )

    save_model_comparison(
        {
            "ridge_ic": linear_model_metrics["ridge_ic"],
            "lasso_ic": linear_model_metrics["lasso_ic"],
            "elastic_net_ic": linear_model_metrics["elastic_net_ic"],
            "random_forest_ic": random_forest_metrics,
            "xgboost_ic": xgboost_metrics,
        },
        filename="ml_model_robustness_comparison.csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility runner for the legacy generic backtest export layer."
    )
    parser.add_argument(
        "--legacy-artifacts",
        action="store_true",
        help="Explicitly generate the legacy generic backtest artifacts. The canonical dissertation pipeline no longer requires these files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.legacy_artifacts:
        print(
            "run_backtest.py no longer writes the generic backtest layer by default.\n"
            "Use the canonical dissertation pipeline instead.\n"
            "From a clean checkout, run:\n"
            "  1. python3 scripts/download_data.py\n"
            "  2. python3 scripts/build_features.py\n"
            "  3. python3 scripts/generate_momentum_diagnostics.py\n"
            "  4. python3 scripts/generate_liquidity_diagnostics.py\n"
            "  5. python3 scripts/generate_volatility_diagnostics.py\n"
            "  6. python3 scripts/run_common_shrinkage_selection.py\n"
            "  7. python3 scripts/run_repeated_walkforward_family_comparison.py\n"
            "  8. python3 scripts/run_turnover_framework_backtests.py\n"
            "  9. python3 scripts/generate_factor_contribution_assets.py\n"
            " 10. python3 scripts/generate_liquidity_cost_exploratory.py\n"
            " 11. python3 scripts/generate_report_assets.py\n\n"
            "If you need the deprecated generic artifacts temporarily, rerun with:\n"
            "  python3 scripts/run_backtest.py --legacy-artifacts"
        )
        return

    print("Generating deprecated generic backtest artifacts for compatibility...")
    run_legacy_artifact_export()


if __name__ == "__main__":
    main()
