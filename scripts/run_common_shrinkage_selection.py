from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_liquidity_diagnostics import build_master_panel as build_liquidity_panel  # type: ignore
from generate_momentum_diagnostics import build_master_panel as build_momentum_panel  # type: ignore
from generate_volatility_diagnostics import build_master_panel as build_volatility_panel  # type: ignore
from adaptive_weighting.backtest.engine import prepare_panel, run_equal_weight_backtest, run_scored_backtest
from run_backtest import (  # type: ignore
    BACKTEST_CONFIG,
    ELASTIC_NET_IC_CONFIG,
    LASSO_IC_CONFIG,
    PANEL_PATH,
    RANDOM_FOREST_IC_CONFIG,
    RIDGE_IC_CONFIG,
    ROLLING_IC_CONFIG,
    XGBOOST_IC_CONFIG,
    apply_predicted_ic_score,
    build_xgboost_ic_feature_frame,
    compute_monthly_factor_ic,
    load_yaml,
    predict_factor_ic_with_linear_model,
    predict_factor_ic_with_random_forest,
    predict_factor_ic_with_xgboost,
    save_backtest_outputs,
)
from adaptive_weighting.backtest.evaluation import summarize_performance
from adaptive_weighting.models.rolling_ic_weighting import apply_rolling_ic_score, build_rolling_ic_weights
from adaptive_weighting.models.xgboost_ic_weighting import build_predicted_ic_weights


OUTPUT_DIR = ROOT / "outputs" / "backtests"
TABLES_DIR = ROOT / "outputs" / "tables"
MAIN_FRAMEWORK = "holding_buffer_top6"

CORE3_FACTOR_COLUMNS = [
    "momentum_score_z",
    "liquidity_1m_z",
    "volatility_score_z",
]
COMMON_SHRINKAGE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


@dataclass(frozen=True)
class SplitWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_validation: int
    n_test: int


def build_trial_panel() -> pd.DataFrame:
    base_panel = prepare_panel(PANEL_PATH)
    momentum_panel = build_momentum_panel()[["Date", "symbol", "residual_momentum_12_1_z"]].copy()
    liquidity_panel = build_liquidity_panel()[["Date", "symbol", "amihud_illiquidity_winsorized_z"]].copy()
    volatility_panel = build_volatility_panel()[["Date", "symbol", "idio_volatility_score_z"]].copy()

    panel = base_panel.merge(momentum_panel, on=["Date", "symbol"], how="left")
    panel = panel.merge(liquidity_panel, on=["Date", "symbol"], how="left")
    panel = panel.merge(volatility_panel, on=["Date", "symbol"], how="left")
    panel["momentum_score_z"] = panel["residual_momentum_12_1_z"]
    panel["liquidity_1m_z"] = panel["amihud_illiquidity_winsorized_z"]
    panel["volatility_score_z"] = panel["idio_volatility_score_z"]
    panel["fixed_weight_score"] = panel[CORE3_FACTOR_COLUMNS].mean(axis=1)
    return panel


def build_trial_tradable_panel(scored: pd.DataFrame, extra_required_columns: list[str] | None = None) -> pd.DataFrame:
    required_columns = [
        "momentum_score_z",
        "liquidity_1m_z",
        "volatility_score_z",
        "next_month_return",
    ]
    if extra_required_columns:
        required_columns.extend(extra_required_columns)
    return scored.dropna(subset=required_columns).copy()


def compute_common_window(model_returns: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for frame in model_returns.values():
        dates = pd.to_datetime(frame["Date"])
        starts.append(dates.min())
        ends.append(dates.max())
    return max(starts), min(ends)


def build_split_window(start: pd.Timestamp, end: pd.Timestamp) -> SplitWindow:
    dates = pd.date_range(start=start, end=end, freq="ME")
    n_total = len(dates)
    n_train = round(n_total * 0.6)
    n_validation = round(n_total * 0.2)
    n_test = n_total - n_train - n_validation
    return SplitWindow(
        train_start=dates[0],
        train_end=dates[n_train - 1],
        validation_start=dates[n_train],
        validation_end=dates[n_train + n_validation - 1],
        test_start=dates[n_train + n_validation],
        test_end=dates[-1],
        n_train=n_train,
        n_validation=n_validation,
        n_test=n_test,
    )


def summarize_slice(portfolio_returns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    window = portfolio_returns.copy()
    window["Date"] = pd.to_datetime(window["Date"])
    window = window[(window["Date"] >= start) & (window["Date"] <= end)].copy()
    if window.empty:
        return {
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "calmar_ratio": np.nan,
            "turnover": np.nan,
            "net_return_after_costs": np.nan,
            "average_transaction_cost_rate": np.nan,
        }
    return summarize_performance(
        return_series=window["portfolio_return"],
        turnover_series=window["turnover"],
        transaction_cost_rate_series=window["transaction_cost_rate"],
    )


def compute_net_sharpe(portfolio_returns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    window = portfolio_returns.copy()
    window["Date"] = pd.to_datetime(window["Date"])
    window = window[(window["Date"] >= start) & (window["Date"] <= end)].copy()
    if window.empty:
        return np.nan
    net_returns = window["net_portfolio_return"].fillna(0.0)
    if len(net_returns) <= 1:
        return np.nan
    annualized_return = (1.0 + net_returns).prod() ** (12.0 / len(net_returns)) - 1.0
    annualized_volatility = net_returns.std(ddof=1) * np.sqrt(12.0)
    if annualized_volatility == 0 or np.isnan(annualized_volatility):
        return np.nan
    return float(annualized_return / annualized_volatility)


def run_model_backtest_under_framework(
    tradable: pd.DataFrame,
    score_column: str,
    top_n: int,
    transaction_cost_bps: float,
    framework: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected, portfolio_returns, metrics = run_scored_backtest(
        tradable,
        score_column=score_column,
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=framework,
    )
    return selected, portfolio_returns, metrics


def run_fullsample_core3_models(
    scored: pd.DataFrame,
    top_n: int,
    transaction_cost_bps: float,
    baseline_weight: float,
    factor_ic_frame: pd.DataFrame,
    rolling_ic_cfg: dict,
    linear_prediction_frames: dict[str, pd.DataFrame],
    linear_model_cfgs: dict[str, dict],
    random_forest_prediction_frame: pd.DataFrame,
    random_forest_cfg: dict,
    xgboost_prediction_frame: pd.DataFrame,
    xgboost_cfg: dict,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    shrinkage_ic_weight = 1.0 - baseline_weight
    shrinkage_baseline_weight = baseline_weight
    fallback_weights = {factor: 1.0 / len(CORE3_FACTOR_COLUMNS) for factor in CORE3_FACTOR_COLUMNS}

    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}

    weights_frame = build_rolling_ic_weights(
        factor_ic_frame=factor_ic_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        lookback_months=rolling_ic_cfg["model"]["lookback_months"],
        fallback=rolling_ic_cfg["model"]["fallback"],
        negative_ic_weight=rolling_ic_cfg["model"]["negative_ic_weight"],
        fallback_weights=fallback_weights,
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    rolling_scored = apply_rolling_ic_score(scored, weights_frame, CORE3_FACTOR_COLUMNS)
    rolling_tradable = build_trial_tradable_panel(
        rolling_scored,
        extra_required_columns=["rolling_ic_score", "momentum_score_z_weight", "liquidity_1m_z_weight", "volatility_score_z_weight"],
    )
    outputs["rolling_ic"] = run_model_backtest_under_framework(
        rolling_tradable,
        score_column="rolling_ic_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=MAIN_FRAMEWORK,
    )

    for model_name, prediction_frame in linear_prediction_frames.items():
        model_cfg = linear_model_cfgs[model_name]
        weights = build_predicted_ic_weights(
            prediction_frame=prediction_frame,
            factor_columns=CORE3_FACTOR_COLUMNS,
            negative_prediction_weight=model_cfg["model"]["negative_prediction_weight"],
            fallback=model_cfg["model"]["fallback"],
            fallback_weights=fallback_weights,
            shrinkage_ic_weight=shrinkage_ic_weight,
            shrinkage_baseline_weight=shrinkage_baseline_weight,
        )
        model_scored = apply_predicted_ic_score(scored, weights, CORE3_FACTOR_COLUMNS)
        model_tradable = build_trial_tradable_panel(
            model_scored,
            extra_required_columns=["xgboost_ic_score", "momentum_score_z_weight", "liquidity_1m_z_weight", "volatility_score_z_weight"],
        )
        outputs[model_name] = run_model_backtest_under_framework(
            model_tradable,
            score_column="xgboost_ic_score",
            top_n=top_n,
            transaction_cost_bps=transaction_cost_bps,
            framework=MAIN_FRAMEWORK,
        )

    rf_weights = build_predicted_ic_weights(
        prediction_frame=random_forest_prediction_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        negative_prediction_weight=random_forest_cfg["model"]["negative_prediction_weight"],
        fallback=random_forest_cfg["model"]["fallback"],
        fallback_weights=fallback_weights,
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    rf_scored = apply_predicted_ic_score(scored, rf_weights, CORE3_FACTOR_COLUMNS)
    rf_tradable = build_trial_tradable_panel(
        rf_scored,
        extra_required_columns=["xgboost_ic_score", "momentum_score_z_weight", "liquidity_1m_z_weight", "volatility_score_z_weight"],
    )
    outputs["random_forest_ic"] = run_model_backtest_under_framework(
        rf_tradable,
        score_column="xgboost_ic_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=MAIN_FRAMEWORK,
    )

    xgb_weights = build_predicted_ic_weights(
        prediction_frame=xgboost_prediction_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        negative_prediction_weight=xgboost_cfg["model"]["negative_prediction_weight"],
        fallback=xgboost_cfg["model"]["fallback"],
        fallback_weights=fallback_weights,
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    xgb_scored = apply_predicted_ic_score(scored, xgb_weights, CORE3_FACTOR_COLUMNS)
    xgb_tradable = build_trial_tradable_panel(
        xgb_scored,
        extra_required_columns=["xgboost_ic_score", "momentum_score_z_weight", "liquidity_1m_z_weight", "volatility_score_z_weight"],
    )
    outputs["xgboost_ic"] = run_model_backtest_under_framework(
        xgb_tradable,
        score_column="xgboost_ic_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=MAIN_FRAMEWORK,
    )

    return outputs


def build_core3_weight_histories(
    baseline_weight: float,
    factor_ic_frame: pd.DataFrame,
    rolling_ic_cfg: dict,
    linear_prediction_frames: dict[str, pd.DataFrame],
    linear_model_cfgs: dict[str, dict],
    random_forest_prediction_frame: pd.DataFrame,
    random_forest_cfg: dict,
    xgboost_prediction_frame: pd.DataFrame,
    xgboost_cfg: dict,
) -> dict[str, pd.DataFrame]:
    shrinkage_ic_weight = 1.0 - baseline_weight
    shrinkage_baseline_weight = baseline_weight
    fallback_weights = {factor: 1.0 / len(CORE3_FACTOR_COLUMNS) for factor in CORE3_FACTOR_COLUMNS}

    weights_by_model: dict[str, pd.DataFrame] = {}
    weights_by_model["rolling_ic"] = build_rolling_ic_weights(
        factor_ic_frame=factor_ic_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        lookback_months=rolling_ic_cfg["model"]["lookback_months"],
        fallback=rolling_ic_cfg["model"]["fallback"],
        negative_ic_weight=rolling_ic_cfg["model"]["negative_ic_weight"],
        fallback_weights=fallback_weights,
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )

    for model_name, prediction_frame in linear_prediction_frames.items():
        model_cfg = linear_model_cfgs[model_name]
        weights_by_model[model_name] = build_predicted_ic_weights(
            prediction_frame=prediction_frame,
            factor_columns=CORE3_FACTOR_COLUMNS,
            negative_prediction_weight=model_cfg["model"]["negative_prediction_weight"],
            fallback=model_cfg["model"]["fallback"],
            fallback_weights=fallback_weights,
            shrinkage_ic_weight=shrinkage_ic_weight,
            shrinkage_baseline_weight=shrinkage_baseline_weight,
        )

    weights_by_model["random_forest_ic"] = build_predicted_ic_weights(
        prediction_frame=random_forest_prediction_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        negative_prediction_weight=random_forest_cfg["model"]["negative_prediction_weight"],
        fallback=random_forest_cfg["model"]["fallback"],
        fallback_weights=fallback_weights,
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    weights_by_model["xgboost_ic"] = build_predicted_ic_weights(
        prediction_frame=xgboost_prediction_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        negative_prediction_weight=xgboost_cfg["model"]["negative_prediction_weight"],
        fallback=xgboost_cfg["model"]["fallback"],
        fallback_weights=fallback_weights,
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    return weights_by_model


def build_selection_tables() -> tuple[pd.DataFrame, pd.DataFrame, float, SplitWindow]:
    backtest_cfg = load_yaml(BACKTEST_CONFIG)
    rolling_ic_cfg = load_yaml(ROLLING_IC_CONFIG)
    ridge_ic_cfg = load_yaml(RIDGE_IC_CONFIG)
    lasso_ic_cfg = load_yaml(LASSO_IC_CONFIG)
    elastic_net_ic_cfg = load_yaml(ELASTIC_NET_IC_CONFIG)
    random_forest_ic_cfg = load_yaml(RANDOM_FOREST_IC_CONFIG)
    xgboost_ic_cfg = load_yaml(XGBOOST_IC_CONFIG)

    top_n = backtest_cfg["portfolio"]["top_n"]
    transaction_cost_bps = backtest_cfg["costs"]["transaction_cost_bps"]

    panel = build_trial_panel()
    scored = panel.copy()
    factor_ic_frame = compute_monthly_factor_ic(scored, CORE3_FACTOR_COLUMNS, "next_month_return")

    model_feature_cfg = elastic_net_ic_cfg["model"]["features"]
    feature_frame, feature_columns = build_xgboost_ic_feature_frame(
        panel=scored,
        factor_ic_frame=factor_ic_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        ic_lag_months=model_feature_cfg["ic_lag_months"],
        ic_rolling_means=model_feature_cfg["ic_rolling_means"],
    )

    linear_model_cfgs = {
        "ridge_ic": ridge_ic_cfg,
        "lasso_ic": lasso_ic_cfg,
        "elastic_net_ic": elastic_net_ic_cfg,
    }
    linear_prediction_frames: dict[str, pd.DataFrame] = {}
    for model_name, model_cfg in linear_model_cfgs.items():
        estimator_cfg = model_cfg["model"]["estimator"].copy()
        estimator_type = str(estimator_cfg.pop("type"))
        estimator_cfg.pop("family", None)
        linear_prediction_frames[model_name] = predict_factor_ic_with_linear_model(
            feature_frame=feature_frame,
            factor_columns=CORE3_FACTOR_COLUMNS,
            feature_columns=feature_columns,
            training_window_months=model_cfg["model"]["training_window_months"],
            min_training_rows=model_cfg["model"]["min_training_rows"],
            estimator_type=estimator_type,
            estimator_params=estimator_cfg,
        )

    rf_estimator_cfg = random_forest_ic_cfg["model"]["estimator"].copy()
    rf_estimator_cfg.pop("family", None)
    rf_estimator_cfg.pop("type", None)
    random_forest_prediction_frame = predict_factor_ic_with_random_forest(
        feature_frame=feature_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        feature_columns=feature_columns,
        training_window_months=random_forest_ic_cfg["model"]["training_window_months"],
        min_training_rows=random_forest_ic_cfg["model"]["min_training_rows"],
        estimator_params=rf_estimator_cfg,
    )
    xgboost_prediction_frame = predict_factor_ic_with_xgboost(
        feature_frame=feature_frame,
        factor_columns=CORE3_FACTOR_COLUMNS,
        feature_columns=feature_columns,
        training_window_months=xgboost_ic_cfg["model"]["training_window_months"],
        min_training_rows=xgboost_ic_cfg["model"]["min_training_rows"],
        xgb_params=xgboost_ic_cfg["model"]["xgb_params"],
    )

    first_candidate_outputs = run_fullsample_core3_models(
        scored=scored,
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        baseline_weight=COMMON_SHRINKAGE_GRID[0],
        factor_ic_frame=factor_ic_frame,
        rolling_ic_cfg=rolling_ic_cfg,
        linear_prediction_frames=linear_prediction_frames,
        linear_model_cfgs=linear_model_cfgs,
        random_forest_prediction_frame=random_forest_prediction_frame,
        random_forest_cfg=random_forest_ic_cfg,
        xgboost_prediction_frame=xgboost_prediction_frame,
        xgboost_cfg=xgboost_ic_cfg,
    )
    common_start, common_end = compute_common_window({name: frame[1] for name, frame in first_candidate_outputs.items()})
    split = build_split_window(common_start, common_end)

    rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    candidate_validation_scores: dict[float, dict[str, float]] = {}
    cached_outputs: dict[float, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {
        COMMON_SHRINKAGE_GRID[0]: first_candidate_outputs
    }

    for baseline_weight in COMMON_SHRINKAGE_GRID:
        outputs = cached_outputs.get(baseline_weight)
        if outputs is None:
            outputs = run_fullsample_core3_models(
                scored=scored,
                top_n=top_n,
                transaction_cost_bps=transaction_cost_bps,
                baseline_weight=baseline_weight,
                factor_ic_frame=factor_ic_frame,
                rolling_ic_cfg=rolling_ic_cfg,
                linear_prediction_frames=linear_prediction_frames,
                linear_model_cfgs=linear_model_cfgs,
                random_forest_prediction_frame=random_forest_prediction_frame,
                random_forest_cfg=random_forest_ic_cfg,
                xgboost_prediction_frame=xgboost_prediction_frame,
                xgboost_cfg=xgboost_ic_cfg,
            )
            cached_outputs[baseline_weight] = outputs

        validation_scores: dict[str, float] = {}
        for model_name, (_, portfolio_returns, _) in outputs.items():
            validation_metrics = summarize_slice(portfolio_returns, split.validation_start, split.validation_end)
            test_metrics = summarize_slice(portfolio_returns, split.test_start, split.test_end)
            validation_net_sharpe = compute_net_sharpe(portfolio_returns, split.validation_start, split.validation_end)
            test_net_sharpe = compute_net_sharpe(portfolio_returns, split.test_start, split.test_end)
            validation_scores[model_name] = validation_net_sharpe
            rows.append(
                {
                    "baseline_weight": baseline_weight,
                    "ic_weight": 1.0 - baseline_weight,
                    "model": model_name,
                    "validation_net_sharpe": validation_net_sharpe,
                    "validation_net_return": validation_metrics["net_return_after_costs"],
                    "validation_turnover": validation_metrics["turnover"],
                    "test_net_sharpe": test_net_sharpe,
                    "test_net_return": test_metrics["net_return_after_costs"],
                    "test_turnover": test_metrics["turnover"],
                }
            )
        candidate_validation_scores[baseline_weight] = validation_scores

    rank_rows: list[dict[str, object]] = []
    objective_matrix = pd.DataFrame(candidate_validation_scores).T.sort_index()
    for baseline_weight in COMMON_SHRINKAGE_GRID:
        candidate_scores = objective_matrix.loc[baseline_weight]
        rank_rows.append(
            {
                "baseline_weight": baseline_weight,
                "ic_weight": 1.0 - baseline_weight,
                "mean_validation_net_sharpe": candidate_scores.mean(),
                "median_validation_net_sharpe": candidate_scores.median(),
                "net_sharpe_standard_error": candidate_scores.std(ddof=1) / np.sqrt(candidate_scores.count()),
                "mean_validation_net_return": pd.DataFrame(rows)
                .query("baseline_weight == @baseline_weight")["validation_net_return"]
                .mean(),
            }
        )
    selection_summary = pd.DataFrame(rank_rows).sort_values("baseline_weight").reset_index(drop=True)
    best_row = selection_summary.loc[selection_summary["median_validation_net_sharpe"].idxmax()]
    threshold = float(best_row["median_validation_net_sharpe"] - best_row["net_sharpe_standard_error"])
    eligible = selection_summary[selection_summary["median_validation_net_sharpe"] >= threshold].copy()
    selected_baseline_weight = float(eligible["baseline_weight"].max())
    selection_summary["selected_by_one_se_rule"] = selection_summary["baseline_weight"].eq(selected_baseline_weight)
    selection_summary["common_window_start"] = split.train_start.date().isoformat()
    selection_summary["common_window_end"] = split.test_end.date().isoformat()
    selection_summary["validation_start"] = split.validation_start.date().isoformat()
    selection_summary["validation_end"] = split.validation_end.date().isoformat()
    selection_summary["test_start"] = split.test_start.date().isoformat()
    selection_summary["test_end"] = split.test_end.date().isoformat()

    selected_outputs = cached_outputs[selected_baseline_weight]
    selected_weight_histories = build_core3_weight_histories(
        baseline_weight=selected_baseline_weight,
        factor_ic_frame=factor_ic_frame,
        rolling_ic_cfg=rolling_ic_cfg,
        linear_prediction_frames=linear_prediction_frames,
        linear_model_cfgs=linear_model_cfgs,
        random_forest_prediction_frame=random_forest_prediction_frame,
        random_forest_cfg=random_forest_ic_cfg,
        xgboost_prediction_frame=xgboost_prediction_frame,
        xgboost_cfg=xgboost_ic_cfg,
    )
    for model_name, (selected, portfolio_returns, _) in selected_outputs.items():
        selected_metrics = summarize_slice(portfolio_returns, split.test_start, split.test_end)
        test_rows.append(
            {
                "model": model_name,
                "framework": MAIN_FRAMEWORK,
                "baseline_weight": selected_baseline_weight,
                "ic_weight": 1.0 - selected_baseline_weight,
                "test_net_sharpe": compute_net_sharpe(portfolio_returns, split.test_start, split.test_end),
                **selected_metrics,
            }
        )
        save_backtest_outputs(
            prefix=f"common_shrinkage_{MAIN_FRAMEWORK}_{model_name}",
            selected=selected,
            portfolio_returns=portfolio_returns,
            metrics_frame=pd.DataFrame([selected_metrics]),
        )
        weight_history = selected_weight_histories.get(model_name)
        if weight_history is not None:
            weight_history.to_csv(
                OUTPUT_DIR / f"common_shrinkage_{MAIN_FRAMEWORK}_{model_name}_weight_history.csv",
                index=False,
            )

    fixed_tradable = build_trial_tradable_panel(scored)
    fixed_selected, fixed_portfolio_returns, _ = run_model_backtest_under_framework(
        fixed_tradable,
        score_column="fixed_weight_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=MAIN_FRAMEWORK,
    )
    fixed_test_metrics = summarize_slice(fixed_portfolio_returns, split.test_start, split.test_end)
    test_rows.append(
        {
            "model": "fixed_weight",
            "framework": MAIN_FRAMEWORK,
            "baseline_weight": np.nan,
            "ic_weight": np.nan,
            "test_net_sharpe": compute_net_sharpe(fixed_portfolio_returns, split.test_start, split.test_end),
            **fixed_test_metrics,
        }
    )
    save_backtest_outputs(
        prefix=f"common_shrinkage_{MAIN_FRAMEWORK}_fixed_weight",
        selected=fixed_selected,
        portfolio_returns=fixed_portfolio_returns,
        metrics_frame=pd.DataFrame([fixed_test_metrics]),
    )

    benchmark_selected, benchmark_portfolio_returns, _ = run_equal_weight_backtest(
        fixed_tradable,
        transaction_cost_bps=transaction_cost_bps,
    )
    benchmark_test_metrics = summarize_slice(benchmark_portfolio_returns, split.test_start, split.test_end)
    test_rows.append(
        {
            "model": "equal_weight_benchmark",
            "framework": MAIN_FRAMEWORK,
            "baseline_weight": np.nan,
            "ic_weight": np.nan,
            "test_net_sharpe": compute_net_sharpe(benchmark_portfolio_returns, split.test_start, split.test_end),
            **benchmark_test_metrics,
        }
    )
    save_backtest_outputs(
        prefix=f"common_shrinkage_{MAIN_FRAMEWORK}_equal_weight_benchmark",
        selected=benchmark_selected,
        portfolio_returns=benchmark_portfolio_returns,
        metrics_frame=pd.DataFrame([benchmark_test_metrics]),
    )

    detailed = pd.DataFrame(rows).sort_values(["baseline_weight", "model"]).reset_index(drop=True)
    test_comparison = pd.DataFrame(test_rows).sort_values("model").reset_index(drop=True)
    return detailed, selection_summary, selected_baseline_weight, split, test_comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    detailed, selection_summary, selected_baseline_weight, split, test_comparison = build_selection_tables()

    detailed.to_csv(TABLES_DIR / "table_sh1_common_shrinkage_validation_grid.csv", index=False)
    selection_summary.to_csv(TABLES_DIR / "table_sh2_common_shrinkage_selection_summary.csv", index=False)
    test_comparison.to_csv(TABLES_DIR / "table_sh3_common_shrinkage_test_comparison.csv", index=False)

    conclusion = pd.DataFrame(
        [
            {
                "common_window_start": split.train_start.date().isoformat(),
                "common_window_end": split.test_end.date().isoformat(),
                "train_period": f"{split.train_start.date()} to {split.train_end.date()}",
                "validation_period": f"{split.validation_start.date()} to {split.validation_end.date()}",
                "test_period": f"{split.test_start.date()} to {split.test_end.date()}",
                "framework": MAIN_FRAMEWORK,
                "selected_baseline_weight": selected_baseline_weight,
                "selected_ic_weight": 1.0 - selected_baseline_weight,
                "selection_rule": "one-standard-error over family-wide median validation net Sharpe; choose strongest shrinkage within 1-SE",
            }
        ]
    )
    conclusion.to_csv(TABLES_DIR / "table_sh4_common_shrinkage_selection_conclusion.csv", index=False)
    print(f"Saved {TABLES_DIR / 'table_sh1_common_shrinkage_validation_grid.csv'}")
    print(f"Saved {TABLES_DIR / 'table_sh2_common_shrinkage_selection_summary.csv'}")
    print(f"Saved {TABLES_DIR / 'table_sh3_common_shrinkage_test_comparison.csv'}")
    print(f"Saved {TABLES_DIR / 'table_sh4_common_shrinkage_selection_conclusion.csv'}")
    print(conclusion.to_string(index=False))


if __name__ == "__main__":
    main()
