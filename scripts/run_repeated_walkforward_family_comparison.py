from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_common_shrinkage_selection import (  # type: ignore
    BACKTEST_CONFIG,
    COMMON_SHRINKAGE_GRID,
    CORE3_FACTOR_COLUMNS,
    ELASTIC_NET_IC_CONFIG,
    LASSO_IC_CONFIG,
    MAIN_FRAMEWORK,
    RANDOM_FOREST_IC_CONFIG,
    RIDGE_IC_CONFIG,
    ROLLING_IC_CONFIG,
    TABLES_DIR,
    XGBOOST_IC_CONFIG,
    build_trial_panel,
    build_trial_tradable_panel,
    compute_common_window,
    compute_net_sharpe,
    compute_monthly_factor_ic,
    build_xgboost_ic_feature_frame,
    load_yaml,
    predict_factor_ic_with_linear_model,
    predict_factor_ic_with_random_forest,
    predict_factor_ic_with_xgboost,
    run_equal_weight_backtest,
    run_fullsample_core3_models,
    run_model_backtest_under_framework,
    summarize_slice,
)


OUTPUT_TABLE_1 = TABLES_DIR / "table_wf1_repeated_walkforward_fold_selection.csv"
OUTPUT_TABLE_2 = TABLES_DIR / "table_wf2_repeated_walkforward_test_results.csv"
OUTPUT_TABLE_3 = TABLES_DIR / "table_wf3_repeated_walkforward_family_comparison.csv"

MODEL_ORDER = [
    "equal_weight_benchmark",
    "fixed_weight",
    "rolling_ic",
    "ridge_ic",
    "lasso_ic",
    "elastic_net_ic",
    "random_forest_ic",
    "xgboost_ic",
]
ADAPTIVE_MODEL_ORDER = [
    "rolling_ic",
    "ridge_ic",
    "lasso_ic",
    "elastic_net_ic",
    "random_forest_ic",
    "xgboost_ic",
]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_validation: int
    n_test: int


def build_walkforward_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_train_months: int = 120,
    validation_months: int = 60,
    test_months: int = 24,
) -> list[WalkForwardFold]:
    dates = pd.date_range(start=start, end=end, freq="ME")
    folds: list[WalkForwardFold] = []
    test_end_idx = len(dates) - 1
    fold_counter = 1

    while True:
        test_start_idx = test_end_idx - test_months + 1
        validation_end_idx = test_start_idx - 1
        validation_start_idx = validation_end_idx - validation_months + 1
        train_end_idx = validation_start_idx - 1
        n_train = train_end_idx + 1

        if test_start_idx < 0 or validation_start_idx < 0 or n_train < min_train_months:
            break

        folds.append(
            WalkForwardFold(
                fold_id=fold_counter,
                train_start=dates[0],
                train_end=dates[train_end_idx],
                validation_start=dates[validation_start_idx],
                validation_end=dates[validation_end_idx],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx],
                n_train=n_train,
                n_validation=validation_months,
                n_test=test_months,
            )
        )
        fold_counter += 1
        test_end_idx -= test_months

    folds.reverse()
    return [
        WalkForwardFold(
            fold_id=index + 1,
            train_start=fold.train_start,
            train_end=fold.train_end,
            validation_start=fold.validation_start,
            validation_end=fold.validation_end,
            test_start=fold.test_start,
            test_end=fold.test_end,
            n_train=fold.n_train,
            n_validation=fold.n_validation,
            n_test=fold.n_test,
        )
        for index, fold in enumerate(folds)
    ]


def build_fullsample_outputs() -> tuple[
    dict[float, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]],
    tuple[pd.DataFrame, pd.DataFrame],
    tuple[pd.DataFrame, pd.DataFrame],
]:
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

    cached_outputs: dict[float, dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]] = {}
    for baseline_weight in COMMON_SHRINKAGE_GRID:
        cached_outputs[baseline_weight] = run_fullsample_core3_models(
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

    fixed_tradable = build_trial_tradable_panel(scored)
    fixed_selected, fixed_returns, _ = run_model_backtest_under_framework(
        fixed_tradable,
        score_column="fixed_weight_score",
        top_n=top_n,
        transaction_cost_bps=transaction_cost_bps,
        framework=MAIN_FRAMEWORK,
    )
    benchmark_selected, benchmark_returns, _ = run_equal_weight_backtest(
        fixed_tradable,
        transaction_cost_bps=transaction_cost_bps,
    )
    _ = fixed_selected, benchmark_selected
    return cached_outputs, (fixed_tradable, fixed_returns), (fixed_tradable, benchmark_returns)


def build_walkforward_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cached_outputs, (_, fixed_returns), (_, benchmark_returns) = build_fullsample_outputs()
    first_outputs = cached_outputs[COMMON_SHRINKAGE_GRID[0]]
    common_start, common_end = compute_common_window({name: frame[1] for name, frame in first_outputs.items()})
    folds = build_walkforward_folds(common_start, common_end)

    selection_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for fold in folds:
        baseline_validation_scores: dict[float, dict[str, float]] = {}
        for baseline_weight in COMMON_SHRINKAGE_GRID:
            outputs = cached_outputs[baseline_weight]
            model_scores: dict[str, float] = {}
            for model_name in ADAPTIVE_MODEL_ORDER:
                portfolio_returns = outputs[model_name][1]
                validation_net_sharpe = compute_net_sharpe(
                    portfolio_returns,
                    fold.validation_start,
                    fold.validation_end,
                )
                model_scores[model_name] = validation_net_sharpe
            baseline_validation_scores[baseline_weight] = model_scores

        objective_matrix = pd.DataFrame(baseline_validation_scores).T.sort_index()
        fold_summary_rows: list[dict[str, object]] = []
        for baseline_weight in COMMON_SHRINKAGE_GRID:
            scores = objective_matrix.loc[baseline_weight]
            fold_summary_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "train_period": f"{fold.train_start.date()} to {fold.train_end.date()}",
                    "validation_period": f"{fold.validation_start.date()} to {fold.validation_end.date()}",
                    "test_period": f"{fold.test_start.date()} to {fold.test_end.date()}",
                    "baseline_weight": baseline_weight,
                    "ic_weight": 1.0 - baseline_weight,
                    "mean_validation_net_sharpe": scores.mean(),
                    "median_validation_net_sharpe": scores.median(),
                    "net_sharpe_standard_error": scores.std(ddof=1) / np.sqrt(scores.count()),
                }
            )
        fold_summary = pd.DataFrame(fold_summary_rows).sort_values("baseline_weight").reset_index(drop=True)
        best_row = fold_summary.loc[fold_summary["median_validation_net_sharpe"].idxmax()]
        threshold = float(best_row["median_validation_net_sharpe"] - best_row["net_sharpe_standard_error"])
        eligible = fold_summary[fold_summary["median_validation_net_sharpe"] >= threshold].copy()
        selected_baseline_weight = float(eligible["baseline_weight"].max())
        fold_summary["selected_by_one_se_rule"] = fold_summary["baseline_weight"].eq(selected_baseline_weight)
        selection_rows.extend(fold_summary.to_dict(orient="records"))

        selected_outputs = cached_outputs[selected_baseline_weight]
        current_fold_rows: list[dict[str, object]] = []
        for model_name in ADAPTIVE_MODEL_ORDER:
            portfolio_returns = selected_outputs[model_name][1]
            metrics = summarize_slice(portfolio_returns, fold.test_start, fold.test_end)
            current_fold_rows.append(
                {
                    "fold_id": fold.fold_id,
                    "model": model_name,
                    "framework": MAIN_FRAMEWORK,
                    "baseline_weight": selected_baseline_weight,
                    "ic_weight": 1.0 - selected_baseline_weight,
                    "train_period": f"{fold.train_start.date()} to {fold.train_end.date()}",
                    "validation_period": f"{fold.validation_start.date()} to {fold.validation_end.date()}",
                    "test_period": f"{fold.test_start.date()} to {fold.test_end.date()}",
                    "test_net_sharpe": compute_net_sharpe(portfolio_returns, fold.test_start, fold.test_end),
                    **metrics,
                }
            )

        fixed_metrics = summarize_slice(fixed_returns, fold.test_start, fold.test_end)
        current_fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "model": "fixed_weight",
                "framework": MAIN_FRAMEWORK,
                "baseline_weight": np.nan,
                "ic_weight": np.nan,
                "train_period": f"{fold.train_start.date()} to {fold.train_end.date()}",
                "validation_period": f"{fold.validation_start.date()} to {fold.validation_end.date()}",
                "test_period": f"{fold.test_start.date()} to {fold.test_end.date()}",
                "test_net_sharpe": compute_net_sharpe(fixed_returns, fold.test_start, fold.test_end),
                **fixed_metrics,
            }
        )

        benchmark_metrics = summarize_slice(benchmark_returns, fold.test_start, fold.test_end)
        current_fold_rows.append(
            {
                "fold_id": fold.fold_id,
                "model": "equal_weight_benchmark",
                "framework": MAIN_FRAMEWORK,
                "baseline_weight": np.nan,
                "ic_weight": np.nan,
                "train_period": f"{fold.train_start.date()} to {fold.train_end.date()}",
                "validation_period": f"{fold.validation_start.date()} to {fold.validation_end.date()}",
                "test_period": f"{fold.test_start.date()} to {fold.test_end.date()}",
                "test_net_sharpe": compute_net_sharpe(benchmark_returns, fold.test_start, fold.test_end),
                **benchmark_metrics,
            }
        )

        current_fold_frame = pd.DataFrame(current_fold_rows)
        fixed_fold_sharpe = float(
            current_fold_frame.loc[current_fold_frame["model"] == "fixed_weight", "test_net_sharpe"].iloc[0]
        )
        benchmark_fold_sharpe = float(
            current_fold_frame.loc[current_fold_frame["model"] == "equal_weight_benchmark", "test_net_sharpe"].iloc[0]
        )
        best_fold_sharpe = float(current_fold_frame["test_net_sharpe"].max())
        current_fold_frame["outperforms_fixed_weight"] = current_fold_frame["test_net_sharpe"] > fixed_fold_sharpe
        current_fold_frame["outperforms_equal_weight"] = current_fold_frame["test_net_sharpe"] > benchmark_fold_sharpe
        current_fold_frame["sharpe_gap_vs_fixed_weight"] = current_fold_frame["test_net_sharpe"] - fixed_fold_sharpe
        current_fold_frame["best_model_in_fold"] = current_fold_frame["test_net_sharpe"].eq(best_fold_sharpe)
        test_rows.extend(current_fold_frame.to_dict(orient="records"))

    selection_table = pd.DataFrame(selection_rows).sort_values(["fold_id", "baseline_weight"]).reset_index(drop=True)
    test_table = pd.DataFrame(test_rows)
    test_table["model"] = pd.Categorical(test_table["model"], categories=MODEL_ORDER, ordered=True)
    test_table = test_table.sort_values(["fold_id", "model"]).reset_index(drop=True)

    aggregate = (
        test_table.groupby("model", observed=True)
        .agg(
            n_folds=("fold_id", "count"),
            mean_test_net_sharpe=("test_net_sharpe", "mean"),
            median_test_net_sharpe=("test_net_sharpe", "median"),
            std_test_net_sharpe=("test_net_sharpe", "std"),
            mean_net_return=("net_return_after_costs", "mean"),
            median_net_return=("net_return_after_costs", "median"),
            mean_turnover=("turnover", "mean"),
            median_turnover=("turnover", "median"),
            outperformance_rate_vs_fixed_weight=("outperforms_fixed_weight", "mean"),
            outperformance_rate_vs_equal_weight=("outperforms_equal_weight", "mean"),
            mean_sharpe_gap_vs_fixed_weight=("sharpe_gap_vs_fixed_weight", "mean"),
            best_fold_count=("best_model_in_fold", "sum"),
        )
        .reset_index()
    )
    aggregate["model"] = pd.Categorical(aggregate["model"], categories=MODEL_ORDER, ordered=True)
    aggregate = aggregate.sort_values("model").reset_index(drop=True)
    return selection_table, test_table, aggregate


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    selection_table, test_table, aggregate = build_walkforward_tables()
    selection_table.to_csv(OUTPUT_TABLE_1, index=False)
    test_table.to_csv(OUTPUT_TABLE_2, index=False)
    aggregate.to_csv(OUTPUT_TABLE_3, index=False)
    print(f"Saved {OUTPUT_TABLE_1}")
    print(f"Saved {OUTPUT_TABLE_2}")
    print(f"Saved {OUTPUT_TABLE_3}")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
