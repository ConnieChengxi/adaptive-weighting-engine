from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_weighting.backtest.evaluation import annualized_compound_return, compute_net_returns, summarize_performance
from adaptive_weighting.ic.compute_ic import compute_monthly_factor_ic
from adaptive_weighting.models.fixed_weight import build_fixed_weight_score
from adaptive_weighting.models.rolling_ic_weighting import apply_rolling_ic_score, build_rolling_ic_weights
from adaptive_weighting.models.xgboost_ic_weighting import (
    apply_predicted_ic_score,
    build_predicted_ic_weights,
    build_xgboost_ic_feature_frame,
    predict_factor_ic_with_xgboost,
)
from adaptive_weighting.portfolio.selection import select_top_n_by_score


BACKTEST_CONFIG = ROOT / "config" / "backtest.yaml"
FIXED_WEIGHT_CONFIG = ROOT / "config" / "models" / "fixed_weight.yaml"
ROLLING_IC_CONFIG = ROOT / "config" / "models" / "rolling_ic.yaml"
XGBOOST_IC_CONFIG = ROOT / "config" / "models" / "xgboost_ic.yaml"
PANEL_PATH = ROOT / "data" / "processed" / "monthly_factor_panel.csv"
OUTPUT_DIR = ROOT / "outputs" / "backtests"

FACTOR_COLUMNS = [
    "momentum_score_z",
    "liquidity_1m_z",
    "downside_risk_score_z",
    "volatility_score_z",
]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["Date"]).sort_values(["symbol", "Date"]).reset_index(drop=True)
    panel["next_month_return"] = panel.groupby("symbol")["Close"].shift(-1) / panel["Close"] - 1.0
    return panel


def compute_turnover(selection: pd.DataFrame) -> pd.DataFrame:
    weights = selection.pivot(index="Date", columns="symbol", values="portfolio_weight").fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    return turnover.rename("turnover").reset_index()


def build_portfolio_returns(selection: pd.DataFrame) -> pd.DataFrame:
    portfolio = (
        selection.groupby("Date", as_index=False)
        .agg(
            portfolio_return=("weighted_return", "sum"),
            selected_symbols=("symbol", lambda values: ",".join(sorted(values))),
        )
    )
    return portfolio


def prepare_scored_panel(panel: pd.DataFrame, factor_weights: dict[str, float]) -> pd.DataFrame:
    scored = build_fixed_weight_score(panel, factor_weights)
    return scored


def build_fixed_weight_fallback_mapping(factor_weights: dict[str, float]) -> dict[str, float]:
    return {
        "momentum_score_z": factor_weights["momentum"],
        "liquidity_1m_z": factor_weights["liquidity"],
        "downside_risk_score_z": factor_weights["downside_risk"],
        "volatility_score_z": factor_weights["volatility"],
    }


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


def build_tradable_panel(scored: pd.DataFrame, extra_required_columns: list[str] | None = None) -> pd.DataFrame:
    required_columns = [
        "momentum_3m_z",
        "momentum_6m_z",
        "liquidity_1m_z",
        "downside_risk_score_z",
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
    selected = select_top_n_by_score(tradable, score_column, top_n)
    selected["weighted_return"] = selected["portfolio_weight"] * selected["next_month_return"]

    portfolio_returns = build_portfolio_returns(selected)
    turnover = compute_turnover(selected)
    portfolio_returns = portfolio_returns.merge(turnover, on="Date", how="left")

    metrics = summarize_performance(
        portfolio_returns["portfolio_return"],
        portfolio_returns["turnover"],
        transaction_cost_bps=transaction_cost_bps,
    )
    metrics_frame = pd.DataFrame([metrics])
    return selected, portfolio_returns, metrics_frame


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


def build_transaction_cost_sensitivity(
    fixed_portfolio_returns: pd.DataFrame,
    candidate_portfolio_returns: pd.DataFrame,
    candidate_label: str,
    max_bps: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, float | int | str]] = []

    for bps in range(max_bps + 1):
        fixed_net_ann = annualized_compound_return(
            compute_net_returns(
                fixed_portfolio_returns["portfolio_return"],
                fixed_portfolio_returns["turnover"],
                bps,
            )
        )
        candidate_net_ann = annualized_compound_return(
            compute_net_returns(
                candidate_portfolio_returns["portfolio_return"],
                candidate_portfolio_returns["turnover"],
                bps,
            )
        )
        records.append(
            {
                "transaction_cost_bps": bps,
                "fixed_weight_net_annualized_return": fixed_net_ann,
                f"{candidate_label}_net_annualized_return": candidate_net_ann,
                f"{candidate_label}_minus_fixed_weight": candidate_net_ann - fixed_net_ann,
            }
        )

    sensitivity = pd.DataFrame(records)
    crossing = sensitivity[sensitivity[f"{candidate_label}_minus_fixed_weight"] < 0]
    threshold_bps = int(crossing["transaction_cost_bps"].iloc[0]) if not crossing.empty else None
    summary = pd.DataFrame(
        [
            {
                "candidate_model": candidate_label,
                "max_bps_tested": max_bps,
                "first_bps_where_candidate_underperforms": threshold_bps,
            }
        ]
    )
    return sensitivity, summary


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
        fallback_weights=build_fixed_weight_fallback_mapping(factor_weights),
        shrinkage_ic_weight=shrinkage_ic_weight,
        shrinkage_baseline_weight=shrinkage_baseline_weight,
    )
    weights_frame.to_csv(OUTPUT_DIR / f"{prefix}_weight_history.csv", index=False)

    rolling_scored = apply_rolling_ic_score(scored, weights_frame, FACTOR_COLUMNS)
    rolling_tradable = build_tradable_panel(
        rolling_scored,
        extra_required_columns=[
            "rolling_ic_score",
            "momentum_score_z_weight",
            "liquidity_1m_z_weight",
            "downside_risk_score_z_weight",
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


def main() -> None:
    backtest_cfg = load_yaml(BACKTEST_CONFIG)
    fixed_weight_cfg = load_yaml(FIXED_WEIGHT_CONFIG)
    rolling_ic_cfg = load_yaml(ROLLING_IC_CONFIG)
    xgboost_ic_cfg = load_yaml(XGBOOST_IC_CONFIG)

    top_n = backtest_cfg["portfolio"]["top_n"]
    transaction_cost_bps = backtest_cfg["costs"]["transaction_cost_bps"]
    factor_weights = fixed_weight_cfg["model"]["factors"]

    panel = prepare_panel(PANEL_PATH)
    scored = prepare_scored_panel(panel, factor_weights)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed_weight_tradable = build_tradable_panel(scored)
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
            "fixed_weight": fixed_metrics,
            "rolling_ic": rolling_metrics_by_model["rolling_ic"],
        },
        filename="model_comparison_metrics.csv",
    )
    save_model_comparison(
        {
            "fixed_weight": fixed_metrics,
            "rolling_ic_80_20": rolling_metrics_by_model["rolling_ic"],
            "rolling_ic_no_shrinkage": rolling_metrics_by_model["rolling_ic_no_shrinkage"],
            "rolling_ic_60_40": rolling_metrics_by_model["rolling_ic_shrinkage_60_40"],
        },
        filename="rolling_ic_robustness_comparison.csv",
    )

    xgboost_feature_frame, xgboost_feature_columns = build_xgboost_ic_feature_frame(
        panel=scored,
        factor_ic_frame=factor_ic_frame,
        factor_columns=FACTOR_COLUMNS,
        ic_lag_months=xgboost_ic_cfg["model"]["features"]["ic_lag_months"],
        ic_rolling_means=xgboost_ic_cfg["model"]["features"]["ic_rolling_means"],
    )
    xgboost_feature_frame.to_csv(OUTPUT_DIR / "xgboost_ic_feature_frame.csv", index=False)

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
        fallback_weights=build_fixed_weight_fallback_mapping(factor_weights),
        shrinkage_ic_weight=resolve_model_shrinkage_config(xgboost_ic_cfg)[0],
        shrinkage_baseline_weight=resolve_model_shrinkage_config(xgboost_ic_cfg)[1],
    )
    xgboost_weights.to_csv(OUTPUT_DIR / "xgboost_ic_weight_history.csv", index=False)

    xgboost_scored = apply_predicted_ic_score(scored, xgboost_weights, FACTOR_COLUMNS)
    xgboost_tradable = build_tradable_panel(
        xgboost_scored,
        extra_required_columns=[
            "xgboost_ic_score",
            "momentum_score_z_weight",
            "liquidity_1m_z_weight",
            "downside_risk_score_z_weight",
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
            "fixed_weight": fixed_metrics,
            "rolling_ic_80_20": rolling_metrics_by_model["rolling_ic"],
            "xgboost_ic": xgboost_metrics,
        },
        filename="full_model_comparison_metrics.csv",
    )

    xgboost_tc_sensitivity, xgboost_tc_summary = build_transaction_cost_sensitivity(
        fixed_portfolio_returns=fixed_portfolio_returns,
        candidate_portfolio_returns=xgboost_portfolio_returns,
        candidate_label="xgboost_ic",
        max_bps=50,
    )
    xgboost_tc_sensitivity.to_csv(
        OUTPUT_DIR / "xgboost_ic_transaction_cost_sensitivity.csv",
        index=False,
    )
    xgboost_tc_summary.to_csv(
        OUTPUT_DIR / "xgboost_ic_transaction_cost_threshold_summary.csv",
        index=False,
    )
    print(f"Saved {(OUTPUT_DIR / 'xgboost_ic_transaction_cost_sensitivity.csv').relative_to(ROOT)}")
    print(f"Saved {(OUTPUT_DIR / 'xgboost_ic_transaction_cost_threshold_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
