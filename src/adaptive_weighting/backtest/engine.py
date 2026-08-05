from __future__ import annotations

from pathlib import Path

import pandas as pd

from adaptive_weighting.backtest.evaluation import summarize_performance
from adaptive_weighting.portfolio.selection import select_top_n_by_score, select_top_n_with_holding_buffer


def prepare_panel(
    path: Path,
    date_column: str = "Date",
    symbol_column: str = "symbol",
    close_column: str = "Close",
    forward_return_column: str = "next_month_return",
) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=[date_column]).sort_values([symbol_column, date_column]).reset_index(drop=True)
    panel[forward_return_column] = panel.groupby(symbol_column)[close_column].shift(-1) / panel[close_column] - 1.0
    return panel


def compute_turnover(
    selection: pd.DataFrame,
    date_column: str = "Date",
    symbol_column: str = "symbol",
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    weights = selection.pivot(index=date_column, columns=symbol_column, values=weight_column).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    return turnover.rename("turnover").reset_index()


def compute_time_varying_transaction_costs(
    selection: pd.DataFrame,
    spread_source: pd.DataFrame,
    spread_column: str = "corwin_schultz_spread",
    date_column: str = "Date",
    symbol_column: str = "symbol",
    weight_column: str = "portfolio_weight",
) -> pd.DataFrame:
    weights = selection.pivot(index=date_column, columns=symbol_column, values=weight_column).fillna(0.0)
    traded_weights = weights.diff().abs()
    traded_weights.iloc[0] = weights.iloc[0].abs()

    spreads = (
        spread_source[[date_column, symbol_column, spread_column]]
        .drop_duplicates(subset=[date_column, symbol_column])
        .pivot(index=date_column, columns=symbol_column, values=spread_column)
    )
    spreads = spreads.reindex(index=weights.index, columns=weights.columns)
    spreads = spreads.ffill().bfill()

    transaction_cost_rate = (traded_weights * (spreads / 2.0)).sum(axis=1)
    return transaction_cost_rate.rename("transaction_cost_rate").reset_index()


def build_portfolio_returns(
    selection: pd.DataFrame,
    date_column: str = "Date",
    symbol_column: str = "symbol",
    weighted_return_column: str = "weighted_return",
) -> pd.DataFrame:
    return (
        selection.groupby(date_column, as_index=False)
        .agg(
            portfolio_return=(weighted_return_column, "sum"),
            selected_symbols=(symbol_column, lambda values: ",".join(sorted(values))),
        )
    )


def build_equal_weight_benchmark_panel(
    tradable: pd.DataFrame,
    date_column: str = "Date",
    symbol_column: str = "symbol",
    forward_return_column: str = "next_month_return",
    weight_column: str = "portfolio_weight",
    weighted_return_column: str = "weighted_return",
) -> pd.DataFrame:
    benchmark = tradable.copy()
    counts = benchmark.groupby(date_column)[symbol_column].transform("count")
    benchmark[weight_column] = 1.0 / counts
    benchmark[weighted_return_column] = benchmark[weight_column] * benchmark[forward_return_column]
    return benchmark


def resolve_holding_buffer_rank(framework: str | None) -> int | None:
    if framework in {None, "top_n", "baseline", "pta"}:
        return None
    if framework == "holding_buffer":
        raise ValueError("hold_buffer_rank must be provided when framework='holding_buffer'")
    if framework.startswith("holding_buffer_top"):
        suffix = framework.removeprefix("holding_buffer_top")
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Unsupported framework: {framework}")


def select_portfolio(
    tradable: pd.DataFrame,
    score_column: str,
    top_n: int,
    framework: str | None = "top_n",
    hold_buffer_rank: int | None = None,
) -> pd.DataFrame:
    if framework == "holding_buffer":
        if hold_buffer_rank is None:
            raise ValueError("hold_buffer_rank must be provided when framework='holding_buffer'")
        return select_top_n_with_holding_buffer(
            tradable,
            score_column=score_column,
            top_n=top_n,
            hold_buffer_rank=hold_buffer_rank,
        )

    inferred_rank = resolve_holding_buffer_rank(framework)
    if inferred_rank is not None:
        return select_top_n_with_holding_buffer(
            tradable,
            score_column=score_column,
            top_n=top_n,
            hold_buffer_rank=inferred_rank,
        )

    return select_top_n_by_score(tradable, score_column, top_n)


def run_scored_backtest(
    tradable: pd.DataFrame,
    score_column: str,
    top_n: int,
    transaction_cost_bps: float,
    framework: str | None = "top_n",
    hold_buffer_rank: int | None = None,
    spread_source: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = select_portfolio(
        tradable,
        score_column=score_column,
        top_n=top_n,
        framework=framework,
        hold_buffer_rank=hold_buffer_rank,
    )
    spread_data = tradable if spread_source is None else spread_source
    selected = selected.copy()
    selected["weighted_return"] = selected["portfolio_weight"] * selected["next_month_return"]

    portfolio_returns = build_portfolio_returns(selected)
    turnover = compute_turnover(selected)
    transaction_costs = compute_time_varying_transaction_costs(selected, spread_data)
    portfolio_returns = portfolio_returns.merge(turnover, on="Date", how="left").merge(transaction_costs, on="Date", how="left")
    portfolio_returns["net_portfolio_return"] = (
        portfolio_returns["portfolio_return"] - portfolio_returns["transaction_cost_rate"].fillna(0.0)
    )

    metrics = summarize_performance(
        return_series=portfolio_returns["portfolio_return"],
        turnover_series=portfolio_returns["turnover"],
        transaction_cost_bps=transaction_cost_bps,
        transaction_cost_rate_series=portfolio_returns["transaction_cost_rate"],
    )
    return selected, portfolio_returns, pd.DataFrame([metrics])


def run_equal_weight_backtest(
    tradable: pd.DataFrame,
    transaction_cost_bps: float,
    spread_source: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = build_equal_weight_benchmark_panel(tradable)
    spread_data = tradable if spread_source is None else spread_source
    portfolio_returns = build_portfolio_returns(selected)
    turnover = compute_turnover(selected)
    transaction_costs = compute_time_varying_transaction_costs(selected, spread_data)
    portfolio_returns = portfolio_returns.merge(turnover, on="Date", how="left").merge(transaction_costs, on="Date", how="left")
    portfolio_returns["net_portfolio_return"] = (
        portfolio_returns["portfolio_return"] - portfolio_returns["transaction_cost_rate"].fillna(0.0)
    )

    metrics = summarize_performance(
        return_series=portfolio_returns["portfolio_return"],
        turnover_series=portfolio_returns["turnover"],
        transaction_cost_bps=transaction_cost_bps,
        transaction_cost_rate_series=portfolio_returns["transaction_cost_rate"],
    )
    return selected, portfolio_returns, pd.DataFrame([metrics])
