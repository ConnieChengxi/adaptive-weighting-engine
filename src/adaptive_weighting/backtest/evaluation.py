from __future__ import annotations

import math

import pandas as pd


def compute_drawdown(return_series: pd.Series) -> pd.Series:
    cumulative = (1.0 + return_series.fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    return cumulative / running_max - 1.0


def compute_net_returns(return_series: pd.Series, turnover_series: pd.Series, transaction_cost_bps: float) -> pd.Series:
    transaction_cost_rate = transaction_cost_bps / 10000.0
    return return_series.fillna(0.0) - turnover_series.fillna(0.0) * transaction_cost_rate


def annualized_compound_return(return_series: pd.Series, annualization: float = 12.0) -> float:
    returns = return_series.fillna(0.0)
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    growth = (1.0 + returns).prod()
    return growth ** (annualization / n_periods) - 1.0


def summarize_performance(return_series: pd.Series, turnover_series: pd.Series, transaction_cost_bps: float) -> dict[str, float]:
    gross_returns = return_series.fillna(0.0)
    net_returns = compute_net_returns(gross_returns, turnover_series, transaction_cost_bps)

    n_periods = len(gross_returns)
    annualization = 12.0

    annualized_return = annualized_compound_return(gross_returns, annualization=annualization)
    annualized_volatility = gross_returns.std(ddof=1) * math.sqrt(annualization) if n_periods > 1 else 0.0
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility else 0.0

    drawdown = compute_drawdown(gross_returns)
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown else 0.0

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "turnover": turnover_series.mean() if not turnover_series.empty else 0.0,
        "net_return_after_costs": annualized_compound_return(net_returns, annualization=annualization),
    }
