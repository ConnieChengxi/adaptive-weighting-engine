import pandas as pd

from adaptive_weighting.backtest.evaluation import summarize_performance


def test_summarize_performance_returns_expected_keys() -> None:
    returns = pd.Series([0.01, -0.02, 0.03])
    turnover = pd.Series([1.0, 0.5, 0.5])

    summary = summarize_performance(returns, turnover, transaction_cost_bps=10)

    expected_keys = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "turnover",
        "net_return_after_costs",
    }
    assert set(summary) == expected_keys
