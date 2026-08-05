from pathlib import Path

import pandas as pd

from adaptive_weighting.backtest.engine import prepare_panel, run_scored_backtest


def test_prepare_panel_adds_forward_return(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-31", "2020-02-29", "2020-01-31", "2020-02-29"],
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "Close": [100.0, 110.0, 200.0, 180.0],
        }
    ).to_csv(panel_path, index=False)

    prepared = prepare_panel(panel_path)

    aaa_forward = prepared.loc[prepared["symbol"] == "AAA", "next_month_return"].iloc[0]
    bbb_forward = prepared.loc[prepared["symbol"] == "BBB", "next_month_return"].iloc[0]
    assert round(aaa_forward, 6) == 0.1
    assert round(bbb_forward, 6) == -0.1


def test_run_scored_backtest_supports_holding_buffer_framework() -> None:
    tradable = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-02-29",
                ]
            ),
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "score": [3.0, 2.0, 1.0, 2.5, 1.5, 0.5],
            "next_month_return": [0.02, 0.01, -0.01, 0.01, 0.00, -0.02],
            "corwin_schultz_spread": [0.002, 0.002, 0.002, 0.002, 0.002, 0.002],
        }
    )

    selected, portfolio_returns, metrics = run_scored_backtest(
        tradable=tradable,
        score_column="score",
        top_n=2,
        transaction_cost_bps=10.0,
        framework="holding_buffer_top4",
    )

    assert len(selected) == 4
    assert {"portfolio_return", "turnover", "transaction_cost_rate", "net_portfolio_return"} <= set(portfolio_returns.columns)
    assert {"annualized_return", "net_return_after_costs"} <= set(metrics.columns)
