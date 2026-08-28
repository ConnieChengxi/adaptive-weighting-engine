# Adaptive Weighting Engine

Research pipeline supporting the MSc dissertation *Adaptive and Machine-Learning Signal Weighting for Sector ETF Selection: Evidence on Implementable Performance*.

This repository reproduces the final empirical workflow for monthly U.S. sector ETF selection. The dissertation compares benchmark, static, adaptive, and machine-learning weighting rules under the same retained signals, shrinkage rule, execution convention, and transaction-cost treatment.

## Final Design

- Universe: `XLB`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLU`, `XLV`, `XLY`
- Benchmark market series: `SPY`
- Regime context: `VIX`
- Frequency: monthly cross-sectional ranking

Retained signal block:

- `12-1` residual momentum relative to `SPY`
- expanding-winsorised Amihud illiquidity
- idiosyncratic volatility relative to `SPY`

Model set:

- `B0` naive equal-weight benchmark
- `S1` static equal-dimension model
- `A1` rolling-IC adaptive model
- `L1` Ridge IC
- `L2` Lasso IC
- `L3` Elastic Net IC
- `T1` Random Forest IC
- `T2` XGBoost IC

Retained dissertation settings:

- common shrinkage: `0.4` neutral-prior weight + `0.6` model-implied weight
- main execution rule: Top-3 portfolio with Top-6 holding buffer
- primary holdout: `2021-07-31` to `2026-05-31`

Note: `scripts/run_backtest.py` retains the legacy `0.5 / 0.5` defaults for compatibility; the dissertation results use the validation-selected `0.4 / 0.6` setting.

## Repository Layout

```text
config/                 Configuration
data/raw/               Downloaded raw ETF, SPY, and VIX data
data/processed/         Processed monthly and daily panels
outputs/backtests/      Backtest outputs
outputs/figures/        Dissertation figures
outputs/tables/         Dissertation tables
scripts/                Pipeline entry points
src/adaptive_weighting/ Core research package
tests/                  Tests
```

## Environment

Use Python `3.10` or newer.
The same version requirement applies to `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

If starting from a clean checkout:

```bash
python3 scripts/download_data.py
```

Then run:

```bash
python3 scripts/build_features.py
python3 scripts/generate_momentum_diagnostics.py
python3 scripts/generate_liquidity_diagnostics.py
python3 scripts/generate_volatility_diagnostics.py
python3 scripts/run_common_shrinkage_selection.py
python3 scripts/run_repeated_walkforward_family_comparison.py
python3 scripts/run_turnover_framework_backtests.py
python3 scripts/generate_factor_contribution_assets.py
python3 scripts/generate_liquidity_cost_exploratory.py
python3 scripts/generate_report_assets.py
```

`scripts/run_backtest.py` remains available only as a legacy compatibility path for the older generic export flow, which reads config-level `0.5 / 0.5` shrinkage defaults. The canonical dissertation pipeline instead reselects and applies the retained `0.4 / 0.6` setting through `scripts/run_common_shrinkage_selection.py`, so this file is kept for backward compatibility.

## Main Outputs

- `run_common_shrinkage_selection.py`
  - common shrinkage validation and retained setting
- `run_repeated_walkforward_family_comparison.py`
  - repeated walk-forward robustness results
- `run_turnover_framework_backtests.py`
  - execution-framework comparison and holding-buffer sensitivity
- `generate_report_assets.py`
  - dissertation-ready figures and tables

Main retained-result reporting uses the `common_shrinkage_holding_buffer_top6_*` backtest outputs. Framework and holding-buffer comparison outputs come from the `baseline_*`, `pta_*`, and `holding_buffer_top{4,5,6}_*` backtests.
