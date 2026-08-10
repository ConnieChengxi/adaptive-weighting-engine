# Adaptive Weighting Engine

Research codebase for a dissertation on monthly U.S. sector ETF selection under a retained three-signal design, common shrinkage discipline, and turnover-aware execution.

## Current Design

The current retained design is:

- Universe: 9 U.S. sector ETFs
  - `XLB`, `XLE`, `XLF`, `XLI`, `XLK`, `XLP`, `XLU`, `XLV`, `XLY`
- Market benchmark: `SPY`
- Frequency: monthly cross-sectional ranking
- Retained signal block:
  - `12-1` residual momentum
  - winsorised Amihud illiquidity
  - idiosyncratic volatility relative to `SPY`
- Model families:
  - `B0` equal-weight benchmark
  - `S1` static equal-dimension model
  - `A1` rolling-IC adaptive model
  - `L1` Ridge IC
  - `L2` Lasso IC
  - `L3` Elastic Net IC
  - `T1` Random Forest IC
  - `T2` XGBoost IC
- Common shrinkage:
  - selected under ordered validation
  - retained setting: `0.4` baseline weight / `0.6` model-implied IC weight
- Retained execution rule:
  - top-3 portfolio
  - Top-6 holding-buffer boundary

The machine-learning layer predicts dimension-level signal usefulness, not ETF returns directly.

## Repository Layout

```text
config/                 Project configuration
data/raw/               Downloaded raw market data
data/processed/         Processed monthly and daily panels
outputs/backtests/      Backtest-level intermediate and model outputs
outputs/figures/        Dissertation-ready figures
outputs/tables/         Dissertation-ready tables
scripts/                Reproducible entry-point scripts
src/adaptive_weighting/ Core research package
tests/                  Unit tests for core logic
```

## Setup

This repository does not require Poetry for the main research pipeline. The scripts below can be run directly in a standard Python environment.

Use Python `3.10` or newer. The project metadata in `pyproject.toml` requires `>=3.10`, and `pytest` under Poetry will fail if the active interpreter is still `3.9.x`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Raw Data Download

```bash
python3 scripts/download_data.py
```

This populates `data/raw/` with ETF, `SPY`, and `VIX` data.

## Execution Sequence

Run the scripts below in order to rebuild the current retained research pipeline.

### 1. Build processed panels

```bash
python3 scripts/build_features.py
```

Outputs:

- `data/processed/monthly_factor_panel.csv`
- `data/processed/daily_spread_panel.csv`

### 2. Generate retained-dimension diagnostics

```bash
python3 scripts/generate_momentum_diagnostics.py
python3 scripts/generate_liquidity_diagnostics.py
python3 scripts/generate_volatility_diagnostics.py
```

These scripts generate the retained proxy-comparison tables and figures used in the methodology and appendix material.

### 3. Run common shrinkage selection

```bash
python3 scripts/run_common_shrinkage_selection.py
```

Key outputs:

- `outputs/tables/table_sh1_common_shrinkage_validation_grid.csv`
- `outputs/tables/table_sh2_common_shrinkage_selection_summary.csv`
- `outputs/tables/table_sh3_common_shrinkage_test_comparison.csv`
- `outputs/tables/table_sh4_common_shrinkage_selection_conclusion.csv`

### 4. Run repeated walk-forward validation

```bash
python3 scripts/run_repeated_walkforward_family_comparison.py
```

Key outputs:

- `outputs/tables/table_wf1_repeated_walkforward_fold_selection.csv`
- `outputs/tables/table_wf2_repeated_walkforward_test_results.csv`
- `outputs/tables/table_wf3_repeated_walkforward_family_comparison.csv`

### 5. Run turnover-framework comparison

```bash
python3 scripts/run_turnover_framework_backtests.py
```

This generates the framework-level backtest artifacts used for:

- turnover framework comparison
- holding-buffer sensitivity
- Appendix F execution diagnostics

### 6. Generate contribution and design figures

```bash
python3 scripts/generate_factor_contribution_assets.py
```

### 7. Generate Appendix I liquidity-cost evidence

```bash
python3 scripts/generate_liquidity_cost_exploratory.py
```

This script generates the final Appendix I diagnostic assets. It compares two execution settings:

- `F1`: baseline selections with no additional turnover control
- `F3`: retained common-shrinkage Top-6 holding-buffer traded legs

The script writes:

- `outputs/tables/table_i1_execution_setting_liquidity_linkage.csv`
- `outputs/tables/table_i2_traded_leg_model_summary.csv`
- `outputs/figures/figure_i1_amihud_vs_effective_cost_by_setting.png`
- `outputs/figures/figure_i2_amihud_quintile_unit_cost_by_setting.png`

The F3 traded-leg diagnostic is rebuilt from `outputs/backtests/baseline_equal_weight_benchmark_selections.csv`, so it can be reproduced from the backtest outputs without relying on `data/processed/monthly_factor_panel.csv`.

### 8. Compile report tables and figures

```bash
python3 scripts/generate_report_assets.py
```

This is the final reporting step for the current pipeline. It reads from `outputs/backtests/` and writes cleaned presentation-ready files to:

- `outputs/tables/`
- `outputs/figures/`

## Full Reproduction

For a complete end-to-end rebuild of the current retained design, run:

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

This sequence rebuilds:

- processed monthly and daily panels
- retained proxy-comparison diagnostics
- common shrinkage selection outputs
- repeated walk-forward evidence
- turnover-framework and holding-buffer sensitivity outputs
- contribution, concentration, and model-weight figures
- Appendix I liquidity-cost evidence
- dissertation-ready tables and figures in `outputs/tables/` and `outputs/figures/`
