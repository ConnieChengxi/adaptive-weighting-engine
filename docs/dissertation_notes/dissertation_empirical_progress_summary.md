# Dissertation Empirical Progress Summary

## Project Title

Machine-Learning-Assisted Adaptive Factor Weighting for ETF Selection and Risk-Adjusted Portfolio Performance

## Current Empirical Setup

### Research design

The empirical design compares a transparent equal-weight benchmark and a static multi-factor ETF selection model against two adaptive weighting extensions:

- Model 0: equal-weight benchmark across the ETF universe
- Model 1: fixed-weight multi-factor scoring model
- Model 2: rolling Information Coefficient model with adaptive factor weights
- Model 3: XGBoost-predicted Information Coefficient model

The machine learning component is not used to predict ETF returns directly. Instead, it predicts factor effectiveness, measured through factor-level Information Coefficients, and converts predicted factor effectiveness into portfolio scoring weights.

### Asset universe and sample

- ETF universe: `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLI`, `XLU`, `XLP`, `XLB`
- Benchmark and market state proxy: `SPY`
- Volatility regime proxy: `VIX`
- Monthly empirical sample used in the aligned backtests: `2010-07-31` to `2024-11-30`
- Number of monthly backtest observations: `173`

### Factors

The portfolio construction framework currently uses four transparent factors:

- Momentum
- Liquidity
- Downside Risk
- Volatility

All factors are standardized cross-sectionally before scoring.

### Model definitions

#### Model 0: Equal-weight benchmark

Model 0 holds the ETF universe with equal portfolio weights each month. It serves as the simplest non-adaptive benchmark and helps distinguish whether the value added in later models comes from factor-informed ranking or from adaptive weighting itself.

#### Model 1: Fixed-weight model

The baseline fixed-weight model uses:

- Momentum: `0.35`
- Liquidity: `0.15`
- Downside Risk: `0.25`
- Volatility: `0.25`

#### Model 2: Rolling IC model

The current main specification is:

- Rolling IC lookback: `12 months`
- Negative IC handling: truncated at zero
- Fallback: fixed-weight baseline
- Main shrinkage rule: `80/20`
  - `0.8 * dynamic IC weights + 0.2 * fixed baseline weights`

Robustness variants already estimated:

- No shrinkage
- `60/40` shrinkage

#### Model 3: XGBoost IC model

The current XGBoost specification is:

- Target: next-period factor IC
- Training window: `36 months`
- Minimum training rows: `24`
- Features:
  - market variables (`SPY` return, `SPY` drawdown, `VIX` level, `VIX` change, high-VIX regime flag)
  - lagged factor IC features
  - rolling IC mean features
- Fallback: fixed-weight baseline
- Current shrinkage rule: `80/20`
  - `0.8 * predicted IC weights + 0.2 * fixed baseline weights`
- Additional regularization has been added to reduce extreme weight concentration

### Backtest design

- Rebalance frequency: monthly
- Portfolio rule for Models 1 to 3: equal-weight among top `3` ranked ETFs
- Portfolio rule for Model 0: equal-weight across the full ETF universe
- Reported metrics include:
  - annualized return
  - annualized volatility
  - Sharpe ratio
  - maximum drawdown
  - Calmar ratio
  - turnover
  - net return after transaction costs

## Key Tables

### Main dissertation tables

- Table 1: [table_1_model_specification_summary.csv](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/tables/table_1_model_specification_summary.csv)
- Table 2: [table_2_main_model_performance_comparison.csv](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/tables/table_2_main_model_performance_comparison.csv)
- Table 3: [table_3_rolling_ic_robustness_comparison.csv](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/tables/table_3_rolling_ic_robustness_comparison.csv)
- Table 4: [table_4_transaction_cost_sensitivity_summary.csv](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/tables/table_4_transaction_cost_sensitivity_summary.csv)

### Data exploration tables

- Table D1: [table_d1_etf_summary_statistics.csv](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/tables/table_d1_etf_summary_statistics.csv)
- XGBoost concentration diagnostics: [table_xgboost_weight_concentration_diagnostics.csv](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/tables/table_xgboost_weight_concentration_diagnostics.csv)

## Key Figures

### Main dissertation figures

- Figure 1: [figure_1_cumulative_portfolio_wealth.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_1_cumulative_portfolio_wealth.png)
- Figure 2: [figure_2_rolling_ic_weight_evolution.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_2_rolling_ic_weight_evolution.png)
- Figure 3: [figure_3_xgboost_weight_concentration.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_3_xgboost_weight_concentration.png)
- Figure 4: [figure_4_gross_versus_net_returns.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_4_gross_versus_net_returns.png)
- Figure 5: [figure_5_transaction_cost_break_even.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_5_transaction_cost_break_even.png)

### Data exploration figures

- Figure D1: [figure_d1_etf_return_correlation_heatmap.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_d1_etf_return_correlation_heatmap.png)
- Figure D2: [figure_d2_factor_score_correlation_heatmap.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_d2_factor_score_correlation_heatmap.png)
- Figure D3a: [figure_d3a_vix_level_threshold.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_d3a_vix_level_threshold.png)
- Figure D3b: [figure_d3b_vix_stress_regime_distribution.png](/Users/chengxima/Desktop/adaptive-weighting-engine/outputs/figures/figure_d3b_vix_stress_regime_distribution.png)

## Main Interpretation

### 1. The fixed-weight model remains a strong benchmark

Under the current setup, the fixed-weight model remains highly competitive relative to both the equal-weight benchmark and the adaptive alternatives. Although the equal-weight benchmark delivers the highest gross and net annualized return, it does so with materially higher volatility and deeper drawdowns. The fixed-weight model produces the highest Sharpe ratio among the main specifications, which suggests that a transparent static weighting rule remains difficult to beat on a risk-adjusted basis in this ETF universe.

### 2. The Rolling IC model adds adaptiveness, but does not yet outperform the benchmark

The rolling IC model is empirically meaningful because it shows how factor importance changes through time, but under the current specification it does not outperform the fixed-weight model on a risk-adjusted basis. The `80/20` version is currently the preferred main specification because it is more stable and more interpretable than the `60/40` version, while still preserving dynamic behavior.

### 3. The XGBoost IC model improves gross return, but turnover is the key limitation

The XGBoost model currently delivers slightly higher gross annualized return than the fixed-weight baseline. However, it also produces materially higher turnover, which makes its advantage fragile once transaction costs are applied. The current transaction cost sensitivity exercise suggests that the XGBoost model begins to underperform the fixed-weight model at around `5 bps`.

### 4. Weight concentration remains an important empirical issue

Even after adding stronger regularization and applying the same `80/20` shrinkage rule used in Model 2, the XGBoost model still produces concentrated factor allocations in a meaningful share of months. This is an important empirical finding because it helps explain why the ML model can improve gross returns while still struggling to dominate after costs.

### 5. The current empirical story is already coherent

At this stage, the dissertation has a clear and defensible empirical narrative:

- a simple equal-weight benchmark provides a clean baseline
- a strong static factor benchmark exists
- adaptive IC-based weighting is intuitive and interpretable, but does not automatically improve performance
- machine-learning-assisted weighting can improve gross return, but its practical value is constrained by turnover and trading frictions

## Recommended Next Step

The next phase should focus on turning the current empirical package into dissertation-ready writing rather than immediately adding more model complexity.

Recommended next steps:

- write Chapter 4 captions for the current tables and figures
- draft the main empirical results section around the existing findings
- if needed, add only targeted extra robustness checks rather than additional new models
