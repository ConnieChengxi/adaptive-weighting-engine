# Adaptive Weighting Engine

Research codebase for a dissertation on machine-learning-assisted adaptive factor weighting for ETF selection and risk-adjusted portfolio performance.

## Research Goal

This project compares four portfolio construction approaches on a U.S. sector ETF universe:

- Model 0: equal-weight benchmark
- Model 1: fixed-weight multi-factor scoring
- Model 2: rolling Information Coefficient dynamic weighting
- Model 3: XGBoost-predicted IC adaptive weighting

The machine learning component is used to predict factor effectiveness, not ETF returns directly.

## Initial Universe

- Sector ETFs: `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLI`, `XLU`, `XLP`, `XLB`
- Benchmark: `SPY`
- Optional extensions later: `XLRE`, `XLC`
- Sample target: `2010-01-01` to `2024-12-31`

## Repository Layout

```text
config/                 Project configuration
data/raw/               Downloaded raw market data
data/interim/           Intermediate research datasets
data/processed/         Clean model-ready datasets
docs/                   Proposal, methodology, dissertation notes
notebooks/              Exploratory notebooks
outputs/                Figures, tables, logs, backtest artifacts
scripts/                Reproducible entry-point scripts
src/adaptive_weighting/ Core research package
tests/                  Unit tests for research logic
```

## Quick Start

1. Create a virtual environment.
2. Install dependencies.
3. Download the initial dataset.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_data.py
```

Downloaded files will be saved into `data/raw/`.

## First Milestones

- Milestone 1: download and validate ETF, SPY, and VIX data
- Milestone 2: construct monthly factor panel
- Milestone 3: implement fixed-weight baseline
- Milestone 4: implement rolling IC dynamic weighting
- Milestone 5: implement XGBoost IC prediction and full walk-forward backtest

## Reproducibility Notes

- Keep raw downloaded data immutable after collection.
- Put experiment choices in `config/` instead of hardcoding them into notebooks.
- Use scripts for repeatable steps and notebooks only for exploration and diagnostics.
