from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "tables"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.generate_liquidity_diagnostics as liquidity_diag

OUTCOME_SPECS = {
    "transaction_cost_rate": "Transaction cost rate",
    "effective_cost_per_unit_turnover": "Effective cost per unit turnover",
}


def safe_log10(series: pd.Series, floor: float = 1e-16) -> pd.Series:
    return np.log10(series.clip(lower=floor))


def regression_rows(linkage_frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sample_iterator = [("Pooled", linkage_frame)] + list(linkage_frame.groupby("model", sort=True))

    for sample_name, frame in sample_iterator:
        for outcome_column, outcome_label in OUTCOME_SPECS.items():
            sample = frame[["avg_selected_winsorised_amihud", outcome_column]].dropna().copy()
            if len(sample) < 3:
                continue

            x_raw = sample["avg_selected_winsorised_amihud"]
            x_log = safe_log10(x_raw)
            y = sample[outcome_column]

            raw_fit = linregress(x_raw, y)
            log_fit = linregress(x_log, y)
            raw_pearson = pearsonr(x_raw, y)
            log_pearson = pearsonr(x_log, y)
            rank_corr = spearmanr(x_raw, y)

            rows.append(
                {
                    "sample": sample_name,
                    "outcome": outcome_label,
                    "n_obs": int(len(sample)),
                    "pearson_r_raw_x": float(raw_pearson.statistic),
                    "pearson_p_raw_x": float(raw_pearson.pvalue),
                    "spearman_rho": float(rank_corr.statistic),
                    "spearman_p": float(rank_corr.pvalue),
                    "ols_slope_raw_x": float(raw_fit.slope),
                    "ols_intercept_raw_x": float(raw_fit.intercept),
                    "ols_r2_raw_x": float(raw_fit.rvalue**2),
                    "ols_p_raw_x": float(raw_fit.pvalue),
                    "delta_y_per_1sd_raw_x": float(raw_fit.slope * x_raw.std(ddof=1)),
                    "delta_y_per_1sd_raw_x_bps": float(raw_fit.slope * x_raw.std(ddof=1) * 10000.0),
                    "pearson_r_log10_x": float(log_pearson.statistic),
                    "pearson_p_log10_x": float(log_pearson.pvalue),
                    "ols_slope_log10_x": float(log_fit.slope),
                    "ols_intercept_log10_x": float(log_fit.intercept),
                    "ols_r2_log10_x": float(log_fit.rvalue**2),
                    "ols_p_log10_x": float(log_fit.pvalue),
                    "delta_y_per_1sd_log10_x": float(log_fit.slope * x_log.std(ddof=1)),
                    "delta_y_per_1sd_log10_x_bps": float(log_fit.slope * x_log.std(ddof=1) * 10000.0),
                }
            )
    return rows


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    linkage_frame = liquidity_diag.build_transaction_cost_linkage_frame()
    summary = pd.DataFrame(regression_rows(linkage_frame))
    output_path = TABLES_DIR / "table_l5_regression_linkage_summary.csv"
    summary.to_csv(output_path, index=False)
    print(f"Saved {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
