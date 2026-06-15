from __future__ import annotations

import pandas as pd


def compute_monthly_factor_ic(
    panel: pd.DataFrame,
    factor_columns: list[str],
    return_column: str,
) -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []

    for date, group in panel.groupby("Date"):
        row: dict[str, float | pd.Timestamp] = {"Date": date}
        for factor_column in factor_columns:
            valid = group[[factor_column, return_column]].dropna()
            if len(valid) < 2:
                row[f"{factor_column}_ic"] = float("nan")
                continue
            row[f"{factor_column}_ic"] = valid[factor_column].corr(valid[return_column], method="spearman")
        records.append(row)

    return pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
