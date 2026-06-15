from __future__ import annotations

import pandas as pd


def select_top_n_by_score(panel: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    ranked = panel.sort_values(["Date", score_column, "symbol"], ascending=[True, False, True]).copy()
    selected = ranked.groupby("Date", group_keys=False).head(top_n).copy()
    selected["portfolio_weight"] = 1.0 / top_n
    return selected
