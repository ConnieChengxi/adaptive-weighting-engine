from __future__ import annotations

import pandas as pd


def select_top_n_by_score(panel: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    ranked = panel.sort_values(["Date", score_column, "symbol"], ascending=[True, False, True]).copy()
    selected = ranked.groupby("Date", group_keys=False).head(top_n).copy()
    selected["portfolio_weight"] = 1.0 / top_n
    return selected


def select_top_n_with_holding_buffer(
    panel: pd.DataFrame,
    score_column: str,
    top_n: int,
    hold_buffer_rank: int,
) -> pd.DataFrame:
    ranked = panel.sort_values(["Date", score_column, "symbol"], ascending=[True, False, True]).copy()
    selected_frames: list[pd.DataFrame] = []
    previous_holdings: list[str] = []

    for _, date_frame in ranked.groupby("Date", sort=True):
        ranked_frame = date_frame.copy().reset_index(drop=True)
        ranked_frame["rank"] = ranked_frame[score_column].rank(method="first", ascending=False).astype(int)

        retained = ranked_frame[
            ranked_frame["symbol"].isin(previous_holdings) & (ranked_frame["rank"] <= hold_buffer_rank)
        ].copy()
        retained = retained.sort_values(["rank", "symbol"]).head(top_n)

        remaining_slots = top_n - len(retained)
        if remaining_slots > 0:
            retained_symbols = set(retained["symbol"].tolist())
            additions = ranked_frame[~ranked_frame["symbol"].isin(retained_symbols)].head(remaining_slots).copy()
            current_selection = pd.concat([retained, additions], ignore_index=True)
        else:
            current_selection = retained.copy()

        current_selection = current_selection.sort_values(["rank", "symbol"]).head(top_n).copy()
        current_selection["portfolio_weight"] = 1.0 / top_n
        selected_frames.append(current_selection.drop(columns=["rank"]))
        previous_holdings = current_selection["symbol"].tolist()

    return pd.concat(selected_frames, ignore_index=True)
