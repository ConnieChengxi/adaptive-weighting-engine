from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


FIELD_NAMES = ("Close", "High", "Low", "Open", "Volume")


def _parse_yfinance_column(column: str) -> str:
    """Convert saved yfinance tuple-like column names into plain field names."""
    if column == "Date":
        return column

    try:
        parsed = ast.literal_eval(column.replace("_", " "))
    except (SyntaxError, ValueError):
        return column

    if isinstance(parsed, tuple) and parsed:
        field_name = str(parsed[0]).strip()
        if field_name in FIELD_NAMES:
            return field_name
    return column


def clean_raw_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.rename(columns={column: _parse_yfinance_column(column) for column in df.columns}).copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"])
    cleaned = cleaned.sort_values("Date").drop_duplicates(subset=["Date"])

    numeric_columns = [column for column in cleaned.columns if column != "Date"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    return cleaned.reset_index(drop=True)


def load_and_clean_price_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return clean_raw_price_frame(raw)

