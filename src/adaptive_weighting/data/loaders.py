from pathlib import Path

import pandas as pd


def load_price_csv(path: str | Path) -> pd.DataFrame:
    """Load a raw price file saved by the download script."""
    return pd.read_csv(path, parse_dates=["Date"])
