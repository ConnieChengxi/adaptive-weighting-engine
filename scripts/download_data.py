from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_CONFIG = ROOT / "config" / "base.yaml"
DATA_CONFIG = ROOT / "config" / "data.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace(" ", "_") for col in df.columns]
    return df.reset_index()


def download_symbol(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No data returned for {symbol}")
    return normalize_columns(data)


def save_symbol_csv(symbol: str, df: pd.DataFrame, output_dir: Path) -> Path:
    safe_symbol = symbol.replace("^", "")
    output_path = output_dir / f"{safe_symbol}.csv"
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    base_cfg = load_yaml(BASE_CONFIG)
    data_cfg = load_yaml(DATA_CONFIG)

    output_dir = ROOT / data_cfg["downloads"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    start_date = base_cfg["project"]["start_date"]
    end_date = base_cfg["project"]["end_date"]
    etfs = base_cfg["universe"]["etfs"]
    benchmark = base_cfg["universe"]["benchmark"]
    vix_symbol = data_cfg["downloads"]["vix_symbol"]

    symbols = [*etfs, benchmark, vix_symbol]
    downloaded: list[Path] = []

    for symbol in symbols:
        print(f"Downloading {symbol}...")
        df = download_symbol(symbol, start_date, end_date)
        path = save_symbol_csv(symbol, df, output_dir)
        downloaded.append(path)
        print(f"Saved {path.relative_to(ROOT)}")

    summary = pd.DataFrame(
        {
            "symbol": [path.stem for path in downloaded],
            "rows": [pd.read_csv(path).shape[0] for path in downloaded],
            "path": [str(path.relative_to(ROOT)) for path in downloaded],
        }
    )
    summary_path = output_dir / "download_manifest.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
