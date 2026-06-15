from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
