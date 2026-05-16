import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_DIR  = ROOT / "input"
BRONZE_DIR = ROOT / "pipeline" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILES = {
    "transactions":           "transactions_history_final.csv",
    "outlet_master":          "outlet_master.csv",
    "outlet_coordinates":     "outlet_coordinates.csv",
    "distributor_seasonality":"distributor_seasonality_details.csv",
    "holiday_list":           "holiday_list.csv",
}


def ingest_file(name: str, filename: str) -> None:
    src = INPUT_DIR / filename
    dst = BRONZE_DIR / f"{name}.parquet"

    print(f"  Ingesting  {filename}  ->  {dst.name}", end="  ")
    df = pd.read_csv(src, low_memory=False)
    df.to_parquet(dst, index=False, engine="pyarrow")
    print(f"({len(df):,} rows)")


def main():
    print("=" * 60)
    print("BRONZE LAYER - Raw Ingestion")
    print("=" * 60)
    for name, filename in SOURCE_FILES.items():
        ingest_file(name, filename)
    print("\n[OK]  Bronze ingestion complete.\n")


if __name__ == "__main__":
    main()
