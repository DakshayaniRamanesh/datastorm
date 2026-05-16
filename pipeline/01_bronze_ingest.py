"""
BRONZE LAYER - Raw Ingestion
============================
Reads all source flat files verbatim and saves them as Parquet with no
transformations. The purpose of this layer is to preserve the original
data exactly as it arrived so that every downstream step is fully
reproducible from the raw source.

IMPROVEMENTS OVER v1:
  - [IMP-1] File existence check: missing CSVs are caught cleanly with a
            clear error message instead of a raw Python crash.
  - [IMP-2] Empty file guard: a CSV that loads with 0 rows is flagged
            immediately — silent empty files cause confusing downstream failures.
  - [IMP-3] Schema snapshot: column names and dtypes are saved to
            bronze/schema_snapshot.txt so any upstream schema change is
            immediately detectable in future runs.
  - [IMP-4] Ingestion manifest: bronze/ingestion_manifest.csv records the
            filename, row count, column count, and timestamp of every file
            ingested. This is your audit trail for the judges.
  - [IMP-5] Removed unused imports (os, sys).

Usage:
    python pipeline/01_bronze_ingest.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
INPUT_DIR  = ROOT / "input"
BRONZE_DIR = ROOT / "pipeline" / "bronze"
BRONZE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source -> Bronze registry
# ---------------------------------------------------------------------------
SOURCE_FILES = {
    "transactions":            "transactions_history_final.csv",
    "outlet_master":           "outlet_master.csv",
    "outlet_coordinates":      "outlet_coordinates.csv",
    "distributor_seasonality": "distributor_seasonality_details.csv",
    "holiday_list":            "holiday_list.csv",
}


def ingest_file(name: str, filename: str) -> dict:
    """
    Ingest one CSV file into the Bronze layer as Parquet.

    Returns a manifest record (dict) for the ingestion log.
    Raises FileNotFoundError if the source file does not exist.
    Raises ValueError if the file loads with zero rows.
    """
    src = INPUT_DIR / filename
    dst = BRONZE_DIR / f"{name}.parquet"

    # ------------------------------------------------------------------
    # [IMP-1] File existence check
    # ------------------------------------------------------------------
    if not src.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Source file not found: {src}"
            f"\n        Please place '{filename}' in the '{INPUT_DIR}' folder."
        )

    print(f"  Ingesting  {filename}  ->  {dst.name}", end="  ")

    df = pd.read_csv(src, low_memory=False)

    # ------------------------------------------------------------------
    # [IMP-2] Empty file guard
    # ------------------------------------------------------------------
    if len(df) == 0:
        raise ValueError(
            f"\n[ERROR] File loaded with 0 rows: {filename}"
            f"\n        The file exists but appears empty or unreadable."
        )

    df.to_parquet(dst, index=False, engine="pyarrow")
    print(f"({len(df):,} rows, {len(df.columns)} columns)")

    return {
        "name":         name,
        "filename":     filename,
        "rows":         len(df),
        "columns":      len(df.columns),
        "column_names": ", ".join(df.columns.tolist()),
        "ingested_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_schema_snapshot(records: list) -> None:
    """
    [IMP-3] Write a human-readable schema snapshot to bronze/schema_snapshot.txt
    Records column names and dtypes for every ingested file.
    Useful for detecting upstream schema changes in future runs.
    """
    snapshot_path = BRONZE_DIR / "schema_snapshot.txt"
    lines = [
        "BRONZE LAYER - Schema Snapshot",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
    ]
    for rec in records:
        parquet_path = BRONZE_DIR / f"{rec['name']}.parquet"
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        lines.append(f"\n[{rec['name']}]  ({rec['rows']:,} rows)")
        lines.append(f"  Columns ({len(df.columns)}):")
        for col in df.columns:
            lines.append(f"    {col:<40} {str(df[col].dtype)}")

    with open(snapshot_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Schema snapshot  -> {snapshot_path.name}")


def save_manifest(records: list) -> None:
    """
    [IMP-4] Write ingestion_manifest.csv to bronze/.
    Provides a timestamped audit trail of every ingested file.
    Judges can verify the pipeline ran correctly and see row counts.
    """
    manifest_path = BRONZE_DIR / "ingestion_manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"  Ingestion manifest -> {manifest_path.name}")


def main():
    print("=" * 60)
    print("BRONZE LAYER - Raw Ingestion")
    print("=" * 60)

    manifest_records = []

    for name, filename in SOURCE_FILES.items():
        record = ingest_file(name, filename)
        manifest_records.append(record)

    save_schema_snapshot(manifest_records)
    save_manifest(manifest_records)

    print("\n[OK]  Bronze ingestion complete.\n")


if __name__ == "__main__":
    main()