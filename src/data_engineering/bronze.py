import pandas as pd
from pathlib import Path
from deltalake import write_deltalake
import os

ROOT = Path(__file__).parent.parent.parent
INPUT_DIR = ROOT / "input"
BRONZE_DIR = ROOT / "data" / "bronze"

SOURCE_FILES = {
    "transactions": "transactions_history_final.csv",
    "outlet_master": "outlet_master.csv",
    "outlet_coordinates": "outlet_coordinates.csv",
    "distributor_seasonality": "distributor_seasonality_details.csv",
    "holiday_list": "holiday_list.csv",
}

def ingest_to_bronze():
    os.makedirs(BRONZE_DIR, exist_ok=True)
    for name, filename in SOURCE_FILES.items():
        src = INPUT_DIR / filename
        if not src.exists():
            print(f"[WARNING] Missing input file: {src}")
            continue
            
        df = pd.read_csv(src, low_memory=False)
        dest_path = str(BRONZE_DIR / name)
        
        # Write to Delta Lake
        write_deltalake(dest_path, df, mode="overwrite")
        print(f"[BRONZE] Ingested {filename} -> {dest_path} ({len(df)} rows)")

if __name__ == "__main__":
    ingest_to_bronze()
