import pandas as pd
from pathlib import Path
from deltalake import write_deltalake, DeltaTable
from src.data_engineering.validation import (
    validate_dataframe, TransactionModel, OutletMasterModel,
    OutletCoordinatesModel, DistributorSeasonalityModel, HolidayListModel
)
import os

ROOT = Path(__file__).parent.parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
REJECTED_DIR = ROOT / "data" / "rejected"

def clean_and_validate():
    os.makedirs(SILVER_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)
    
    datasets = {
        "transactions": TransactionModel,
        "outlet_master": OutletMasterModel,
        "outlet_coordinates": OutletCoordinatesModel,
        "distributor_seasonality": DistributorSeasonalityModel,
        "holiday_list": HolidayListModel
    }
    
    for name, model in datasets.items():
        bronze_path = str(BRONZE_DIR / name)
        if not Path(bronze_path).exists():
            continue
            
        try:
            df = DeltaTable(bronze_path).to_pandas()
        except Exception as e:
            print(f"Skipping {name} due to read error: {e}")
            continue
        
        # Convert date column for Holiday_List to proper date format
        if name == "holiday_list":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
            
        # Pydantic validation
        clean_df, rejected_df = validate_dataframe(df, model)
        
        # Save to Delta Lake
        if not clean_df.empty:
            write_deltalake(str(SILVER_DIR / name), clean_df, mode="overwrite")
        
        # Save rejected records
        if not rejected_df.empty:
            rejected_df.to_csv(REJECTED_DIR / f"{name}_rejected.csv", index=False)
            
        print(f"[SILVER] {name}: {len(clean_df)} clean, {len(rejected_df)} rejected.")

if __name__ == "__main__":
    clean_and_validate()
