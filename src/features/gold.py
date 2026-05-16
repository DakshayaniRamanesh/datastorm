import pandas as pd
from deltalake import write_deltalake, DeltaTable
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).parent.parent.parent
SILVER_DIR = ROOT / "data" / "silver"
POI_DIR = ROOT / "data" / "poi_features"
GOLD_DIR = ROOT / "data" / "gold"

def build_gold_features():
    os.makedirs(GOLD_DIR, exist_ok=True)
    
    try:
        tx = DeltaTable(str(SILVER_DIR / "transactions")).to_pandas()
        outlets = DeltaTable(str(SILVER_DIR / "outlet_master")).to_pandas()
        coords = DeltaTable(str(SILVER_DIR / "outlet_coordinates")).to_pandas()
        poi = DeltaTable(str(POI_DIR)).to_pandas()
    except Exception as e:
        print(f"Missing data for Gold build: {e}")
        return

    monthly = tx.groupby(["Outlet_ID", "Year", "Month"]).agg({
        "Volume_Liters": "sum",
        "Total_Bill_Value": "sum"
    }).reset_index()
    
    monthly["time_idx"] = monthly["Year"] * 12 + monthly["Month"]
    monthly = monthly.sort_values(["Outlet_ID", "time_idx"])
    monthly["Target_Volume"] = monthly.groupby("Outlet_ID")["Volume_Liters"].shift(-1)
    
    thresholds = monthly.groupby("Outlet_ID")["Volume_Liters"].quantile(0.95).reset_index(name="Censor_Threshold")
    monthly = monthly.merge(thresholds, on="Outlet_ID", how="left")
    
    monthly["Is_Censored"] = (monthly["Target_Volume"] >= monthly["Censor_Threshold"]).astype(int)
    
    gold = monthly.merge(outlets, on="Outlet_ID", how="left")
    if "poi_density" in poi.columns:
        gold = gold.merge(poi[["Outlet_ID", "poi_density"]], on="Outlet_ID", how="left")
    
    gold = gold.dropna(subset=["Target_Volume"])
    
    # Map types
    gold["Outlet_Type"] = gold["Outlet_Type"].astype(str)
    gold["Outlet_Size"] = gold["Outlet_Size"].astype(str)
    
    numeric_cols = gold.select_dtypes(include=[np.number]).columns
    gold[numeric_cols] = gold[numeric_cols].fillna(0)
    
    write_deltalake(str(GOLD_DIR), gold, mode="overwrite")
    print(f"[GOLD] Features built. {len(gold)} monthly records prepared for modeling.")

if __name__ == "__main__":
    build_gold_features()
