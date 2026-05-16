import pandas as pd
from deltalake import write_deltalake, DeltaTable
from pathlib import Path
import h3
import os

ROOT = Path(__file__).parent.parent.parent
SILVER_DIR = ROOT / "data" / "silver"
POI_DIR = ROOT / "data" / "poi_features"

H3_RESOLUTION = 9 

def build_poi_features():
    os.makedirs(POI_DIR, exist_ok=True)
    coords_path = str(SILVER_DIR / "outlet_coordinates")
    if not Path(coords_path).exists():
        return
        
    df = DeltaTable(coords_path).to_pandas()
    
    df["h3_index"] = df.apply(
        lambda row: h3.geo_to_h3(row["Latitude"], row["Longitude"], H3_RESOLUTION),
        axis=1
    )
    
    density = df.groupby("h3_index").size().reset_index(name="poi_density")
    df = df.merge(density, on="h3_index", how="left")
    
    write_deltalake(str(POI_DIR), df, mode="overwrite")
    print(f"[POI] Indexed {len(df)} outlets with H3. Resolution: {H3_RESOLUTION}")

if __name__ == "__main__":
    build_poi_features()
