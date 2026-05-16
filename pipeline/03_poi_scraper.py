"""
POI SCRAPER - OpenStreetMap / Overpass API
==========================================
Fetches Points of Interest (POI) near each outlet using the free
Overpass API (no key required).  Results are cached in `pipeline/poi_cache/`
so re-runs never re-hit the API.

Catchment Drivers Targeted
---------------------------
POI Category        | Overpass Tag                   | Demand Rationale
--------------------|--------------------------------|-------------------------------------
Schools             | amenity=school/college/univ    | Youth impulse + canteen supply
Bus Stands          | highway=bus_stop / amenity=bus_station | High footfall / commuter
Hospitals/Clinics   | amenity=hospital/clinic        | High visitor footfall
Tourist Attractions | tourism=*                      | Premium/seasonal demand
Markets / Supermarkets | shop=supermarket/market     | Competing supply signal
Mosques/Temples/Churches | amenity=place_of_worship  | Festive demand spikes
Religious Sites (Buddhist) | amenity=monastery/temple | Sri Lanka specific
Petrol Stations     | amenity=fuel                   | Passing trade proxy
Restaurants/Cafes   | amenity=restaurant/cafe/bar    | Adjacent F&B demand
Residential Areas   | place=village/suburb           | Population density proxy

Usage:
    python pipeline/03_poi_scraper.py [--sample N]
    (--sample N runs on first N outlets only - useful for testing)
"""

import sys
import time
import json
import math
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
SILVER    = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
POI_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Overpass query configuration
# ---------------------------------------------------------------------------
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Radius in metres around each outlet
SEARCH_RADIUS_M = 1000  # 1 km radius for POI catchment

POI_TAGS = {
    "school":        'amenity~"school|college|university|kindergarten"',
    "bus_stop":      'amenity~"bus_station" | highway~"bus_stop"',
    "hospital":      'amenity~"hospital|clinic|doctors|pharmacy"',
    "tourism":       'tourism~"attraction|hotel|museum|viewpoint|zoo|theme_park"',
    "market":        'shop~"supermarket|convenience|mall|department_store"',
    "place_worship": 'amenity~"place_of_worship"',
    "fuel_station":  'amenity~"fuel"',
    "restaurant":    'amenity~"restaurant|cafe|bar|fast_food|food_court"',
    "bank_atm":      'amenity~"bank|atm"',
}

# Overpass QL timeout (seconds)
OVERPASS_TIMEOUT = 25


def build_overpass_query(lat: float, lon: float, radius: int, tag_filter: str) -> str:
    """Build an Overpass QL query for a circular area."""
    return f"""
    [out:json][timeout:{OVERPASS_TIMEOUT}];
    (
      node[{tag_filter}](around:{radius},{lat},{lon});
      way[{tag_filter}](around:{radius},{lat},{lon});
    );
    out count;
    """


def fetch_poi_count(lat: float, lon: float, poi_key: str, tag_filter: str) -> int:
    """Return count of POI type within radius, using cache."""
    cache_key = f"{lat:.5f}_{lon:.5f}_{poi_key}"
    cache_file = POI_CACHE / f"{cache_key}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f).get("count", 0)

    query = build_overpass_query(lat, lon, SEARCH_RADIUS_M, tag_filter)
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=OVERPASS_TIMEOUT + 5,
        )
        resp.raise_for_status()
        data = resp.json()
        count = data.get("elements", [{}])[0].get("tags", {}).get("total", 0)
        # Overpass `out count` returns total in the first element
        elements = data.get("elements", [])
        count = len(elements) if elements else 0
    except Exception as e:
        count = -1  # -1 signals scraping failure; will be treated as 0 in features

    with open(cache_file, "w") as f:
        json.dump({"count": count}, f)

    time.sleep(1.0)  # Respectful rate-limiting to Overpass public API
    return count


def scrape_pois_for_outlets(coords_df: pd.DataFrame, sample_n: int = None) -> pd.DataFrame:
    """
    For each outlet with valid coordinates, fetch POI counts for all categories.

    Parameters
    ----------
    coords_df : Silver outlet_coordinates DataFrame.
    sample_n  : If provided, only process first N outlets (testing mode).

    Returns
    -------
    DataFrame with Outlet_ID + one column per POI category.
    """
    if sample_n:
        coords_df = coords_df.head(sample_n).copy()
        print(f"  [SAMPLE MODE] Processing {sample_n} outlets only.")

    results = []
    total = len(coords_df)

    for i, (_, row) in enumerate(coords_df.iterrows()):
        oid   = row["Outlet_ID"]
        lat   = row["Latitude"]
        lon   = row["Longitude"]
        rec   = {"Outlet_ID": oid, "poi_lat": lat, "poi_lon": lon}

        for poi_key, tag_filter in POI_TAGS.items():
            count = fetch_poi_count(lat, lon, poi_key, tag_filter)
            rec[f"poi_{poi_key}"] = max(count, 0)  # treat -1 as 0

        results.append(rec)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total} outlets scraped.", end="\r")

    print()
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N outlets (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("POI SCRAPER - OpenStreetMap / Overpass API")
    print("=" * 60)

    coords_df = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    print(f"  Outlets with valid coordinates: {len(coords_df):,}")

    poi_df = scrape_pois_for_outlets(coords_df, sample_n=args.sample)
    out_path = ROOT / "pipeline" / "poi_cache" / "poi_features.parquet"
    poi_df.to_parquet(out_path, index=False)
    print(f"\n[OK]  POI features saved -> {out_path}  ({len(poi_df):,} rows)")
    print(f"   Columns: {list(poi_df.columns)}\n")


if __name__ == "__main__":
    main()
