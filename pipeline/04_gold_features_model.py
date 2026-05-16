"""
GOLD LAYER - Feature Engineering + Potential Estimation
========================================================
Reads all Silver-layer datasets (plus POI features) and produces:
  1. A model-ready Gold feature table (per outlet).
  2. Final Maximum_Monthly_Liters predictions for January 2026.

Methodology: Latent Demand Uncapping via Censored Regression / Tobit-inspired Ceiling
======================================================================================

PROBLEM FRAMING
---------------
Observed monthly volume Y_obs is left-censored:
    Y_obs = min(D_true, C)
where D_true is true latent demand and C is the binding systemic constraint
(credit limit, delivery cap, or stockout).

MULTI-SIGNAL CENSORING DETECTION (5 signals, composited):
  Signal 1 - PLATEAU DETECTION:   Rolling 3-month CV < 10% indicates volume
             is flat (stuck at a cap, not naturally stable).
  Signal 2 - DISTRIBUTOR CAP PROXY: Outlet volume mirrors the distributor's
             own median delivery for the same month (delivery ceiling).
  Signal 3 - INTER-YEAR STAGNATION: YoY growth near zero while market grows
             indicates the outlet is supply-capped, not demand-saturated.
  Signal 4 - LOW CV SCORE: Very consistent volumes (low CV) suggest a hard
             ceiling rather than organic demand patterns.
  Signal 5 - Q4 SUPPRESSION: Beverage demand spikes in Q4. Outlets showing
             no Q4 uplift are likely constrained by supply/credit.

UNCAPPING APPROACH: Probabilistic Ceiling Estimation
----------------------------------------------------
  Potential = jan_base x potential_multiplier x jan_season_factor x growth_factor

Where potential_multiplier = size_factor x type_factor x censoring_uplift
                             x poi_uplift x sfa_uplift

Usage:
    python pipeline/04_gold_features_model.py
"""

import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
SILVER    = ROOT / "pipeline" / "silver"
POI_CACHE = ROOT / "pipeline" / "poi_cache"
GOLD_DIR  = ROOT / "pipeline" / "gold"
OUTPUT    = ROOT / "output"
GOLD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEASONALITY_MULTIPLIER = {
    "Favorable":    1.15,
    "Moderate":     1.00,
    "Un-Favorable": 0.88,
}

SIZE_POTENTIAL_FACTOR = {
    "Extra Large": 1.30,
    "Large":       1.15,
    "Medium":      1.00,
    "Small":       0.88,
}

TYPE_POTENTIAL_FACTOR = {
    "Grocery":  1.10,
    "Hotel":    1.20,
    "Pharmacy": 0.90,
    "Kiosk":    0.85,
    "Eatery":   1.05,
    "Bakery":   0.95,
    "SMMT":     1.25,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_holiday(name: str) -> float:
    """
    Classify holiday impact based on social/consumption signature.
    Returns a weight: 
      - Special (High Impact): 1.15 (Festivals, New Year)
      - Normal (Standard Impact): 1.05 (Poya, minor religious days)
    """
    name = str(name).lower()
    special_keywords = [
        "new year", "thai pongal", "vesak", "christmas", 
        "independence day", "ramazan", "hajji", "deepavali"
    ]
    if any(kw in name for kw in special_keywords):
        return 1.15
    return 1.05


def peer_efficiency_gap(outlet_vol: float, peer_vols: np.ndarray,
                         frontier_pctile: int = 90) -> float:
    """
    Stochastic Frontier Analysis (SFA) proxy.
    Returns the ratio of the peer-frontier volume to the outlet's own volume.
    Values > 1.0 indicate the outlet is underperforming versus its peer group.
    """
    if len(peer_vols) < 3:
        return 1.0
    frontier = np.percentile(peer_vols, frontier_pctile)
    if outlet_vol <= 0:
        return 1.5
    ratio = frontier / outlet_vol
    return float(np.clip(ratio, 1.0, 3.0))   # hard cap at 3x


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("GOLD LAYER - Feature Engineering & Potential Estimation")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Load Silver datasets
    # -----------------------------------------------------------------------
    print("\n[1/7] Loading Silver datasets...")
    tx       = pd.read_parquet(SILVER / "transactions.parquet")
    outlet   = pd.read_parquet(SILVER / "outlet_master.parquet")
    coords   = pd.read_parquet(SILVER / "outlet_coordinates.parquet")
    season   = pd.read_parquet(SILVER / "distributor_seasonality.parquet")
    holidays = pd.read_parquet(SILVER / "holiday_list.parquet")

    # [IMP] Calculate Weighted Holiday Impact
    holidays["Date"] = pd.to_datetime(holidays["Date"])
    holidays["Year"] = holidays["Date"].dt.year
    holidays["Month"] = holidays["Date"].dt.month
    holidays["Weight"] = holidays["Holiday_Name"].apply(classify_holiday)

    # Aggregate to monthly impact score
    monthly_holiday_impact = (
        holidays.groupby(["Year", "Month"])["Weight"]
        .sum()
        .reset_index()
        .rename(columns={"Weight": "holiday_impact_score"})
    )

    # Average January impact (2023-2025)
    hist_jan_impact = monthly_holiday_impact[
        (monthly_holiday_impact["Month"] == 1) & (monthly_holiday_impact["Year"] < 2026)
    ]["holiday_impact_score"].mean()

    # Jan 2026 impact
    jan_2026_impact = monthly_holiday_impact[
        (monthly_holiday_impact["Year"] == 2026) & (monthly_holiday_impact["Month"] == 1)
    ]["holiday_impact_score"].iloc[0] if not monthly_holiday_impact[
        (monthly_holiday_impact["Year"] == 2026) & (monthly_holiday_impact["Month"] == 1)
    ].empty else hist_jan_impact

    holiday_uplift_factor = float(np.clip(jan_2026_impact / (hist_jan_impact + 1e-9), 0.95, 1.10))

    print(f"  Transactions: {len(tx):,}")
    print(f"  Outlets:      {len(outlet):,}")
    print(f"  Coordinates:  {len(coords):,}")
    print(f"  Holiday Uplift (Jan 2026 vs Hist Jan): {holiday_uplift_factor:.3f}")

    # -----------------------------------------------------------------------
    # 2. Aggregate transactions to monthly outlet level
    # -----------------------------------------------------------------------
    print("\n[2/7] Aggregating transactions to monthly outlet level...")
    monthly = (
        tx.groupby(["Outlet_ID", "Year", "Month", "Distributor_ID"])
        .agg(
            monthly_volume=("Volume_Liters", "sum"),
            monthly_revenue=("Total_Bill_Value", "sum"),
            sku_count=("SKU_ID", "nunique"),
            transaction_count=("SKU_ID", "count"),
        )
        .reset_index()
    )
    print(f"  Monthly records: {len(monthly):,}")

    # -----------------------------------------------------------------------
    # 3. Per-outlet historical features with 5-signal censoring detection
    # -----------------------------------------------------------------------
    print("\n[3/7] Computing per-outlet features with multi-signal censoring detection...")

    # Distributor-month median volumes (delivery cap proxy for Signal 2)
    dist_month_median = (
        monthly.groupby(["Distributor_ID", "Month"])["monthly_volume"]
        .median()
        .rename("dist_month_median")
        .reset_index()
    )
    monthly = monthly.merge(dist_month_median, on=["Distributor_ID", "Month"], how="left")

    def outlet_features(grp):
        vols   = grp["monthly_volume"].values
        n      = len(vols)
        months = grp["Month"].values

        # --- Basic stats ---
        mean_vol   = float(np.mean(vols))
        median_vol = float(np.median(vols))
        max_vol    = float(np.max(vols))
        p75_vol    = float(np.percentile(vols, 75))
        p90_vol    = float(np.percentile(vols, 90))
        std_vol    = float(np.std(vols))
        cv         = std_vol / (mean_vol + 1e-9)

        # --- SIGNAL 1: Plateau / rolling CV < 10% ---
        plateau_score = 0.0
        if n >= 3:
            cvs = []
            for i in range(n - 2):
                w = vols[i:i+3]
                cvs.append(np.std(w) / (np.mean(w) + 1e-9))
            plateau_score = float(np.mean(np.array(cvs) < 0.10))

        # --- SIGNAL 2: Tracking distributor delivery cap ---
        dist_med  = grp["dist_month_median"].values
        at_cap    = float(np.mean(np.abs(vols - dist_med) / (dist_med + 1e-9) < 0.15))

        # --- SIGNAL 3: Inter-year stagnation ---
        yearly    = grp.groupby("Year")["monthly_volume"].mean()
        stagnation = 0.0
        yoy_growth = 0.0
        if len(yearly) >= 2:
            rates      = yearly.pct_change().dropna().values
            stagnation = float(np.mean(np.abs(rates) < 0.05))
            yoy_growth = float(np.clip(rates.mean(), -0.30, 0.50))

        # --- SIGNAL 4: Low overall CV ---
        cv_score = float(max(0.0, (0.30 - cv) / 0.30))

        # --- SIGNAL 5: Missing Q4 uplift (beverage demand peak) ---
        q4_vols = vols[np.isin(months, [10, 11, 12])]
        if len(q4_vols) >= 2:
            q4_prem   = q4_vols.mean() / (mean_vol + 1e-9) - 1.0
            q4_supp   = float(np.clip(-q4_prem * 2.0, 0, 1))
        else:
            q4_supp = 0.0

        # --- COMPOSITE CENSORING SCORE ---
        censoring_score = (
            0.30 * plateau_score
            + 0.25 * at_cap
            + 0.20 * stagnation
            + 0.15 * cv_score
            + 0.10 * q4_supp
        )

        # --- January historical base ---
        jan_vols      = vols[np.isin(months, [1])]
        jan_hist_mean = float(jan_vols.mean()) if len(jan_vols) > 0 else mean_vol

        return pd.Series({
            "hist_mean_vol":       mean_vol,
            "hist_median_vol":     median_vol,
            "hist_max_vol":        max_vol,
            "hist_p75_vol":        p75_vol,
            "hist_p90_vol":        p90_vol,
            "hist_std_vol":        std_vol,
            "hist_cv":             cv,
            "hist_months":         n,
            "censoring_score":     float(np.clip(censoring_score, 0, 1)),
            "cens_plateau":        plateau_score,
            "cens_dist_cap":       at_cap,
            "cens_stagnation":     stagnation,
            "cens_cv_score":       cv_score,
            "cens_q4_suppression": q4_supp,
            "yoy_growth":          yoy_growth,
            "jan_hist_mean":       jan_hist_mean,
            "primary_dist":        grp.groupby("Distributor_ID")["monthly_volume"]
                                       .sum().idxmax(),
        })

    outlet_hist = monthly.groupby("Outlet_ID").apply(outlet_features).reset_index()
    n_constrained = (outlet_hist["censoring_score"] > 0.30).sum()
    print(f"  Outlet features computed: {len(outlet_hist):,} outlets")
    print(f"  Outlets with censoring_score > 0.30: {n_constrained:,} ({100*n_constrained/len(outlet_hist):.1f}%)")

    # -----------------------------------------------------------------------
    # 4. Load POI features (if available, else zero-fill)
    # -----------------------------------------------------------------------
    print("\n[4/7] Loading POI features...")
    poi_path = POI_CACHE / "poi_features.parquet"
    if poi_path.exists():
        poi = pd.read_parquet(poi_path)
        poi_cols = [c for c in poi.columns
                    if c.startswith("poi_") and c not in ["poi_lat", "poi_lon"]]
        print(f"  POI feature columns: {poi_cols}")
        print(f"  Outlets with POI data: {len(poi):,}")
    else:
        print("  [INFO] POI features not found - run 03_poi_scraper.py to enrich with POI data.")
        poi = coords.copy()
        poi_cols = []

    # -----------------------------------------------------------------------
    # 5. Merge all feature tables
    # -----------------------------------------------------------------------
    print("\n[5/7] Merging feature tables...")
    gold = outlet_hist.merge(outlet, on="Outlet_ID", how="left")
    gold = gold.merge(coords, on="Outlet_ID", how="left")

    # [IMP] Spatial Density Index (Outlet Clustering)
    # Using KDTree on radians for fast spherical distance approximation
    print("  Computing Spatial Density Index (Outlet Clustering)...")
    valid_coords = coords.dropna(subset=["Latitude", "Longitude"]).copy()
    if not valid_coords.empty:
        rads = np.radians(valid_coords[["Latitude", "Longitude"]].values)
        tree = KDTree(rads)
        
        # 1km radius in radians (approx)
        EARTH_RADIUS_KM = 6371.0
        radius_rad = 1.0 / EARTH_RADIUS_KM
        
        # Count neighbors (minus self)
        density = [len(tree.query_ball_point(p, radius_rad)) - 1 for p in rads]
        valid_coords["spatial_density_index"] = density
        
        gold = gold.merge(valid_coords[["Outlet_ID", "spatial_density_index"]], 
                          on="Outlet_ID", how="left")
        gold["spatial_density_index"] = gold["spatial_density_index"].fillna(0)
    else:
        gold["spatial_density_index"] = 0

    if poi_cols:
        gold = gold.merge(poi[["Outlet_ID"] + poi_cols], on="Outlet_ID", how="left")
        for c in poi_cols:
            gold[c] = gold[c].fillna(0)

    # Map categorical factors
    gold["size_factor"] = gold["Outlet_Size"].map(SIZE_POTENTIAL_FACTOR).fillna(1.0)
    gold["type_factor"] = gold["Outlet_Type"].map(TYPE_POTENTIAL_FACTOR).fillna(1.0)

    # January 2026 seasonality: forward-fill from historical Jan (consistent 3-year pattern)
    jan_season = (
        season[season["Month"] == 1]
        .groupby("Distributor_ID")["Seasonality_Index"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"Seasonality_Index": "jan_seasonality"})
    )
    gold = gold.merge(
        jan_season, left_on="primary_dist", right_on="Distributor_ID",
        how="left", suffixes=("", "_dist")
    )
    gold["jan_seasonality"]  = gold["jan_seasonality"].fillna("Moderate")
    gold["jan_season_factor"]= gold["jan_seasonality"].map(SEASONALITY_MULTIPLIER).fillna(1.0)

    # -----------------------------------------------------------------------
    # 6. POI catchment score
    # -----------------------------------------------------------------------
    if poi_cols:
        weights = {
            "poi_school":        2.0,   # youth impulse purchase
            "poi_bus_stop":      2.0,   # high-footfall commuter catchment
            "poi_hospital":      1.5,   # visitor / staff footfall
            "poi_tourism":       1.8,   # premium + volume demand
            "poi_market":        0.5,   # competing supply signal (mild)
            "poi_place_worship": 1.2,   # festive / daily footfall
            "poi_fuel_station":  1.0,   # passing trade
            "poi_restaurant":    1.3,   # adjacent F&B demand
            "poi_bank_atm":      1.0,   # commercial area proxy
        }
        gold["poi_catchment_raw"] = sum(
            gold.get(col, pd.Series(0, index=gold.index)) * w
            for col, w in weights.items()
            if col in gold.columns
        )
        poi_max = gold["poi_catchment_raw"].quantile(0.95)
        gold["poi_catchment_score"] = (
            gold["poi_catchment_raw"] / (poi_max + 1e-9)
        ).clip(0, 1)
    else:
        gold["poi_catchment_score"] = 0.0

    # -----------------------------------------------------------------------
    # 7. Stochastic Frontier peer efficiency gap
    # -----------------------------------------------------------------------
    print("\n[6/7] Computing peer efficiency gaps (SFA proxy)...")

    peer_p90 = (
        gold.groupby(["Outlet_Type", "Outlet_Size"])["hist_median_vol"]
        .apply(np.array)
        .to_dict()
    )

    def get_peer_gap(row):
        key   = (row["Outlet_Type"], row["Outlet_Size"])
        peers = peer_p90.get(key, np.array([]))
        if len(peers) < 5:
            return 1.0
        return peer_efficiency_gap(row["hist_median_vol"], peers)

    gold["peer_efficiency_gap"] = gold.apply(get_peer_gap, axis=1)

    # -----------------------------------------------------------------------
    # 8. Final Potential Calculation
    # -----------------------------------------------------------------------
    print("\n[7/7] Computing Maximum Monthly Liters for January 2026...")

    # Base volume: use p90 for constrained outlets, p75 otherwise
    gold["base_vol"] = np.where(
        gold["censoring_score"] > 0.30,
        gold["hist_p90_vol"],
        gold["hist_p75_vol"],
    )

    # Censoring uplift: max +60% for highly constrained outlets
    gold["censoring_uplift"] = 1.0 + (gold["censoring_score"] * 0.60)

    # POI uplift: max +25% from catchment richness
    gold["poi_uplift"] = 1.0 + (gold["poi_catchment_score"] * 0.25)

    # Spatial Density uplift: max +10% from being in a high-traffic cluster
    # Scale density index to [0, 1] using 95th percentile as cap
    density_max = gold["spatial_density_index"].quantile(0.95)
    gold["density_score"] = (gold["spatial_density_index"] / (density_max + 1e-9)).clip(0, 1)
    gold["density_uplift"] = 1.0 + (gold["density_score"] * 0.10)

    # SFA peer gap uplift: close 35% of the efficiency gap
    gold["sfa_uplift"] = (
        1.0 + np.clip(gold["peer_efficiency_gap"] - 1.0, 0, 1.5) * 0.35
    )

    # Compose potential multiplier
    gold["potential_multiplier"] = (
        gold["size_factor"]
        * gold["type_factor"]
        * gold["censoring_uplift"]
        * gold["poi_uplift"]
        * gold["density_uplift"]
        * gold["sfa_uplift"]
    ).clip(upper=5.0)

    # Growth factor (YoY trend projected to Jan 2026)
    gold["growth_factor"] = (1.0 + gold["yoy_growth"].clip(-0.20, 0.35))

    # Jan historical base (use actual Jan data where available)
    gold["jan_base"] = gold["jan_hist_mean"].where(
        gold["jan_hist_mean"].notna() & (gold["jan_hist_mean"] > 0),
        gold["hist_median_vol"]
    )

    # FINAL PREDICTION
    gold["Maximum_Monthly_Liters"] = (
        gold["jan_base"]
        * gold["potential_multiplier"]
        * gold["jan_season_factor"]
        * gold["growth_factor"]
        * holiday_uplift_factor
    ).round(2).clip(lower=1.0)

    # -----------------------------------------------------------------------
    # 9. Save outputs
    # -----------------------------------------------------------------------
    extra_cens_cols = ["cens_plateau", "cens_dist_cap", "cens_stagnation",
                       "cens_cv_score", "cens_q4_suppression"]
    gold_cols_to_save = [
        "Outlet_ID", "Outlet_Type", "Outlet_Size", "Cooler_Count",
        "Latitude", "Longitude",
        "hist_mean_vol", "hist_median_vol", "hist_max_vol", "hist_p75_vol", "hist_p90_vol",
        "hist_cv", "hist_months", "censoring_score", "yoy_growth",
        "jan_hist_mean", "primary_dist", "jan_seasonality",
        "size_factor", "type_factor", "jan_season_factor",
        "poi_catchment_score", "peer_efficiency_gap",
        "censoring_uplift", "poi_uplift", "sfa_uplift",
        "potential_multiplier", "growth_factor", "jan_base",
        "Maximum_Monthly_Liters",
    ] + extra_cens_cols + [c for c in poi_cols if c in gold.columns]

    gold_save = gold[[c for c in gold_cols_to_save if c in gold.columns]].copy()
    gold_save.to_parquet(GOLD_DIR / "gold_features.parquet", index=False)
    print(f"  Gold features saved: {GOLD_DIR / 'gold_features.parquet'}")

    predictions = gold_save[["Outlet_ID", "Maximum_Monthly_Liters"]].copy()
    predictions.to_csv(OUTPUT / "DataStorm_predictions.csv", index=False)
    print(f"  Predictions saved:   {OUTPUT / 'DataStorm_predictions.csv'}")

    print("\n[STATS] Prediction Summary:")
    print(predictions["Maximum_Monthly_Liters"].describe().round(2))
    print(f"\nTotal outlets predicted:        {len(predictions):,}")
    print(f"Total potential (Jan 2026):     {predictions['Maximum_Monthly_Liters'].sum():,.0f} L")
    print(f"Avg potential multiplier:       {gold['potential_multiplier'].mean():.2f}x")
    n_hi = (gold["censoring_score"] > 0.30).sum()
    print(f"High-censoring outlets (>0.30): {n_hi:,} ({100*n_hi/len(gold):.1f}%)")

    print("\n[OK]  Gold layer complete.\n")


if __name__ == "__main__":
    main()
