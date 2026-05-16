# DataStorm 2026: Latent Monthly Purchase Potential Prediction

## Team: Antigravity

This repository contains the end-to-end analytical pipeline designed to predict the **Maximum Monthly Purchase Potential** (in liters) for 20,000 traditional retail outlets in Sri Lanka for January 2026.

### 1. Project Overview
Our framework shifts from historical-based allocation to **Potential-Based Allocation**. We address the "hidden variable" problem of latent demand by identifying systemic constraints (censoring) and applying a logically defensible "uncapping" methodology.

### 2. Repository Structure
The codebase follows a standard Data Lakehouse architecture:
- `pipeline/bronze/`: Raw ingestion (CSV to Parquet).
- `pipeline/silver/`: Cleaned and sanitized datasets with a robust DQ framework.
- `pipeline/gold/`: Feature engineering and the "Latent Potential" estimation model.
- `pipeline/rejected/`: Quarantined records that failed DQ checks.
- `pipeline/poi_cache/`: Cached geospatial data from OpenStreetMap.
- `output/`: Final predictions and EDA visualizations.
- `run_pipeline.py`: Master script to execute the entire pipeline end-to-end.

### 3. Setup and Usage

#### Prerequisites
- Python 3.10+
- Required libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `pyarrow`, `requests`, `overpy`

To install dependencies:
```bash
pip install pandas numpy scipy matplotlib pyarrow requests overpy
```

#### Running the Pipeline
To run the full pipeline (Bronze -> Silver -> Gold -> EDA):
```bash
python run_pipeline.py
```

To run with **live POI scraping** (Note: this is slow as it hits the Overpass API):
```bash
python run_pipeline.py --poi
```
Or run a sample POI scrape:
```bash
python run_pipeline.py --poi-sample 200
```

### 4. Methodology Highlights
- **Data Forensics**: We implemented a reusable DQ framework to trap system anomalies such as negative volumes, extreme outliers, and GPS dropouts.
- **Censored Demand Modeling**: We utilized a 5-signal composite score to detect outlets hitting supply or credit ceilings (Plateau detection, Distributor Cap proxy, Inter-year stagnation, CV Scoring, and Q4 Suppression).
- **POI Integration**: Catchment drivers (schools, bus stands, hospitals, etc.) were scraped via the Overpass API and used as potential multipliers.
- **SFA Proxy**: Stochastic Frontier Analysis logic was used to calculate the efficiency gap between an outlet and its top-performing peers.

### 5. Final Output
The final predictions are saved to `output/DataStorm_predictions.csv`.
- `Outlet_ID`: Unique identifier for the outlet.
- `Maximum_Monthly_Liters`: Predicted latent potential for January 2026.
