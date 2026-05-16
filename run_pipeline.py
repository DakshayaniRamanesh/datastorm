"""
DataStorm 2026 - Master Pipeline Runner
========================================
Runs the full Bronze -> Silver -> Gold pipeline end to end.

Usage:
    python run_pipeline.py              # Full pipeline (skips live POI scraping)
    python run_pipeline.py --poi        # Also runs live POI scraping (slow, ~8h for 20k outlets)
    python run_pipeline.py --poi-sample 200  # POI on first 200 outlets (testing)
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    ("Bronze Ingestion",      "pipeline/01_bronze_ingest.py",     []),
    ("Silver Cleaning",       "pipeline/02_silver_clean.py",      []),
    ("Gold Features & Model", "pipeline/04_gold_features_model.py",[]),
    ("EDA Dashboard",         "pipeline/05_eda.py",               []),
]

# Ensure child processes use UTF-8 so any remaining unicode in print() works
_child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

def run_step(label: str, script: str, extra_args: list) -> bool:
    print(f"\n{'='*60}")
    print(f">>  {label}")
    print(f"{'='*60}")
    cmd = [sys.executable, str(ROOT / script)] + extra_args
    result = subprocess.run(cmd, cwd=str(ROOT), env=_child_env)
    if result.returncode != 0:
        print(f"[FAIL]  Step failed: {label}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="DataStorm 2026 Pipeline Runner")
    parser.add_argument("--poi", action="store_true",
                        help="Run live POI scraping (slow)")
    parser.add_argument("--poi-sample", type=int, default=None,
                        help="POI scrape sample size (for testing)")
    args = parser.parse_args()

    steps = STEPS.copy()

    # Inject POI step if requested
    if args.poi or args.poi_sample:
        poi_args = []
        if args.poi_sample:
            poi_args = ["--sample", str(args.poi_sample)]
        steps.insert(2, ("POI Scraping (OSM/Overpass)", "pipeline/03_poi_scraper.py", poi_args))

    print("\n>>>  DataStorm 2026 - Latent Potential Pipeline")
    print(f"    Steps to run: {[s[0] for s in steps]}\n")

    for label, script, extra_args in steps:
        ok = run_step(label, script, extra_args)
        if not ok:
            print(f"\nPipeline aborted at: {label}")
            sys.exit(1)

    print("\n" + "="*60)
    print("[DONE]  PIPELINE COMPLETE")
    print("="*60)
    print(f"\n   Predictions -> output/DataStorm_predictions.csv")
    print(f"   Gold table  -> pipeline/gold/gold_features.parquet")
    print(f"   Rejected    -> pipeline/rejected/*.csv")
    print(f"   EDA plot    -> output/eda_dashboard.png\n")


if __name__ == "__main__":
    main()
