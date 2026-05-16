import sys
from src.data_engineering.bronze import ingest_to_bronze
from src.data_engineering.silver import clean_and_validate
from src.features.poi import build_poi_features
from src.features.gold import build_gold_features
from src.models.tobit_lgbm import train_and_predict

def main():
    print("=" * 60)
    print("DataStorm v7.0 - Lakehouse Pipeline")
    print("=" * 60)
    
    print("\n>>> Stage 1: Bronze Ingestion")
    ingest_to_bronze()
    
    print("\n>>> Stage 2: Silver Cleaning & Validation")
    clean_and_validate()
    
    print("\n>>> Stage 3: POI Feature Indexing (H3)")
    build_poi_features()
    
    print("\n>>> Stage 4: Gold Feature Engineering")
    build_gold_features()
    
    print("\n>>> Stage 5: Modeling (Tobit-LGBM)")
    train_and_predict()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
