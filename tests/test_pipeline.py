import pytest
import pandas as pd
from src.data_engineering.validation import validate_dataframe, TransactionModel

def test_validation_rejects_negative_volume():
    data = [
        {"Outlet_ID": "O1", "Year": 2025, "Month": 1, "Distributor_ID": "D1", "SKU_ID": "S1", "Volume_Liters": -10.0, "Total_Bill_Value": 50.0},
        {"Outlet_ID": "O2", "Year": 2025, "Month": 1, "Distributor_ID": "D1", "SKU_ID": "S1", "Volume_Liters": 10.0, "Total_Bill_Value": 50.0}
    ]
    df = pd.DataFrame(data)
    
    clean, rejected = validate_dataframe(df, TransactionModel)
    
    assert len(clean) == 1
    assert len(rejected) == 1
    assert clean.iloc[0]["Outlet_ID"] == "O2"
    assert rejected.iloc[0]["Outlet_ID"] == "O1"

def test_validation_rejects_future_dates():
    data = [
        {"Outlet_ID": "O1", "Year": 2027, "Month": 1, "Distributor_ID": "D1", "SKU_ID": "S1", "Volume_Liters": 10.0, "Total_Bill_Value": 50.0}
    ]
    df = pd.DataFrame(data)
    
    clean, rejected = validate_dataframe(df, TransactionModel)
    
    assert len(clean) == 0
    assert len(rejected) == 1
