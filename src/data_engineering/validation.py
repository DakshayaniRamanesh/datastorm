import pandas as pd
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Any
from datetime import date

class TransactionModel(BaseModel):
    Outlet_ID: str
    Year: int = Field(ge=2020, le=2026)
    Month: int = Field(ge=1, le=12)
    Distributor_ID: str
    SKU_ID: str
    Volume_Liters: float = Field(gt=0.0)
    Total_Bill_Value: float = Field(gt=0.0)

class OutletMasterModel(BaseModel):
    Outlet_ID: str
    Outlet_Type: str
    Outlet_Size: Optional[str] = None
    Cooler_Count: int = Field(ge=0, le=50)

class OutletCoordinatesModel(BaseModel):
    Outlet_ID: str
    Latitude: float = Field(ge=5.9, le=9.9)  # Sri Lanka Bounding Box
    Longitude: float = Field(ge=79.6, le=82.0)

class DistributorSeasonalityModel(BaseModel):
    Distributor_ID: str
    Year: int = Field(ge=2020, le=2030)
    Month: int = Field(ge=1, le=12)
    Seasonality_Index: str

class HolidayListModel(BaseModel):
    Date: date
    Holiday_Name: str

def validate_dataframe(df: pd.DataFrame, model: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validates a pandas DataFrame row-by-row using a Pydantic model.
    Routes invalid entries to the quarantine store.
    """
    clean_rows = []
    rejected_rows = []
    
    records = df.to_dict(orient="records")
    
    for row in records:
        try:
            model(**row)
            clean_rows.append(row)
        except ValidationError as e:
            # Flatten the errors into a readable string
            error_msgs = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            row["dq_failure_reason"] = error_msgs
            rejected_rows.append(row)
            
    clean_df = pd.DataFrame(clean_rows) if clean_rows else pd.DataFrame(columns=df.columns)
    rejected_df = pd.DataFrame(rejected_rows) if rejected_rows else pd.DataFrame(columns=list(df.columns) + ["dq_failure_reason"])
    
    return clean_df, rejected_df
