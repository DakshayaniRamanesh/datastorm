import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from deltalake import DeltaTable
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
GOLD_DIR = ROOT / "data" / "gold"
OUTPUT_DIR = ROOT / "output"

def custom_asymmetric_objective(y_true, y_pred, is_censored):
    """
    Custom loss function proxying Tobit regression.
    Penalizes UNDER-prediction heavily for censored records.
    """
    residual = (y_true - y_pred).astype("float")
    grad = np.where(is_censored == 1,
                    np.where(residual > 0, -2 * residual, -0.1 * residual),
                    -2 * residual)
    hess = np.where(is_censored == 1,
                    np.where(residual > 0, 2.0, 0.1),
                    2.0)
    return grad, hess

def train_and_predict():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        df = DeltaTable(str(GOLD_DIR)).to_pandas()
    except Exception as e:
        print(f"Error loading Gold features: {e}")
        return

    df = df.sort_values(["Year", "Month"])
    
    features = ["Volume_Liters", "Total_Bill_Value", "Censor_Threshold"]
    if "poi_density" in df.columns:
        features.append("poi_density")
        
    X = df[features]
    y = df["Target_Volume"]
    censored = df["Is_Censored"]
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        c_train = censored.iloc[train_index]
        
        def lgbm_tobit_obj(preds, train_data):
            labels = train_data.get_label()
            c_vals = c_train.values
            return custom_asymmetric_objective(labels, preds, c_vals)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        params = {'learning_rate': 0.05, 'max_depth': 6, 'verbose': -1}
        model = lgb.train(params, train_data, num_boost_round=50, fobj=lgbm_tobit_obj)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - preds)**2))
        print(f"[Fold {fold}] TimeSeries CV RMSE: {rmse:.2f}")

    print("[MODEL] Training final Tobit-LGBM on all data...")
    def lgbm_tobit_obj_all(preds, train_data):
        labels = train_data.get_label()
        return custom_asymmetric_objective(labels, preds, censored.values)
        
    train_data_all = lgb.Dataset(X, label=y)
    final_model = lgb.train({'learning_rate': 0.05, 'max_depth': 6, 'verbose': -1}, train_data_all, num_boost_round=100, fobj=lgbm_tobit_obj_all)
    
    last_month = df.groupby("Outlet_ID").tail(1)
    preds_jan_2026 = final_model.predict(last_month[features])
    last_month["Maximum_Monthly_Liters"] = preds_jan_2026
    
    out = last_month[["Outlet_ID", "Maximum_Monthly_Liters"]]
    out.to_csv(OUTPUT_DIR / "DataStorm_predictions.csv", index=False)
    print(f"[MODEL] Predictions saved to {OUTPUT_DIR / 'DataStorm_predictions.csv'}")

if __name__ == "__main__":
    train_and_predict()
