import os
import sys

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import warnings
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import prepare_dataframe, engineer_features, remove_outliers

def validate_model(model_path: Path, data_path: Path, clean_outliers: bool = True):
    print(f"Loading model: {model_path}")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    artifact = joblib.load(model_path)
    model = artifact["model"]
    model_name = artifact.get("best_model_name", "Unknown")
    feature_columns = artifact.get("feature_columns")
    location_stats = artifact.get("location_stats")

    print(f"Model Algorithm: {model_name}")
    print(f"Loading data: {data_path}")
    df_raw = pd.read_csv(data_path)
    df = prepare_dataframe(df_raw)

    if clean_outliers:
        print("Removing outliers for consistent comparison (stdev=3.5)...")
        df = remove_outliers(df, "price", n_std=3.5)

    df, _ = engineer_features(df, location_stats=location_stats)

    if "price" not in df:
        print("Error: 'price' column not found in dataset.")
        return

    y = df["price"]
    X = df.drop(columns=["price"])
    if "listed_date" in X:
        X = X.drop(columns=["listed_date"])

    for col in feature_columns:
        if col not in X:
            X[col] = np.nan

    X = X[feature_columns]

    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)

    print("\n" + "="*30)
    print("Holdout / Self-Test Metrics:")
    print(f"R²:   {r2:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    print("\nRunning 5-Fold Cross-Validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=cv, 
        scoring=['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error']
    )

    print("\n5-Fold Cross-Validation Metrics:")
    print(f"R²:   {np.mean(cv_results['test_r2']):.2f}")
    print(f"MAE:  {np.abs(np.mean(cv_results['test_neg_mean_absolute_error'])):.2f}")
    print(f"MSE:  {np.abs(np.mean(cv_results['test_neg_mean_squared_error'])):.2f}")
    print(f"RMSE: {np.abs(np.mean(cv_results['test_neg_root_mean_squared_error'])):.2f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a trained real estate model.")
    parser.add_argument("--model", type=str, default="models/best_model.pkl", help="Path to model artifact.")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV data for testing.")
    
    args = parser.parse_args()
    validate_model(Path(args.model), Path(args.data))
