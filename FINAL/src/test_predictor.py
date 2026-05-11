import sys
from pathlib import Path
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.predictor import PricePredictor

def test_prediction():
    print("Initializing PricePredictor...")
    try:
        predictor = PricePredictor()
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return

    print("Model loaded successfully.")

    if hasattr(predictor, "location_stats") and predictor.location_stats is not None:
        print(f"Location stats loaded: {len(predictor.location_stats)} locations.")
        print(f"Sample locations: {list(predictor.location_stats.index)[:10]}")

        if "Centrs" in predictor.location_stats.index:
             print("SUCCESS: 'Centrs' found in location_stats")
        else:
             print("FAILURE: 'Centrs' NOT found in location_stats")
    else:
        print("WARNING: Location stats NOT loaded or empty.")

    print(f"Feature columns count: {len(predictor.feature_columns)}")
    if "loc_median" in predictor.feature_columns:
        print("SUCCESS: loc_median is in feature_columns")
    else:
        print("FAILURE: loc_median is NOT in feature_columns")

    features_known = {
        "location": "Centrs", 
        "rooms": 3,
        "area": 75.0,
        "floor": 3,
        "total_floors": 6,
        "house_seria": "Pre-war",
        "house_type": "Masonry",
        "condition": "All amenities"
    }
    
    try:
        price = predictor.predict_current(features_known)
        print(f"Prediction for Centrs (75m2): {price:,.2f} EUR")
        output = predictor.predict_with_horizons(features_known)
        print(f"1Y Forecast: {output.price_1y:,.2f} EUR")
    except Exception as e:
        print(f"Prediction FAILED for known location: {e}")
        import traceback
        traceback.print_exc()

    features_unknown = features_known.copy()
    features_unknown["location"] = "NonExistentDistrict123"
    
    try:
        price = predictor.predict_current(features_unknown)
        print(f"Prediction for Unknown Location: {price:,.2f} EUR")
    except Exception as e:
        print(f"Prediction FAILED for unknown location: {e}")

if __name__ == "__main__":
    test_prediction()
