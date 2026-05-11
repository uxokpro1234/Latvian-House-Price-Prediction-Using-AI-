import os
import sys
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.text_utils import normalize_text
from src.model_utils import log1p_transform, expm1_inverse
import joblib
import numpy as np
import pandas as pd
import shap


import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.model_training import engineer_features, prepare_dataframe
except ImportError:
    try:
        from model_training import engineer_features, prepare_dataframe
    except ImportError:
        pass

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "regression_model.pkl"


def _resolve_model_path(model_path: Path | str | None = None) -> Path:
    """Resolve the preferred model path while preserving explicit path errors."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        return path
    if model_path is None and LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    raise FileNotFoundError(f"Model file not found at {path}")


@dataclass
class PredictionOutput:
    current_price: float
    price_1y: float
    price_5y: float
    price_10y: float
    explanation: Dict[str, float]
    scraped_price: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    street: Optional[str] = None
    condition: Optional[str] = None
    images: Optional[list[str]] = None
    location: Optional[str] = None
    rooms: Optional[int] = None
    area: Optional[float] = None
    floor: Optional[int] = None
    total_floors: Optional[int] = None
    building_type: Optional[str] = None
    year: Optional[int] = None


class PricePredictor:
    def __init__(self, model_path: Path | str | None = None) -> None:
        path = _resolve_model_path(model_path)
        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_columns: List[str] = artifact.get("feature_columns", [])
        self.annual_growth_rate: float = artifact.get("annual_growth_rate", 0.0)
        self.district_growth_rates: Dict[str, float] = artifact.get("district_growth_rates", {})
        self.location_stats: Optional[pd.DataFrame] = artifact.get("location_stats", None)

        self.explainer = None
        self.preprocessor = None
        self.shap_feature_names = []

        bg_data = artifact.get("background_data")
        if bg_data is not None:
            try:
                self.preprocessor = self.model.named_steps["prep"]
                model_step = self.model.named_steps["model"]

                bg_transformed = self.preprocessor.transform(bg_data)
                if hasattr(bg_transformed, "toarray"):
                    bg_transformed = bg_transformed.toarray()

                try:
                    self.shap_feature_names = list(self.preprocessor.get_feature_names_out())
                except Exception:
                    self.shap_feature_names = [f"Feature {i}" for i in range(bg_transformed.shape[1])]

                if "GradientBoosting" in str(type(model_step)) or "RandomForest" in str(type(model_step)) or "XGB" in str(type(model_step)):
                   self.explainer = shap.TreeExplainer(model_step)
                else:
                   self.explainer = shap.Explainer(model_step, bg_transformed)

            except Exception:
                self.explainer = None


    def _prepare_row(self, features: Dict) -> pd.DataFrame:
        features = dict(features)

        if "year" not in features or features.get("year") is None:
            features["year"] = pd.Timestamp.now().year

        df = pd.DataFrame([features])

        df = prepare_dataframe(df)

        df, _ = engineer_features(df, location_stats=self.location_stats)

        df_final = pd.DataFrame([
            {col: df[col].iloc[0] if col in df else np.nan for col in self.feature_columns}
        ])
        return df_final

    def predict_current(self, features: Dict) -> float:
        df = self._prepare_row(features)
        pred = self.model.predict(df)
        return float(pred[0])

    def _predict_with_year_shift(self, features: Dict, years_ahead: int) -> Optional[float]:
        shifted = dict(features)
        try:
            shifted["year"] = float(shifted.get("year", pd.Timestamp.now().year)) + years_ahead
        except Exception:
            return None
        df = self._prepare_row(shifted)
        pred = self.model.predict(df)
        return float(pred[0])

    def predict_with_horizons(self, features: Dict) -> PredictionOutput:
        current = self.predict_current(features)

        explanation = {}
        if self.explainer and self.preprocessor:
            try:
                df = self._prepare_row(features)
                X_transformed = self.preprocessor.transform(df)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()

                shap_values = self.explainer.shap_values(X_transformed)
                if isinstance(shap_values, list):
                    vals = shap_values[0]
                else:
                    vals = shap_values

                if len(vals.shape) > 1:
                    vals = vals[0]

                if len(vals) == len(self.shap_feature_names):
                    for name, val in zip(self.shap_feature_names, vals):
                        clean_name = name.replace("num__", "").replace("cat__", "").replace("remainder__", "")
                        explanation[clean_name] = float(val)
            except Exception:
                pass

        p1 = self._predict_with_year_shift(features, 1)
        p5 = self._predict_with_year_shift(features, 5)
        p10 = self._predict_with_year_shift(features, 10)

        loc = features.get("location")
        growth = self.district_growth_rates.get(loc, self.annual_growth_rate) if loc else self.annual_growth_rate
        if not growth:
             growth = self.annual_growth_rate or 0.0

        if abs(growth) < 0.01:
            growth = 0.02
        if p1 is None:
            p1 = current * (1 + growth) ** 1
        if p5 is None:
            p5 = current * (1 + growth) ** 5
        if p10 is None:
            p10 = current * (1 + growth) ** 10

        if abs(p1 - current) < 1e-6 and abs(p5 - current) < 1e-6 and abs(p10 - current) < 1e-6:
            p1 = current * (1 + growth) ** 1
            p5 = current * (1 + growth) ** 5
            p10 = current * (1 + growth) ** 10

        return PredictionOutput(
            current_price=current,
            price_1y=p1,
            price_5y=p5,
            price_10y=p10,
            explanation=explanation,
            scraped_price=features.get("price"),
            lat=features.get("lat"),
            lon=features.get("lon"),
            street=features.get("street"),
            condition=features.get("condition"),
            images=features.get("images"),
            location=features.get("location"),
            rooms=features.get("rooms"),
            area=features.get("area"),
            floor=features.get("floor"),
            total_floors=features.get("total_floors"),
            building_type=features.get("building_type"),
            year=features.get("year"),
        )


def predict_apartment_price(features: Dict, model_path: Path | str | None = None) -> Dict[str, float]:
    predictor = PricePredictor(model_path)
    result = predictor.predict_with_horizons(features)
    return {
        "current_estimated_price": result.current_price,
        "predicted_price_1y": result.price_1y,
        "predicted_price_5y": result.price_5y,
        "predicted_price_10y": result.price_10y,
        "explanation": result.explanation,
    }
