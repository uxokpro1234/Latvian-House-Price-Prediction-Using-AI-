from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.model_training import engineer_features, prepare_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models" / "regression_model.pkl"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "models" / "hypothesis_validation_report.json"


@dataclass
class HypothesisValidationResult:
    baseline_year: int
    validation_year: int
    years_ahead: int
    growth_rate: float
    max_mape: float
    model_name: str
    row_count: int
    operation_filter: Optional[str]
    metrics: Dict[str, float]
    hypothesis_supported: bool
    baseline_predictions: List[float]
    predictions: List[float]
    actual_prices: List[float]
    warning: Optional[str] = None


def compound_predictions(
    baseline_predictions: Iterable[float],
    annual_growth_rate: float,
    years_ahead: int,
) -> np.ndarray:
    if years_ahead < 0:
        raise ValueError("validation_year must be greater than or equal to baseline_year.")
    if annual_growth_rate <= -1:
        raise ValueError("annual_growth_rate must be greater than -1.0.")
    values = np.asarray(list(baseline_predictions), dtype=float)
    return values * ((1 + annual_growth_rate) ** years_ahead)


def evaluate_forecast(
    actual: Iterable[float],
    predicted: Iterable[float],
    *,
    max_mape: float = 25.0,
) -> Dict[str, float]:
    actual_arr = np.asarray(list(actual), dtype=float)
    predicted_arr = np.asarray(list(predicted), dtype=float)
    if len(actual_arr) == 0:
        raise ValueError("At least one validation row is required.")
    if len(actual_arr) != len(predicted_arr):
        raise ValueError("actual and predicted arrays must have the same length.")

    mae = float(mean_absolute_error(actual_arr, predicted_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))
    r2 = float(r2_score(actual_arr, predicted_arr)) if len(actual_arr) > 1 else float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((actual_arr - predicted_arr) / np.clip(np.abs(actual_arr), 1e-9, None)) * 100
    mape = float(np.mean(ape))
    median_ape = float(np.median(ape))
    within_10_percent = float(np.mean(ape <= 10.0) * 100)
    within_20_percent = float(np.mean(ape <= 20.0) * 100)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "median_ape": median_ape,
        "within_10_percent": within_10_percent,
        "within_20_percent": within_20_percent,
        "hypothesis_supported": bool(mape <= max_mape),
    }


def _resolve_model_path(model_path: Path | str | None = None) -> Path:
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        return path
    if model_path is None and LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH
    raise FileNotFoundError(f"Model file not found: {path}")


def _load_artifact(model_path: Path | str | None = None) -> Dict:
    path = _resolve_model_path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    artifact = joblib.load(path)
    if "model" not in artifact:
        raise ValueError("Model artifact must contain a 'model' key.")
    if not artifact.get("feature_columns"):
        raise ValueError("Model artifact must contain non-empty 'feature_columns'.")
    return artifact


def _normalize_operation_filter(operation: Optional[str]) -> Optional[str]:
    if operation is None:
        return None
    normalized = str(operation).lower().strip()
    aliases = {"for sale": "sale", "sale": "sale", "for rent": "rent", "rent": "rent", "rental": "rent"}
    if normalized not in aliases:
        raise ValueError("operation must be either 'sale' or 'rent'.")
    return aliases[normalized]


def _prepare_validation_features(
    raw_df: pd.DataFrame,
    artifact: Dict,
    *,
    operation: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = prepare_dataframe(raw_df)
    operation_filter = _normalize_operation_filter(operation)
    if operation_filter is not None:
        if "rental_or_sale" not in df:
            raise ValueError("operation filter requires a rental_or_sale/op_type compatible column.")
        df = df[df["rental_or_sale"] == operation_filter].copy()
    if "price" not in df:
        raise ValueError("Validation data must contain actual 2026 prices in a 'price' compatible column.")
    if df.empty:
        raise ValueError("Validation data has no usable rows after cleaning actual 2026 prices.")

    y_actual = df["price"].astype(float).to_numpy()
    engineered, _ = engineer_features(df, location_stats=artifact.get("location_stats"))
    features = engineered.drop(columns=["price", "listed_date"], errors="ignore")

    feature_columns = list(artifact["feature_columns"])
    for column in feature_columns:
        if column not in features:
            features[column] = np.nan
    return features[feature_columns], y_actual


def validate_future_dataset(
    *,
    model_path: Path | str | None = None,
    future_data_path: Path | str,
    baseline_year: int = 2016,
    validation_year: int = 2026,
    annual_growth_rate: Optional[float] = None,
    max_mape: float = 25.0,
    operation: Optional[str] = None,
) -> HypothesisValidationResult:
    if validation_year <= baseline_year:
        raise ValueError("validation_year must be greater than baseline_year.")

    artifact = _load_artifact(model_path)
    future_df = pd.read_csv(future_data_path)
    operation_filter = _normalize_operation_filter(operation)
    features, actual_prices = _prepare_validation_features(future_df, artifact, operation=operation_filter)

    baseline_predictions = np.asarray(artifact["model"].predict(features), dtype=float)
    growth_rate = (
        float(annual_growth_rate)
        if annual_growth_rate is not None
        else float(artifact.get("annual_growth_rate", 0.0) or 0.0)
    )
    years_ahead = validation_year - baseline_year
    predictions = compound_predictions(baseline_predictions, growth_rate, years_ahead)
    metrics = evaluate_forecast(actual_prices, predictions, max_mape=max_mape)
    warning = None
    if abs(growth_rate) < 1e-12:
        warning = (
            "Annual growth rate is 0.0. The validation measures cross-sectional model quality, "
            "not a real 2016 -> 2026 market forecast. Provide --growth-rate or an external index."
        )

    return HypothesisValidationResult(
        baseline_year=baseline_year,
        validation_year=validation_year,
        years_ahead=years_ahead,
        growth_rate=growth_rate,
        max_mape=max_mape,
        model_name=str(artifact.get("best_model_name", "Unknown")),
        row_count=int(len(actual_prices)),
        operation_filter=operation_filter,
        metrics={key: value for key, value in metrics.items() if key != "hypothesis_supported"},
        hypothesis_supported=bool(metrics["hypothesis_supported"]),
        baseline_predictions=baseline_predictions.tolist(),
        predictions=predictions.tolist(),
        actual_prices=actual_prices.tolist(),
        warning=warning,
    )


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def save_report(result: HypothesisValidationResult, output_path: Path | str = DEFAULT_REPORT_PATH) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(asdict(result))
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a 2016-trained model against actual 2026 prices.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to a trained baseline-year model.")
    parser.add_argument("--future-data", required=True, help="CSV with validation-year listings and actual prices.")
    parser.add_argument("--baseline-year", type=int, default=2016, help="Training data snapshot year.")
    parser.add_argument("--validation-year", type=int, default=2026, help="Actual validation data year.")
    parser.add_argument(
        "--growth-rate",
        type=float,
        default=None,
        help="External annual market growth assumption, e.g. 0.04 for 4%%.",
    )
    parser.add_argument("--max-mape", type=float, default=25.0, help="MAPE threshold for accepting the hypothesis.")
    parser.add_argument("--operation", choices=["sale", "rent"], default=None, help="Optionally validate only sale or rent rows.")
    parser.add_argument("--report-output", default=str(DEFAULT_REPORT_PATH), help="Where to save JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_future_dataset(
        model_path=args.model,
        future_data_path=args.future_data,
        baseline_year=args.baseline_year,
        validation_year=args.validation_year,
        annual_growth_rate=args.growth_rate,
        max_mape=args.max_mape,
        operation=args.operation,
    )
    save_report(result, args.report_output)
    print(json.dumps(_json_safe(asdict(result)), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
