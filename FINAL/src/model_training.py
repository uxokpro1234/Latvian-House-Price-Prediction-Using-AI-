from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_utils import normalize_text
from src.model_utils import log1p_transform, expm1_inverse

import os
os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

import warnings
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from scipy.stats import randint, uniform

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not available. Install with: pip install catboost")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


@dataclass
class TrainingResult:
    best_model_name: str
    best_model: Pipeline
    metrics: Dict[str, float]
    annual_growth_rate: float
    district_growth_rates: Dict[str, float]
    training_rows: int
    feature_columns: List[str]
    model_path: Path
    background_data: Optional[pd.DataFrame] = None
    location_stats: Optional[pd.DataFrame] = None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias = {
        "op_type": "rental_or_sale",
        "operation": "rental_or_sale",
        "district": "district",
        "city": "city",
        "address": "address",
        "street": "street",
        "sqm": "area",
        "total_area": "area",
        "rooms_count": "rooms",
        "floors_total": "total_floors",
        "totalfloors": "total_floors",
        "house_type": "house_type",
        "house_seria": "house_seria",
        "series": "building_type",
        "built_year": "year",
        "year_built": "year",
        "construction_year": "year",
        "transaction_year": "year",
        "listing_date": "listed_date",
        "date_listed": "listed_date",
        "published_at": "listed_date",
    }
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower)
    df = df.rename(columns={k.lower(): v for k, v in alias.items() if k.lower() in df.columns})
    return df


def _clean_price(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.,]", "", regex=True)
        .str.replace(",", ".", regex=False)
        .replace("", np.nan)
        .astype(float)
    )


def _standardize_operation(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .replace({"for rent": "rent", "rent": "rent", "rental": "rent", "for sale": "sale", "sale": "sale"})
    )


def _consolidate_location(df: pd.DataFrame) -> Optional[pd.Series]:
    cols = [c for c in ["location", "district", "city", "address", "street"] if c in df.columns]
    if not cols:
        return None
    return pd.concat([df[c] for c in cols], axis=1).bfill(axis=1).iloc[:, 0]


def _consolidate_building_type(df: pd.DataFrame) -> Optional[pd.Series]:
    cols = [c for c in ["building_type", "house_seria", "house_type"] if c in df.columns]
    if not cols:
        return None
    return pd.concat([df[c] for c in cols], axis=1).bfill(axis=1).iloc[:, 0]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(df)
    if "price" in df:
        df["price"] = _clean_price(df["price"])
    if "rental_or_sale" in df:
        df["rental_or_sale"] = _standardize_operation(df["rental_or_sale"])
    if "listed_date" in df:
        df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
    for col in ["area", "rooms", "floor", "total_floors", "year"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cleaned = pd.DataFrame()
    if "price" in df:
        cleaned["price"] = df["price"]
    location = _consolidate_location(df)
    if location is not None:
        cleaned["location"] = location.apply(normalize_text)
    # Preserve coordinates if available
    if "lat" in df:
        cleaned["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df:
        cleaned["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    building_type = _consolidate_building_type(df)
    if building_type is not None:
        cleaned["building_type"] = building_type
    for col in ["area", "rooms", "floor", "total_floors", "year", "rental_or_sale", "condition", "street"]:
        if col in df:
            cleaned[col] = df[col]
    if "listed_date" in df:
        cleaned["listed_date"] = df["listed_date"]

    if "floor" in cleaned and "total_floors" in cleaned:
        cleaned["floor_ratio"] = cleaned["floor"] / cleaned["total_floors"]

    if "price" in cleaned:
        cleaned = cleaned.dropna(subset=["price"])
        cleaned = cleaned[cleaned["price"] > 0]
    if "area" in cleaned:
        cleaned = cleaned[cleaned["area"] > 0]
    cleaned = cleaned.drop_duplicates()
    cleaned.reset_index(drop=True, inplace=True)
    return cleaned


def compute_location_stats(df: pd.DataFrame, m: int = 10) -> pd.DataFrame:
    if "location" in df and "price" in df:
        global_mean = df["price"].mean()

        stats = df.groupby("location")["price"].agg(["count", "mean", "median", "std"])

        stats["loc_smooth_price"] = (
            (stats["count"] * stats["mean"]) + (m * global_mean)
        ) / (stats["count"] + m)

        stats = stats.rename(columns={"median": "loc_median", "std": "loc_std", "mean": "loc_raw_mean"})

        return stats[["loc_smooth_price", "loc_median", "loc_std"]]
    return pd.DataFrame()


def engineer_features(df: pd.DataFrame, location_stats: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = df.copy()
    current_year = pd.Timestamp.now().year

    if "year" in df:
        df["age"] = current_year - df["year"]
        df["age_category"] = pd.cut(df["age"], bins=[-np.inf, 10, 30, 50, np.inf],
                                     labels=["new", "modern", "old", "very_old"])

    if "area" in df and "rooms" in df:
        df["area_per_room"] = df["area"] / df["rooms"].replace(0, np.nan)
        df["rooms_x_area"] = df["rooms"] * df["area"]

    if "floor" in df and "total_floors" in df:
        df["floor_ratio"] = df["floor"] / df["total_floors"]
        df["is_top_floor"] = (df["floor"] == df["total_floors"]).astype(float)
        df["is_ground_floor"] = (df["floor"] == 1).astype(float)
        df["floor_position"] = "middle"
        df.loc[df["floor"] == 1, "floor_position"] = "ground"
        df.loc[df["floor"] == df["total_floors"], "floor_position"] = "top"

    df["rental_or_sale"] = df.get("rental_or_sale", "sale")

    if "lat" in df and "lon" in df:
        center_lat, center_lon = 56.9496, 24.1052

        lat = df["lat"].fillna(center_lat)
        lon = df["lon"].fillna(center_lon)

        df["lat_diff"] = lat - center_lat
        df["lon_diff"] = lon - center_lon
        df["distance_from_center"] = np.sqrt(
            (df["lat_diff"] * 111)**2 + (df["lon_diff"] * 85)**2
        )
        df.drop(["lat_diff", "lon_diff"], axis=1, inplace=True, errors="ignore")

    stats_to_return = None
    if location_stats is not None:
        df = df.merge(location_stats, left_on="location", right_index=True, how="left")
    elif "location" in df and "price" in df:
        location_stats = compute_location_stats(df)
        df = df.merge(location_stats, left_on="location", right_index=True, how="left")
        stats_to_return = location_stats

    if "condition" in df and "age" in df:
        df["condition_age_score"] = df["condition"].map({
            "All amenities": 1.0,
            "Partial amenities": 0.5,
            "Without amenities": 0.0
        }).fillna(0.5) * (1 / (df["age"].replace(0, 1) + 1))

    if "area" in df:
        df["area_squared"] = df["area"] ** 2
        df["area_log"] = np.log1p(df["area"])

    return df, stats_to_return


def load_datasets(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            frames.append(prepare_dataframe(df))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", path, exc)
    if not frames:
        raise ValueError("No valid CSV files loaded.")
    combined = pd.concat(frames, ignore_index=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def build_preprocessor(feature_df: pd.DataFrame, *, dense: bool = False) -> Tuple[ColumnTransformer, List[str], List[str]]:
    feature_columns = list(feature_df.columns)
    numeric_features = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(feature_df[col]) and col not in ["listed_date"]
    ]
    categorical_features = [
        col
        for col in feature_columns
        if not pd.api.types.is_numeric_dtype(feature_df[col]) and col not in ["listed_date"]
    ]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    encoder_kwargs = {"handle_unknown": "ignore"}
    if dense:
        encoder_kwargs["sparse_output"] = False
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(**encoder_kwargs)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def fit_with_search(name: str, pipeline: Pipeline, grid: Dict, X_train, y_train, use_randomized: bool = True) -> Pipeline:
    if grid:
        if use_randomized:
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=grid,
                n_iter=30,
                scoring="neg_mean_absolute_error",
                cv=5,
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                scoring="neg_mean_absolute_error",
                cv=5,
                n_jobs=-1,
                verbose=1
            )
        search.fit(X_train, y_train)
        logger.info("%s best params: %s | CV MAE: %.3f", name, search.best_params_, -search.best_score_)
        return search.best_estimator_
    pipeline.fit(X_train, y_train)
    return pipeline


def estimate_growth_rate(df: pd.DataFrame) -> Dict[str, float]:
    def _fit_rate(sub: pd.DataFrame) -> Optional[float]:
        if len(sub) < 10:
            return None
        lr = LinearRegression()
        if "listed_date" in sub and sub["listed_date"].notna().sum() > 10:
            dated = sub.dropna(subset=["listed_date", "price"])
            if len(dated) < 10:
                pass
            else:
                x = dated["listed_date"].map(pd.Timestamp.toordinal).to_numpy().reshape(-1, 1)
                y = np.log(dated["price"].to_numpy())
                lr.fit(x, y)
                daily_rate = lr.coef_[0]
                return float(np.exp(daily_rate * 365.25) - 1)

        if "year" in sub and sub["year"].notna().sum() > 10:
            dated = sub.dropna(subset=["year", "price"])
            if len(dated) < 10:
                return None
            yearly = dated.groupby("year")["price"].median().reset_index()
            if len(yearly) < 3:
                return None
            x = yearly["year"].to_numpy().reshape(-1, 1)
            y = np.log(yearly["price"].to_numpy())
            lr.fit(x, y)
            annual_rate = lr.coef_[0]
            return float(np.exp(annual_rate) - 1)
        return None

    rates = {}

    global_rate = _fit_rate(df)
    if global_rate is None:
        global_rate = 0.0
    rates["global"] = float(np.clip(global_rate, -0.05, 0.15))

    if "location" in df:
        top_locs = df["location"].value_counts()
        valid_locs = top_locs[top_locs > 20].index

        for loc in valid_locs:
            sub = df[df["location"] == loc]
            r = _fit_rate(sub)
            if r is not None:
                r = (r + rates["global"]) / 2
                rates[loc] = float(np.clip(r, -0.1, 0.20))
            else:
                rates[loc] = rates["global"]

    return rates


def remove_outliers(df: pd.DataFrame, column: str = "price", n_std: float = 3.0) -> pd.DataFrame:
    mean = df[column].mean()
    std = df[column].std()
    df_clean = df[(df[column] >= mean - n_std * std) & (df[column] <= mean + n_std * std)]
    logger.info(f"Removed {len(df) - len(df_clean)} outliers from {column}")
    return df_clean


def train_models(df: pd.DataFrame, model_path: Optional[Path] = None) -> TrainingResult:
    model_path = model_path or DEFAULT_MODEL_PATH

    df = remove_outliers(df, "price", n_std=3.5)

    df, location_stats = engineer_features(df)
    target = df["price"]
    feature_df = df.drop(columns=["price"])
    if "listed_date" in feature_df:
        feature_df = feature_df.drop(columns=["listed_date"])

    preprocessor, _, _ = build_preprocessor(feature_df, dense=False)

    price_quartiles = pd.qcut(target, q=4, labels=False, duplicates="drop")
    X_train, X_val, y_train, y_val = train_test_split(
        feature_df, target, test_size=0.2, random_state=42, stratify=price_quartiles
    )

    candidates = []
    base_models = []

    rf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [300, 500, 700],
        "model__max_depth": [None, 15, 25, 35],
        "model__min_samples_leaf": [1, 2, 4],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2", None],
    }
    candidates.append(("RandomForest", rf, rf_grid))

    gb = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    gb_grid = {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "model__max_depth": [3, 4, 5, 6],
        "model__min_samples_split": [2, 5, 10],
        "model__subsample": [0.8, 0.9, 1.0],
    }
    candidates.append(("GradientBoosting", gb, gb_grid))

    dense_preprocessor, _, _ = build_preprocessor(feature_df, dense=True)
    hgb = Pipeline(
        steps=[
            ("prep", dense_preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    hgb_grid = {
        "regressor__model__max_depth": [8, 10, 12],
        "regressor__model__learning_rate": [0.05, 0.08, 0.1],
        "regressor__model__max_leaf_nodes": [31, 63, 127],
        "regressor__model__min_samples_leaf": [10, 20, 30],
    }
    hgb_ttr = TransformedTargetRegressor(
        regressor=hgb,
        func=log1p_transform,
        inverse_func=expm1_inverse,
    )
    candidates.append(("HistGradientBoosting", hgb_ttr, hgb_grid))

    if HAS_XGBOOST:
        xgb_model = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", xgb.XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist")),
            ]
        )
        xgb_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [4, 6, 8, 10],
            "model__min_child_weight": [1, 3, 5],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
            "model__gamma": [0, 0.1, 0.3],
        }
        candidates.append(("XGBoost", xgb_model, xgb_grid))

    if HAS_LIGHTGBM:
        lgb_model = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)),
            ]
        )
        lgb_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [6, 8, 10, -1],
            "model__num_leaves": [31, 63, 127],
            "model__min_child_samples": [10, 20, 30],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
        }
        candidates.append(("LightGBM", lgb_model, lgb_grid))

    if HAS_CATBOOST:
        cb_model = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", CatBoostRegressor(random_state=42, verbose=0, thread_count=-1, allow_writing_files=False)),
            ]
        )
        cb_grid = {
            "model__iterations": [200, 400, 600],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__depth": [4, 6, 8, 10],
            "model__l2_leaf_reg": [1, 3, 5],
            "model__border_count": [32, 64, 128],
        }
        candidates.append(("CatBoost", cb_model, cb_grid))

    best_name = None
    best_model = None
    best_metrics = None
    top_models = []

    for name, estimator, grid in candidates:
        logger.info("Training candidate: %s", name)
        model = fit_with_search(name, estimator, grid, X_train, y_train, use_randomized=True)
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        logger.info("%s -> MAE %.2f | RMSE %.2f | R2 %.3f | MAPE %.2f%%",
                   name, metrics["mae"], metrics["rmse"], metrics["r2"], metrics["mape"])

        top_models.append((name, model, metrics["mae"]))

        if best_metrics is None or metrics["mae"] < best_metrics["mae"]:
            best_metrics = metrics
            best_model = model
            best_name = name

    top_models.sort(key=lambda x: x[2])
    top_3 = top_models[:min(3, len(top_models))]

    if len(top_3) >= 2:
        logger.info("Creating stacking ensemble from top models: %s", [m[0] for m in top_3])
        try:
            estimators = [(m[0], m[1]) for m in top_3]
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=5,
                n_jobs=-1
            )
            stacking.fit(X_train, y_train)
            stack_preds = stacking.predict(X_val)
            stack_metrics = compute_metrics(y_val, stack_preds)
            logger.info("StackingEnsemble -> MAE %.2f | RMSE %.2f | R2 %.3f | MAPE %.2f%%",
                       stack_metrics["mae"], stack_metrics["rmse"], stack_metrics["r2"], stack_metrics["mape"])

            if stack_metrics["mae"] < best_metrics["mae"]:
                best_name = "StackingEnsemble"
                best_model = stacking
                best_metrics = stack_metrics
                logger.info("Stacking ensemble selected as best model!")
        except Exception as e:
            logger.warning(f"Stacking ensemble failed: {e}")

    growth_rates = estimate_growth_rate(df)
    feature_columns = list(feature_df.columns)

    bg_data = X_train.sample(n=min(100, len(X_train)), random_state=42)

    return TrainingResult(
        best_model_name=best_name or "Unknown",
        best_model=best_model,
        metrics=best_metrics or {},
        annual_growth_rate=growth_rates.get("global", 0.0),
        district_growth_rates=growth_rates,
        training_rows=int(len(df)),
        feature_columns=feature_columns,
        model_path=model_path,
        background_data=bg_data,
        location_stats=location_stats,
    )


def save_artifact(result: TrainingResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": result.best_model,
        "feature_columns": result.feature_columns,
        "annual_growth_rate": result.annual_growth_rate,
        "district_growth_rates": result.district_growth_rates,
        "training_rows": result.training_rows,
        "metrics": result.metrics,
        "best_model_name": result.best_model_name,
        "background_data": result.background_data,
        "location_stats": result.location_stats,
    }
    joblib.dump(artifact, output_path)
    meta = {
        "best_model_name": result.best_model_name,
        "metrics": result.metrics,
        "training_rows": result.training_rows,
        "training_rows": result.training_rows,
        "annual_growth_rate": result.annual_growth_rate,
        "district_growth_rates": result.district_growth_rates,
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def find_csv_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = list(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return files
    raise FileNotFoundError(f"Path not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-accuracy real estate price model.")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data"), help="CSV file or directory path.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save trained model artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_files = find_csv_files(args.data)
    logger.info("Loading %d CSV file(s)...", len(csv_files))
    df = load_datasets(csv_files)
    logger.info("Training on %d rows...", len(df))
    result = train_models(df, Path(args.output))
    save_artifact(result, Path(args.output))
    logger.info(
        "Saved best model (%s) to %s | MAE=%.2f RMSE=%.2f R2=%.3f | Annual growth=%.4f",
        result.best_model_name,
        args.output,
        result.metrics.get("mae", float("nan")),
        result.metrics.get("rmse", float("nan")),
        result.metrics.get("r2", float("nan")),
        result.annual_growth_rate,
    )


if __name__ == "__main__":
    main()
