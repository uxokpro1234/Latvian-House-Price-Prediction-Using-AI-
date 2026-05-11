import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

from src.hypothesis_validation import (
    evaluate_forecast,
    validate_future_dataset,
)


class HypothesisValidationTests(unittest.TestCase):
    def _write_dummy_artifact(self, directory: Path, constant_price: float = 100_000.0) -> Path:
        model = DummyRegressor(strategy="constant", constant=constant_price)
        model.fit(pd.DataFrame({"area": [50.0]}), [constant_price])
        artifact_path = directory / "dummy_model.pkl"
        joblib.dump(
            {
                "model": model,
                "feature_columns": ["area"],
                "annual_growth_rate": 0.02,
                "district_growth_rates": {"global": 0.02},
                "best_model_name": "DummyRegressor",
                "metrics": {},
                "training_rows": 1,
                "location_stats": None,
            },
            artifact_path,
        )
        return artifact_path

    def test_validate_future_dataset_compounds_baseline_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = self._write_dummy_artifact(tmp_path)
            future_csv = tmp_path / "future.csv"
            pd.DataFrame(
                {
                    "area": [50.0],
                    "price": [121_900.0],
                }
            ).to_csv(future_csv, index=False)

            result = validate_future_dataset(
                model_path=artifact_path,
                future_data_path=future_csv,
                baseline_year=2016,
                validation_year=2026,
                max_mape=1.0,
            )

            self.assertEqual(result.baseline_year, 2016)
            self.assertEqual(result.validation_year, 2026)
            self.assertEqual(result.years_ahead, 10)
            self.assertAlmostEqual(result.growth_rate, 0.02)
            self.assertAlmostEqual(result.predictions[0], 121_899.44, places=2)
            self.assertTrue(result.hypothesis_supported)
            self.assertLess(result.metrics["mape"], 1.0)

    def test_evaluate_forecast_returns_regression_metrics_and_decision(self):
        result = evaluate_forecast(
            actual=[100_000.0, 200_000.0],
            predicted=[90_000.0, 210_000.0],
            max_mape=8.0,
        )

        self.assertAlmostEqual(result["mae"], 10_000.0)
        self.assertAlmostEqual(result["mape"], 7.5)
        self.assertTrue(result["hypothesis_supported"])

    def test_validate_future_dataset_requires_actual_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = self._write_dummy_artifact(tmp_path)
            future_csv = tmp_path / "future_without_price.csv"
            pd.DataFrame({"area": [50.0]}).to_csv(future_csv, index=False)

            with self.assertRaisesRegex(ValueError, "actual 2026 prices"):
                validate_future_dataset(
                    model_path=artifact_path,
                    future_data_path=future_csv,
                    baseline_year=2016,
                    validation_year=2026,
                )

    def test_validate_future_dataset_falls_back_to_legacy_default_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = self._write_dummy_artifact(tmp_path)
            future_csv = tmp_path / "future.csv"
            pd.DataFrame({"area": [50.0], "price": [121_900.0]}).to_csv(future_csv, index=False)

            with patch("src.hypothesis_validation.DEFAULT_MODEL_PATH", tmp_path / "missing.pkl"), patch(
                "src.hypothesis_validation.LEGACY_MODEL_PATH", artifact_path
            ):
                result = validate_future_dataset(
                    future_data_path=future_csv,
                    baseline_year=2016,
                    validation_year=2026,
                )

            self.assertAlmostEqual(result.predictions[0], 121_899.44, places=2)

    def test_validate_future_dataset_can_filter_by_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = self._write_dummy_artifact(tmp_path)
            future_csv = tmp_path / "future.csv"
            pd.DataFrame(
                {
                    "op_type": ["For sale", "For rent"],
                    "area": [50.0, 50.0],
                    "price": [121_900.0, 500.0],
                }
            ).to_csv(future_csv, index=False)

            result = validate_future_dataset(
                model_path=artifact_path,
                future_data_path=future_csv,
                baseline_year=2016,
                validation_year=2026,
                operation="sale",
            )

            self.assertEqual(result.row_count, 1)
            self.assertEqual(result.actual_prices, [121_900.0])


if __name__ == "__main__":
    unittest.main()
