import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.predictor import _resolve_model_path


class PredictorLoadingTests(unittest.TestCase):
    def test_resolve_model_path_falls_back_to_legacy_model_when_default_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_default = tmp_path / "best_model.pkl"
            legacy_model = tmp_path / "regression_model.pkl"
            legacy_model.touch()

            with patch("src.predictor.DEFAULT_MODEL_PATH", missing_default), patch(
                "src.predictor.LEGACY_MODEL_PATH", legacy_model
            ):
                self.assertEqual(_resolve_model_path(None), legacy_model)

    def test_resolve_model_path_keeps_explicit_missing_path_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit_path = Path(tmp) / "missing.pkl"

            with self.assertRaisesRegex(FileNotFoundError, "Model file not found"):
                _resolve_model_path(explicit_path)


if __name__ == "__main__":
    unittest.main()
