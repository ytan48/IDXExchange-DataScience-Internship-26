import math
import unittest
from pathlib import Path

from src.inference import build_single_input, load_artifacts, predict_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "xgboost_final"


@unittest.skipUnless(
    (ARTIFACT_DIR / "model.json").is_file(),
    "final model artifact is not available",
)
class ArtifactBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = load_artifacts(ARTIFACT_DIR)
        cls.property_data = {}
        for field in cls.artifacts.input_schema["fields"]:
            default = field["default"]
            if field["name"] == "ValuationDate":
                default = "2026-08-18"
            cls.property_data[field["name"]] = default

    def test_loads_checksums_schema_and_real_model(self):
        prediction = predict_price(
            self.artifacts.model,
            self.property_data,
            self.artifacts.preprocessor,
        )

        self.assertTrue(math.isfinite(prediction))
        self.assertGreater(prediction, 0)
        self.assertNotIn(
            "CloseDate",
            self.artifacts.manifest["raw_input_features"],
        )
        self.assertIn(
            "ValuationDate",
            self.artifacts.manifest["raw_input_features"],
        )

    def test_valuation_date_drives_month_features(self):
        january_property = {
            **self.property_data,
            "ValuationDate": "2026-01-15",
        }
        july_property = {
            **self.property_data,
            "ValuationDate": "2026-07-15",
        }

        january = build_single_input(
            january_property,
            self.artifacts.preprocessor,
        )
        july = build_single_input(
            july_property,
            self.artifacts.preprocessor,
        )

        self.assertEqual(
            january.loc[0, "PropertyAge"],
            july.loc[0, "PropertyAge"],
        )
        self.assertNotEqual(
            january.loc[0, "ValuationMonthCos"],
            july.loc[0, "ValuationMonthCos"],
        )


if __name__ == "__main__":
    unittest.main()
