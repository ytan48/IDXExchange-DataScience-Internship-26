import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import add_engineered_features
from src.inference import build_single_input, predict_price
from src.preprocessing_xgboost import (
    CATEGORICAL_FEATURES,
    FLOAT_FEATURES,
    MODEL_FEATURES,
    XGBoostPreprocessor,
)


def property_record(**overrides):
    record = {
        "ValuationDate": "2026-05-15",
        "LivingArea": 2000,
        "BedroomsTotal": 4,
        "BathroomsTotalInteger": 2,
        "LotSizeSquareFeet": 8000,
        "YearBuilt": 2000,
        "GarageSpaces": 2,
        "ParkingTotal": 2,
        "Stories": 1,
        "Latitude": 34.05,
        "Longitude": -118.25,
        "PostalCode": 90001,
        "CountyOrParish": "Los Angeles",
        "MLSAreaMajor": "C01",
        "Levels": "One",
        "PoolPrivateYN": True,
        "ViewYN": False,
        "AttachedGarageYN": "Yes",
        "NewConstructionYN": "N",
        "FireplaceYN": None,
        "City": "Los Angeles",
        "UnifiedSchoolDistrict": "Los Angeles Unified",
    }
    record.update(overrides)
    return record


class FeatureTests(unittest.TestCase):
    def test_notebook_feature_formulas(self):
        raw = pd.DataFrame([property_record()])

        result = add_engineered_features(raw).iloc[0]

        self.assertEqual(result["PropertyAge"], 26)
        self.assertEqual(result["BathBedRatio"], 0.5)
        self.assertEqual(result["LivingAreaPerBedroom"], 500)
        self.assertEqual(result["LivingAreaPerBathroom"], 1000)
        self.assertEqual(result["LotToLivingRatio"], 4)
        self.assertAlmostEqual(result["LogLivingArea"], np.log1p(2000))
        self.assertAlmostEqual(result["LogLotSize"], np.log1p(8000))
        self.assertEqual(result["AmenityKnownCount"], 4)
        self.assertEqual(result["AmenityCount"], 2)
        self.assertAlmostEqual(result["ValuationMonthSin"], 0.5)
        self.assertAlmostEqual(
            result["ValuationMonthCos"],
            -np.sqrt(3) / 2,
        )

    def test_invalid_age_and_zero_denominators_become_missing(self):
        raw = pd.DataFrame(
            [
                property_record(
                    YearBuilt=2030,
                    BedroomsTotal=0,
                    BathroomsTotalInteger=0,
                    LivingArea=0,
                    LotSizeSquareFeet=0,
                )
            ]
        )

        result = add_engineered_features(raw).iloc[0]

        for feature in [
            "PropertyAge",
            "BathBedRatio",
            "LivingAreaPerBedroom",
            "LivingAreaPerBathroom",
            "LotToLivingRatio",
            "LogLivingArea",
            "LogLotSize",
        ]:
            self.assertTrue(pd.isna(result[feature]))

    def test_all_boolean_amenities_from_app_are_counted(self):
        raw = pd.DataFrame(
            [
                property_record(
                    PoolPrivateYN=False,
                    ViewYN=True,
                    AttachedGarageYN=False,
                    NewConstructionYN=False,
                    FireplaceYN=False,
                )
            ]
        )

        result = add_engineered_features(raw).iloc[0]

        self.assertEqual(result["AmenityKnownCount"], 5)
        self.assertEqual(result["AmenityCount"], 1)


class XGBoostPreprocessorTests(unittest.TestCase):
    def setUp(self):
        raw_train = pd.DataFrame(
            [
                property_record(),
                property_record(
                    PostalCode=92001,
                    City="San Diego",
                    CountyOrParish="San Diego",
                    UnifiedSchoolDistrict="San Diego Unified",
                ),
            ]
        )
        self.engineered_train = add_engineered_features(raw_train)
        self.preprocessor = XGBoostPreprocessor().fit(self.engineered_train)

    def test_forces_notebook_column_order_and_dtypes(self):
        transformed = self.preprocessor.transform(self.engineered_train)

        self.assertEqual(transformed.columns.tolist(), MODEL_FEATURES)
        for feature in FLOAT_FEATURES:
            self.assertEqual(str(transformed[feature].dtype), "float64")
        self.assertEqual(str(transformed["AmenityKnownCount"].dtype), "int64")
        for feature in CATEGORICAL_FEATURES:
            self.assertEqual(str(transformed[feature].dtype), "category")

    def test_unknown_inference_categories_become_missing(self):
        unknown = add_engineered_features(
            pd.DataFrame(
                [
                    property_record(
                        PostalCode=99999,
                        City="Unknown City",
                    )
                ]
            )
        )

        transformed = self.preprocessor.transform(unknown)

        self.assertTrue(pd.isna(transformed.loc[0, "PostalCode"]))
        self.assertTrue(pd.isna(transformed.loc[0, "City"]))

    def test_metadata_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "xgboost_preprocessor.json"
            self.preprocessor.save(path)

            loaded = XGBoostPreprocessor.load(path)
            result = loaded.transform(self.engineered_train)

        self.assertEqual(result.columns.tolist(), MODEL_FEATURES)
        self.assertEqual(
            result["City"].cat.categories.tolist(),
            ["Los Angeles", "San Diego"],
        )


class InferenceTests(unittest.TestCase):
    def test_builds_one_typed_row_and_predicts_one_price(self):
        train = add_engineered_features(pd.DataFrame([property_record()]))
        preprocessor = XGBoostPreprocessor().fit(train)

        class FakeModel:
            def predict(self, data):
                self.data = data
                return np.array([765432.1])

        model = FakeModel()

        model_input = build_single_input(property_record(), preprocessor)
        prediction = predict_price(model, property_record(), preprocessor)

        self.assertEqual(model_input.shape, (1, 32))
        self.assertEqual(str(model_input["City"].dtype), "category")
        self.assertAlmostEqual(prediction, 765432.1)
        self.assertEqual(model.data.columns.tolist(), MODEL_FEATURES)

    def test_rejects_close_date_as_an_inference_input(self):
        train = add_engineered_features(pd.DataFrame([property_record()]))
        preprocessor = XGBoostPreprocessor().fit(train)
        invalid_input = property_record()
        invalid_input["CloseDate"] = invalid_input.pop("ValuationDate")

        with self.assertRaisesRegex(ValueError, "use ValuationDate"):
            build_single_input(invalid_input, preprocessor)


if __name__ == "__main__":
    unittest.main()
