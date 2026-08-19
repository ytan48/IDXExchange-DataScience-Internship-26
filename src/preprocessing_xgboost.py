"""Native-categorical preprocessing for the final XGBoost model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .features import ENGINEERED_FEATURES, VALUATION_DATE_FEATURE


TARGET = "ClosePrice"

BASE_NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "YearBuilt",
    "GarageSpaces",
    "ParkingTotal",
    "Stories",
]

ADDED_NUMERIC_FEATURES = [
    "Latitude",
    "Longitude",
]

ENGINEERED_NUMERIC_FEATURES = [
    *ENGINEERED_FEATURES,
]

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + ADDED_NUMERIC_FEATURES
    + ENGINEERED_NUMERIC_FEATURES
)

BASE_CATEGORICAL_FEATURES = [
    "PostalCode",
    "CountyOrParish",
    "MLSAreaMajor",
    "Levels",
    "PoolPrivateYN",
    "ViewYN",
    "AttachedGarageYN",
    "NewConstructionYN",
    "FireplaceYN",
]

ADDED_CATEGORICAL_FEATURES = [
    "City",
    "UnifiedSchoolDistrict",
]

CATEGORICAL_FEATURES = (
    BASE_CATEGORICAL_FEATURES + ADDED_CATEGORICAL_FEATURES
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RAW_INPUT_FEATURES = [
    VALUATION_DATE_FEATURE,
    *BASE_NUMERIC_FEATURES,
    *ADDED_NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]

INTEGER_FEATURES = ["AmenityKnownCount"]
FLOAT_FEATURES = [
    feature for feature in NUMERIC_FEATURES if feature not in INTEGER_FEATURES
]

_METADATA_VERSION = 1


class XGBoostPreprocessor:
    """Preserve training category codes for validation and inference."""

    def __init__(
        self,
        category_levels: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.category_levels_: dict[str, list[str]] | None = None
        if category_levels is not None:
            self.category_levels_ = _validate_category_levels(category_levels)

    def fit(self, data: pd.DataFrame) -> "XGBoostPreprocessor":
        prepared = _prepare_values(data)
        self.category_levels_ = {
            feature: prepared[feature].dropna().unique().tolist()
            for feature in CATEGORICAL_FEATURES
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.category_levels_ is None:
            raise RuntimeError("XGBoostPreprocessor must be fitted before transform.")

        prepared = _prepare_values(data)
        for feature in CATEGORICAL_FEATURES:
            categories = self.category_levels_[feature]
            values = prepared[feature].where(
                prepared[feature].isna()
                | prepared[feature].isin(categories)
            )
            prepared[feature] = pd.Categorical(
                values,
                categories=categories,
            )

        return prepared

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    def save(self, path: str | Path) -> None:
        if self.category_levels_ is None:
            raise RuntimeError("Cannot save an unfitted XGBoostPreprocessor.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": _METADATA_VERSION,
            "model_features": MODEL_FEATURES,
            "float_features": FLOAT_FEATURES,
            "integer_features": INTEGER_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "category_levels": self.category_levels_,
        }
        destination.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostPreprocessor":
        source = Path(path)
        metadata = json.loads(source.read_text(encoding="utf-8"))
        _validate_metadata(metadata)
        return cls(category_levels=metadata["category_levels"])


def _prepare_values(data: pd.DataFrame) -> pd.DataFrame:
    missing_features = [
        feature for feature in MODEL_FEATURES if feature not in data.columns
    ]
    if missing_features:
        raise ValueError(
            "Cannot preprocess XGBoost input; missing columns: "
            + ", ".join(missing_features)
        )

    prepared = data[MODEL_FEATURES].copy()

    for feature in FLOAT_FEATURES:
        prepared[feature] = pd.to_numeric(
            prepared[feature],
            errors="coerce",
        ).astype("float64")

    amenity_known_count = pd.to_numeric(
        prepared["AmenityKnownCount"],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    if amenity_known_count.isna().any():
        prepared["AmenityKnownCount"] = amenity_known_count.astype("float64")
    else:
        prepared["AmenityKnownCount"] = amenity_known_count.astype("int64")

    prepared[NUMERIC_FEATURES] = prepared[NUMERIC_FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    prepared["PostalCode"] = (
        prepared["PostalCode"].astype("Int64").astype("string")
    )
    for feature in CATEGORICAL_FEATURES[1:]:
        prepared[feature] = prepared[feature].astype("string")

    return prepared


def _validate_category_levels(
    category_levels: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    missing_features = [
        feature for feature in CATEGORICAL_FEATURES if feature not in category_levels
    ]
    extra_features = [
        feature for feature in category_levels if feature not in CATEGORICAL_FEATURES
    ]
    if missing_features or extra_features:
        raise ValueError(
            "Category metadata does not match categorical features. "
            f"Missing: {missing_features}; extra: {extra_features}."
        )

    return {
        feature: [str(value) for value in category_levels[feature]]
        for feature in CATEGORICAL_FEATURES
    }


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    expected_schema = {
        "model_features": MODEL_FEATURES,
        "float_features": FLOAT_FEATURES,
        "integer_features": INTEGER_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    if metadata.get("version") != _METADATA_VERSION:
        raise ValueError("Unsupported XGBoost preprocessing metadata version.")

    for key, expected_value in expected_schema.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"XGBoost preprocessing metadata mismatch: {key}.")

    category_levels = metadata.get("category_levels")
    if not isinstance(category_levels, dict):
        raise ValueError("XGBoost preprocessing metadata has no category levels.")
    _validate_category_levels(category_levels)
