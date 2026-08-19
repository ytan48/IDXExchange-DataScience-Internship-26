"""Feature engineering used by the final XGBoost model."""

from __future__ import annotations

import numpy as np
import pandas as pd


VALUATION_DATE_FEATURE = "ValuationDate"

SOURCE_NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "YearBuilt",
]

AMENITY_FEATURES = [
    "PoolPrivateYN",
    "ViewYN",
    "AttachedGarageYN",
    "NewConstructionYN",
    "FireplaceYN",
]

ENGINEERED_FEATURES = [
    "PropertyAge",
    "BathBedRatio",
    "LivingAreaPerBedroom",
    "LivingAreaPerBathroom",
    "LotToLivingRatio",
    "LogLivingArea",
    "LogLotSize",
    "AmenityCount",
    "AmenityKnownCount",
    "ValuationMonthSin",
    "ValuationMonthCos",
]

BOOLEAN_MAPPING = {
    True: 1,
    False: 0,
    "True": 1,
    "False": 0,
    "true": 1,
    "false": 0,
    "Yes": 1,
    "No": 0,
    "Y": 1,
    "N": 0,
    1: 1,
    0: 0,
}


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``data`` with the final notebook features added."""

    required_features = [
        VALUATION_DATE_FEATURE,
        *SOURCE_NUMERIC_FEATURES,
        *AMENITY_FEATURES,
    ]
    missing_features = [
        feature for feature in required_features if feature not in data.columns
    ]
    if missing_features:
        raise ValueError(
            "Cannot engineer features; missing columns: "
            + ", ".join(missing_features)
        )

    engineered = data.copy()
    engineered[VALUATION_DATE_FEATURE] = pd.to_datetime(
        engineered[VALUATION_DATE_FEATURE],
        errors="coerce",
    )

    for feature in SOURCE_NUMERIC_FEATURES:
        engineered[feature] = pd.to_numeric(
            engineered[feature],
            errors="coerce",
        )

    engineered["PropertyAge"] = (
        engineered[VALUATION_DATE_FEATURE].dt.year - engineered["YearBuilt"]
    )
    engineered.loc[
        engineered["PropertyAge"] < 0,
        "PropertyAge",
    ] = np.nan

    engineered["BathBedRatio"] = (
        engineered["BathroomsTotalInteger"]
        / engineered["BedroomsTotal"].where(
            engineered["BedroomsTotal"] > 0
        )
    )
    engineered["LivingAreaPerBedroom"] = (
        engineered["LivingArea"]
        / engineered["BedroomsTotal"].where(
            engineered["BedroomsTotal"] > 0
        )
    )
    engineered["LivingAreaPerBathroom"] = (
        engineered["LivingArea"]
        / engineered["BathroomsTotalInteger"].where(
            engineered["BathroomsTotalInteger"] > 0
        )
    )
    engineered["LotToLivingRatio"] = (
        engineered["LotSizeSquareFeet"]
        / engineered["LivingArea"].where(engineered["LivingArea"] > 0)
    )

    engineered["LogLivingArea"] = np.log1p(
        engineered["LivingArea"].where(engineered["LivingArea"] > 0)
    )
    engineered["LogLotSize"] = np.log1p(
        engineered["LotSizeSquareFeet"].where(
            engineered["LotSizeSquareFeet"] > 0
        )
    )

    amenity_data = engineered[AMENITY_FEATURES].apply(
        lambda series: series.map(BOOLEAN_MAPPING)
    )
    amenity_data = amenity_data.apply(pd.to_numeric, errors="coerce")

    engineered["AmenityKnownCount"] = amenity_data.notna().sum(axis=1)
    engineered["AmenityCount"] = amenity_data.sum(
        axis=1,
        min_count=1,
    )

    valuation_month = engineered[VALUATION_DATE_FEATURE].dt.month
    engineered["ValuationMonthSin"] = np.sin(
        2 * np.pi * valuation_month / 12
    )
    engineered["ValuationMonthCos"] = np.cos(
        2 * np.pi * valuation_month / 12
    )

    return engineered
