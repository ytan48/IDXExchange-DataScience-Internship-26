import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler
)


TARGET = "ClosePrice"


NUMERIC_FEATURES = [
    "LivingArea",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "LotSizeSquareFeet",
    "YearBuilt",
    "GarageSpaces",
    "ParkingTotal",
    "Stories"
]


CATEGORICAL_FEATURES = [
    "PostalCode",
    "CountyOrParish",
    "MLSAreaMajor",
    "Levels",
    "PoolPrivateYN",
    "ViewYN",
    "AttachedGarageYN",
    "NewConstructionYN",
    "FireplaceYN"
]


FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


LEVEL_ORDER = [
    "One",
    "Two",
    "ThreeOrMore",
    "MultiSplit"
]


def normalize_levels(value):
    if pd.isna(value):
        return np.nan

    values = {
        item.strip()
        for item in str(value).split(",")
    }

    ordered_values = [
        level
        for level in LEVEL_ORDER
        if level in values
    ]

    if not ordered_values:
        return np.nan

    return ",".join(ordered_values)


def prepare_features(data):
    """Apply deterministic cleaning before sklearn preprocessing."""

    data = data[FEATURES].copy()

    data["Levels"] = data["Levels"].apply(
        normalize_levels
    )

    data["MLSAreaMajor"] = data["MLSAreaMajor"].replace(
        "699 - Not Defined",
        np.nan
    )

    for feature in NUMERIC_FEATURES:
        data[feature] = pd.to_numeric(
            data[feature],
            errors="coerce"
        ).astype(float)

    for feature in CATEGORICAL_FEATURES:
        missing_mask = data[feature].isna()

        data[feature] = data[feature].astype("object")

        data.loc[~missing_mask, feature] = (
            data.loc[~missing_mask, feature]
            .astype(str)
        )

        data.loc[missing_mask, feature] = np.nan

    return data


def build_preprocessor(scale_numeric=True):
    """Create a new unfitted preprocessor."""

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        )
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50
                )
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )


def build_model_pipeline(model, scale_numeric=True):
    """Combine deterministic cleaning, preprocessing and model."""

    return Pipeline(
        steps=[
            (
                "prepare",
                FunctionTransformer(
                    prepare_features,
                    validate=False
                )
            ),
            (
                "preprocessor",
                build_preprocessor(
                    scale_numeric=scale_numeric
                )
            ),
            (
                "model",
                model
            )
        ]
    )