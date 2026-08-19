"""Train and export the final XGBoost artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .features import ENGINEERED_FEATURES, add_engineered_features
from .preprocessing_xgboost import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RAW_INPUT_FEATURES,
    TARGET,
    XGBoostPreprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "xgboost"
    / "xgboost_engineered_full.parquet"
)
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "xgboost_final"

TRAIN_START_MONTH = pd.Period("2024-01", freq="M")
LOWER_QUANTILE = 0.005
UPPER_QUANTILE = 0.995

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
}

MODEL_FILENAME = "model.json"
PREPROCESSOR_FILENAME = "preprocessor.json"
FEATURE_SCHEMA_FILENAME = "feature_schema.json"
INPUT_SCHEMA_FILENAME = "input_schema.json"
METRICS_FILENAME = "metrics.json"
MANIFEST_FILENAME = "manifest.json"
CHECKSUMS_FILENAME = "checksums.json"


def train_and_export(
    data_path: str | Path = DEFAULT_DATA_PATH,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, Any]:
    source_path = Path(data_path)
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(source_path)
    train_data, validation_data, test_data, split_metadata = _prepare_splits(data)

    preprocessor = XGBoostPreprocessor()
    X_train = preprocessor.fit_transform(train_data[MODEL_FEATURES])
    X_validation = preprocessor.transform(validation_data[MODEL_FEATURES])
    X_test = preprocessor.transform(test_data[MODEL_FEATURES])

    y_train = train_data[TARGET].astype("float64")
    y_validation = validation_data[TARGET].astype("float64")
    y_test = test_data[TARGET].astype("float64")

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)
    metrics = {
        "validation": _regression_metrics(y_validation, validation_predictions),
        "test": _regression_metrics(y_test, test_predictions),
        "rows": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        **split_metadata,
    }

    model.save_model(destination / MODEL_FILENAME)
    preprocessor.save(destination / PREPROCESSOR_FILENAME)

    feature_schema = _build_feature_schema(X_train)
    input_schema = _build_input_schema(X_train)
    _write_json(destination / FEATURE_SCHEMA_FILENAME, feature_schema)
    _write_json(destination / INPUT_SCHEMA_FILENAME, input_schema)
    _write_json(destination / METRICS_FILENAME, metrics)

    manifest = {
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "xgboost.XGBRegressor",
        "target": TARGET,
        "model_features": MODEL_FEATURES,
        "raw_input_features": RAW_INPUT_FEATURES,
        "date_semantics": {
            "training": (
                "Historical CloseDate is copied to ValuationDate before "
                "PropertyAge and valuation-month features are calculated."
            ),
            "inference": (
                "ValuationDate is supplied by the user. CloseDate is forbidden."
            ),
        },
        "model_params": MODEL_PARAMS,
        "training_window": {
            "start_month": metrics["train_start_month"],
            "end_month": metrics["train_end_month"],
            "validation_month": metrics["validation_month"],
            "test_month": metrics["test_month"],
        },
        "source_data": str(source_path.resolve()),
        "source_data_sha256": _sha256(source_path),
        "python_version": platform.python_version(),
        "package_versions": {
            package: version(package)
            for package in ["numpy", "pandas", "scikit-learn", "xgboost"]
        },
        "files": {
            "model": MODEL_FILENAME,
            "preprocessor": PREPROCESSOR_FILENAME,
            "feature_schema": FEATURE_SCHEMA_FILENAME,
            "input_schema": INPUT_SCHEMA_FILENAME,
            "metrics": METRICS_FILENAME,
            "checksums": CHECKSUMS_FILENAME,
        },
    }
    _write_json(destination / MANIFEST_FILENAME, manifest)

    checksum_targets = [
        MODEL_FILENAME,
        PREPROCESSOR_FILENAME,
        FEATURE_SCHEMA_FILENAME,
        INPUT_SCHEMA_FILENAME,
        METRICS_FILENAME,
        MANIFEST_FILENAME,
    ]
    checksums = {
        filename: _sha256(destination / filename)
        for filename in checksum_targets
    }
    _write_json(destination / CHECKSUMS_FILENAME, checksums)

    return {
        "artifact_dir": str(destination.resolve()),
        "metrics": metrics,
        "manifest": manifest,
    }


def _prepare_splits(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prepared = data.copy()
    prepared["CloseDate"] = pd.to_datetime(
        prepared["CloseDate"],
        errors="coerce",
    )
    prepared[TARGET] = pd.to_numeric(prepared[TARGET], errors="coerce")
    prepared = prepared.dropna(subset=["CloseDate", TARGET]).copy()

    # A historical sale is valued on its closing date. Inference never uses it.
    prepared["ValuationDate"] = prepared["CloseDate"]
    prepared = add_engineered_features(prepared)
    prepared["split_month"] = prepared["CloseDate"].dt.to_period("M")

    test_month = prepared["split_month"].max()
    validation_month = test_month - 1

    train_data = prepared[
        (prepared["split_month"] >= TRAIN_START_MONTH)
        & (prepared["split_month"] < validation_month)
    ].copy()
    validation_data = prepared[
        prepared["split_month"] == validation_month
    ].copy()
    test_data = prepared[prepared["split_month"] == test_month].copy()

    lower_price = train_data[TARGET].quantile(LOWER_QUANTILE)
    upper_price = train_data[TARGET].quantile(UPPER_QUANTILE)

    train_data = _filter_target(train_data, lower_price, upper_price)
    validation_data = _filter_target(
        validation_data,
        lower_price,
        upper_price,
    )
    test_data = _filter_target(test_data, lower_price, upper_price)

    return train_data, validation_data, test_data, {
        "train_start_month": str(TRAIN_START_MONTH),
        "train_end_month": str(validation_month - 1),
        "validation_month": str(validation_month),
        "test_month": str(test_month),
        "target_filter": {
            "lower_quantile": LOWER_QUANTILE,
            "upper_quantile": UPPER_QUANTILE,
            "lower_price": float(lower_price),
            "upper_price": float(upper_price),
        },
    }


def _filter_target(
    data: pd.DataFrame,
    lower_price: float,
    upper_price: float,
) -> pd.DataFrame:
    return data[data[TARGET].between(lower_price, upper_price)].copy()


def _regression_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype="float64")
    predicted_values = np.asarray(predicted, dtype="float64")
    residuals = actual_values - predicted_values
    absolute_errors = np.abs(residuals)
    percentage_errors = absolute_errors / actual_values

    return {
        "r2": float(
            1
            - np.square(residuals).sum()
            / np.square(actual_values - actual_values.mean()).sum()
        ),
        "mae": float(absolute_errors.mean()),
        "mape_percent": float(percentage_errors.mean() * 100),
        "mdape_percent": float(np.median(percentage_errors) * 100),
    }


def _build_feature_schema(X_train: pd.DataFrame) -> dict[str, Any]:
    return {
        "artifact_version": 1,
        "target": TARGET,
        "raw_input_features": RAW_INPUT_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_dtypes": {
            feature: str(X_train[feature].dtype)
            for feature in MODEL_FEATURES
        },
        "forbidden_inference_features": ["CloseDate"],
        "valuation_date_feature": "ValuationDate",
    }


def _build_input_schema(X_train: pd.DataFrame) -> dict[str, Any]:
    number_fields = [
        ("LivingArea", "Living area (sq ft)", "float", 10.0),
        ("BedroomsTotal", "Bedrooms", "int", 1),
        ("BathroomsTotalInteger", "Bathrooms", "int", 1),
        ("LotSizeSquareFeet", "Lot size (sq ft)", "float", 100.0),
        ("YearBuilt", "Year built", "int", 1),
        ("GarageSpaces", "Garage spaces", "float", 1.0),
        ("ParkingTotal", "Total parking", "float", 1.0),
        ("Stories", "Stories", "float", 1.0),
        ("Latitude", "Latitude", "float", 0.0001),
        ("Longitude", "Longitude", "float", 0.0001),
    ]
    select_fields = [
        ("PostalCode", "Postal code"),
        ("CountyOrParish", "County"),
        ("MLSAreaMajor", "MLS area"),
        ("Levels", "Levels"),
        ("City", "City"),
        ("UnifiedSchoolDistrict", "Unified school district"),
    ]
    toggle_fields = [
        ("PoolPrivateYN", "Private pool"),
        ("ViewYN", "View"),
        ("AttachedGarageYN", "Attached garage"),
        ("NewConstructionYN", "New construction"),
        ("FireplaceYN", "Fireplace"),
    ]

    fields: list[dict[str, Any]] = [
        {
            "name": "ValuationDate",
            "label": "Valuation date",
            "widget": "date",
            "required": True,
            "default": "today",
        }
    ]

    for name, label, dtype, step in number_fields:
        values = pd.to_numeric(X_train[name], errors="coerce").dropna()
        default = float(values.median())
        if dtype == "int":
            default = int(round(default))
        fields.append(
            {
                "name": name,
                "label": label,
                "widget": "number",
                "dtype": dtype,
                "required": True,
                "default": default,
                "step": step,
            }
        )

    for name, label in select_fields:
        default = _categorical_mode(X_train[name])
        fields.append(
            {
                "name": name,
                "label": label,
                "widget": "select",
                "required": True,
                "default": default,
                "options_source": "preprocessor.category_levels",
            }
        )

    for name, label in toggle_fields:
        default = _categorical_mode(X_train[name])
        fields.append(
            {
                "name": name,
                "label": label,
                "widget": "toggle",
                "required": True,
                "default": str(default).lower() in {"true", "yes", "y", "1"},
            }
        )

    return {
        "artifact_version": 1,
        "raw_input_features": RAW_INPUT_FEATURES,
        "forbidden_features": ["CloseDate"],
        "fields": fields,
    }


def _categorical_mode(values: pd.Series) -> str:
    non_missing = values.dropna()
    if non_missing.empty:
        return ""
    modes = non_missing.mode()
    if not modes.empty:
        return str(modes.iloc[0])
    return str(non_missing.iloc[0])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and export the final XGBoost artifact bundle."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Engineered source parquet",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Artifact output directory",
    )
    args = parser.parse_args(argv)

    result = train_and_export(args.data, args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
