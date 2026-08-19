"""Single-property inference for the final XGBoost price model."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .features import add_engineered_features
from .preprocessing_xgboost import MODEL_FEATURES, XGBoostPreprocessor


DEFAULT_ARTIFACT_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "xgboost_final"
)


class PriceModel(Protocol):
    def predict(self, data: pd.DataFrame) -> Sequence[float]: ...


@dataclass(frozen=True)
class ModelArtifacts:
    model: PriceModel
    preprocessor: XGBoostPreprocessor
    manifest: dict[str, Any]
    input_schema: dict[str, Any]
    metrics: dict[str, Any]
    artifact_dir: Path


def load_model(model_path: str | Path) -> PriceModel:
    """Load an XGBRegressor saved with its native ``save_model`` method."""

    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise RuntimeError(
            "xgboost is required to load the price model."
        ) from error

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"XGBoost model not found: {path}")

    model = XGBRegressor()
    model.load_model(path)
    return model


def build_single_input(
    property_data: Mapping[str, Any],
    preprocessor: XGBoostPreprocessor,
) -> pd.DataFrame:
    """Engineer and type one raw property record for XGBoost."""

    if "CloseDate" in property_data:
        raise ValueError(
            "CloseDate is not a valid inference input; use ValuationDate."
        )

    raw_input = pd.DataFrame([dict(property_data)])
    engineered_input = add_engineered_features(raw_input)
    return preprocessor.transform(engineered_input)


def predict_price(
    model: PriceModel,
    property_data: Mapping[str, Any],
    preprocessor: XGBoostPreprocessor,
) -> float:
    """Predict the sale price for one raw property record."""

    model_input = build_single_input(property_data, preprocessor)
    predictions = np.asarray(model.predict(model_input)).reshape(-1)
    if predictions.size != 1:
        raise ValueError(
            "The model must return exactly one prediction for a single input."
        )
    return float(predictions[0])


def load_artifacts(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    *,
    verify_checksums: bool = True,
) -> ModelArtifacts:
    """Load and validate the complete app artifact bundle."""

    directory = Path(artifact_dir)
    manifest = _load_json(directory / "manifest.json")
    input_schema = _load_json(directory / "input_schema.json")
    metrics = _load_json(directory / "metrics.json")

    if manifest.get("artifact_version") != 1:
        raise ValueError("Unsupported model artifact version.")
    if manifest.get("model_features") != MODEL_FEATURES:
        raise ValueError("Model artifact feature list does not match source code.")

    if verify_checksums:
        _verify_checksums(directory)

    model = load_model(directory / manifest["files"]["model"])
    preprocessor = XGBoostPreprocessor.load(
        directory / manifest["files"]["preprocessor"]
    )
    return ModelArtifacts(
        model=model,
        preprocessor=preprocessor,
        manifest=manifest,
        input_schema=input_schema,
        metrics=metrics,
        artifact_dir=directory,
    )


def predict_price_from_artifacts(
    artifact_dir: str | Path,
    property_data: Mapping[str, Any],
) -> float:
    """Load the artifact bundle and predict one property price."""

    artifacts = load_artifacts(artifact_dir)
    return predict_price(
        artifacts.model,
        property_data,
        artifacts.preprocessor,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact JSON must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(directory: Path) -> None:
    checksums = _load_json(directory / "checksums.json")
    for filename, expected_digest in checksums.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        if _sha256(path) != expected_digest:
            raise ValueError(f"Artifact checksum mismatch: {filename}")


def _load_property(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Property input JSON must contain one object.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Predict one property sale price with the final XGBoost model."
    )
    parser.add_argument(
        "--artifacts",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Directory containing the model artifact bundle",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON file containing one raw property record",
    )
    args = parser.parse_args(argv)

    prediction = predict_price_from_artifacts(
        artifact_dir=args.artifacts,
        property_data=_load_property(args.input),
    )
    print(f"Predicted price: ${prediction:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
