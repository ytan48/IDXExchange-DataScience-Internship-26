# Project Memory

Last updated: 2026-08-20

This file is the compressed working memory for the IDX Exchange Data Science
Internship project. It records the current project state, important decisions,
validated workflows, and remaining caveats.

## Project Summary

- Project root: `D:\project-VS\IDX Exchange Data Science Internship`
- Product: California single-family-home valuation app.
- Model: native-categorical `xgboost.XGBRegressor`.
- App: Streamlit, implemented in `app.py`.
- Live demo: `https://tan-idx-exchange-ds.streamlit.app/`
- Latest source data: CRMLS sales through June 2026.
- The app accepts a California address or manual coordinates and uses the U.S.
  Census Geocoder for address lookup.
- Predictions are analytical estimates, not appraisals, lending decisions, or
  list-price recommendations.

## Current Validated Data State

- Raw files: 31 `raw/CRMLSSold*.csv` files covering 2022-01 through 2026-06.
- June 2026 raw file: 24,507 rows.
- Merged dataset: `data/merged_crmls_sold.csv`
  - 818,778 rows.
  - All 31 raw filenames were found in `source_file`.
  - Latest `source_period`: `2026-06`.
  - Latest `CloseDate`: `2026-06-30`.
- Canonical cleaned dataset: `data/processed/crmls_sfr_cleaned.csv`
  - 411,295 rows and 70 columns.
  - Covers 2022-01-01 through 2026-06-30.
- Full XGBoost engineered dataset:
  `data/processed/xgboost/xgboost_engineered_full.parquet`
  - 411,295 rows and 83 columns.
  - Covers 2022-01-01 through 2026-06-30.

## Canonical Processed-Data Layout

```text
data/
├── README.md
├── merged_crmls_sold.csv
├── external/
│   └── DistrictAreas2526.geojson
└── processed/
    ├── crmls_sfr_cleaned.csv
    ├── baseline/
    │   └── baseline_modeling_splits.csv
    └── xgboost/
        ├── xgboost_engineered_full.parquet
        └── splits/
            ├── xgboost_train.parquet
            ├── xgboost_validation.parquet
            └── xgboost_test.parquet
```

Decisions made during this work:

- The canonical cleaning output remains directly under `data/processed/`.
- Baseline derivatives are isolated under `data/processed/baseline/`.
- XGBoost derivatives are isolated under `data/processed/xgboost/`.
- The temporary `future_outputs` folder was deleted because it is not needed.
- All notebook and training-script references were updated to the new names.
- `data/README.md` is an English runbook for refreshing data and models.

## Current Train/Validation/Test Split

Current chronological split:

- Train: 2024-01 through 2026-04, 304,822 rows.
- Validation: 2026-05, 11,880 rows.
- Test: 2026-06, 12,697 rows.

There are two related but independent split paths:

1. Notebook 08 path:
   - `notebooks/07_retraining_model.ipynb` dynamically sets the latest month
     as test and the preceding month as validation.
   - Notebook 07 writes the three Parquet files under
     `data/processed/xgboost/splits/`.
   - `notebooks/08_advance_model.ipynb` reads those three files.
   - Notebook 07 and Notebook 08 were rerun and saved with the June 2026 split.
2. Final app path:
   - `src/train_final_model.py::_prepare_splits()` independently creates the
     split directly from `xgboost_engineered_full.parquet`.
   - It does not read the split Parquet files created by Notebook 07.

## Final App Artifact

- Artifact directory: `artifacts/xgboost_final/`.
- The final artifact was rebuilt from the June 2026 engineered dataset.
- Manifest creation time: `2026-08-19T19:48:28.260005+00:00`.
- Manifest source-data SHA-256 matches the current engineered Parquet.
- All artifact internal checksums match.
- Artifact training window:
  - Start: 2024-01
  - End: 2026-04
  - Validation: 2026-05
  - Test: 2026-06
- Current test metrics:
  - R-squared: `0.90429863`
  - MAE: `$155,260.63`
  - MAPE: `11.8058%`
  - MdAPE: `8.0077%`

The app loads the model bundle through `src/inference.py`. Streamlit caches the
loaded artifact, so stop and restart the Streamlit process after retraining.

### Artifact Portability and Checksum Incident

On 2026-08-19, the Linux Streamlit deployment reported:

```text
Model artifacts are unavailable.
Artifact checksum mismatch: feature_schema.json
```

The artifact contents were valid. The root cause was platform-dependent line
endings: checksums had been generated from Windows CRLF bytes, while Git
normalized the committed JSON files to LF for the Linux deployment.

The permanent fix is present on `main` and `origin/main`:

- `.gitattributes` enforces `text eol=lf` for
  `artifacts/xgboost_final/*.json`.
- `src/train_final_model.py::_write_json()` writes JSON with `newline="\n"`.
- `XGBoostPreprocessor.save()` also writes metadata with `newline="\n"`.
- `artifacts/xgboost_final/checksums.json` contains hashes of the canonical LF
  artifact bytes.
- `tests/test_artifact_bundle.py` rejects artifact JSON containing CRLF.

Do not disable checksum verification to work around this error. If an artifact
is intentionally regenerated or edited, regenerate its checksum from the exact
canonical file that will be committed and deployed.

## Required Workflow After Adding a New Raw Month

Run commands from the project root.

1. Merge all raw CSV files:

   ```powershell
   .\.venv\Scripts\python.exe -m src.merge_dataset
   ```

2. Restart the kernel and run all cells in
   `notebooks/02_data_cleaning.ipynb`.
3. Restart the kernel and run all cells in
   `notebooks/06_feature_engineering.ipynb`.
4. To refresh Notebook 08, run and save:
   - `notebooks/07_retraining_model.ipynb`
   - `notebooks/08_advance_model.ipynb`
5. To refresh the final app artifact, run:

   ```powershell
   .\.venv\Scripts\python.exe -m src.train_final_model
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

Notebook 07/08 is not required for the final app artifact. Conversely,
`src.train_final_model` does not update the split files consumed by Notebook
08.

The following notebooks can be skipped for the final XGBoost/app refresh:

- `notebooks/01_exploration.ipynb`
- `notebooks/03_preprocessing_baseline.ipynb`
- `notebooks/04_baseline_model.ipynb`
- `notebooks/05_model_comparison.ipynb`

## Model and Inference Decisions

- Historical `CloseDate` is copied to `ValuationDate` during training.
- App/inference requests must supply `ValuationDate`; `CloseDate` is rejected
  to avoid post-outcome leakage.
- Feature engineering is shared through `src/features.py`.
- Native-categorical preprocessing is implemented in
  `src/preprocessing_xgboost.py`.
- The model bundle includes model, preprocessor, input schema, feature schema,
  metrics, manifest, and checksums.
- Unknown inference categories become missing values rather than receiving
  incorrect category codes.
- Boolean amenity engineering maps each amenity column independently with
  `Series.map(BOOLEAN_MAPPING)`. This avoids the pandas DataFrame replacement
  path that previously produced `pop index out of range` for valid app inputs.

## Validated Berkeley Pricing Case

The test address `663 Grizzly Peak Blvd, Berkeley, CA 94708` is useful for
distinguishing a list price from the model target:

- Screenshot input: 3 beds, 2 baths, 1,664 sq ft, built in 1955, valuation date
  2026-08-19.
- Local prediction after the inference fixes: approximately `$1,597,470`.
- Current public asking price observed on 2026-08-19: `$1,095,000`.
- Redfin's contemporaneous automated estimate: `$1,472,822`, with nearby
  comparable sales reported around `$1.2M` to `$1.5M`.
- The merged CRMLS dataset contains this exact property's 2025-02-19 closed
  sale at `$1,350,000` (list price `$1,349,950`, original list price
  `$1,475,000`).

Relevant public references:

- `https://www.redfin.com/CA/Berkeley/663-Grizzly-Peak-Blvd-94708/home/701945`
- `https://www.propertyshark.com/mason/Property/38561822/663-Grizzly-Peak-Blvd-Berkeley-CA-94708/`

Conclusion: the app predicts a likely **closed sale price**, not the current
asking price. A roughly `$1.5M` result for this address is consistent with the
model target and other sale-value signals; it should not be hard-coded or
clamped to the `$1.095M` asking price. If the product later needs asking-price
prediction, that should be a separate target and evaluation workflow.

## Verification History

As of 2026-08-20:

- 8 notebook files parse as valid JSON.
- Python source syntax checks passed.
- 19 automated tests passed.
- Artifact internal checksums passed.
- `load_artifacts()` succeeded with checksum verification enabled after LF
  normalization.
- Artifact source-data hash matched the current engineered dataset.
- A prior Streamlit smoke test returned HTTP 200.
- Notebook code-cell comparison found no accidental cell overwrite. Notebook
  01 changes were execution outputs only; user edits in Notebook 08 were
  preserved.
- Local `main` and `origin/main` were synchronized at commit `07ef08a` when
  this memory was updated.

## Known Caveats and Follow-Ups

- `data/processed/README.md` is stale: it still lists old row counts and refers
  to the deleted `future_outputs` folder. It does not affect execution.
- `.gitignore` ignores the entire `data/` directory, so `data/README.md` and
  `data/processed/README.md` are not tracked by Git.
- Current `git status --short` reports `PROJECT_MEMORY.md` and the root
  `README.md` as untracked. Add them explicitly if they should be preserved in
  the repository.
- Notebook 06 requires GeoPandas. If the active environment lacks it:

  ```powershell
  .\.venv\Scripts\python.exe -m pip install geopandas ipykernel
  ```

- Most notebooks use `../data/...` paths and assume a notebook working
  directory. Use the VS Code/Jupyter notebook kernel and run from the notebook
  context unless paths are refactored.
- `src/merge_dataset.py` does not currently reject duplicate monthly files and
  loads all raw CSV files into memory before concatenation. Check for duplicate
  month files before rerunning it.
- Baseline output remains available for historical comparison but is not part
  of the final app pipeline and may not reflect the newest month unless the
  baseline notebook is rerun.

## User Working Preferences

- Prefer direct, practical instructions and exact file paths.
- Preserve existing user edits and avoid rewriting notebook structure.
- For review-only requests, inspect and report before modifying files.
- Keep the monthly refresh workflow reproducible and clearly separate
  Notebook 08 data generation from final app artifact training.
