# IDX Exchange Residential Valuation

## Run the app

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The checked-in artifact bundle is stored in `artifacts/xgboost_final/`.
Inference requires `ValuationDate`; `CloseDate` is not accepted by the app or
the inference API.

The app supports either a California street address or manually entered
coordinates. Address lookup uses the public U.S. Census Geocoder, so deployment
does not require a Google Maps API key or a Streamlit secret. City, ZIP code,
and county are matched to known model categories when available. School
district remains editable because Census school-district geography does not
always align with the listing field used to train the model.

## Rebuild the artifact

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -m src.train_final_model
```

The exporter reproduces the final notebook split, target filtering, feature
order, native categorical handling, and XGBoost parameters.

## Processed data layout

The canonical cleaned dataset stays at
`data/processed/crmls_sfr_cleaned.csv`. Model-specific derivatives are kept
separate:

- `data/processed/baseline/` contains the baseline modeling dataset.
- `data/processed/xgboost/` contains the full engineered XGBoost dataset and
  its chronological splits.
- `data/processed/future_outputs/` is a staging area for new experimental
  outputs that have not yet been promoted into a model-specific pipeline.

See `data/processed/README.md` for the complete data catalog and lineage.
