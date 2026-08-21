# IDX Exchange California Home Valuation

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Model%20Development-F37626?logo=jupyter&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-Regression-EB5B24) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white) ![Real Estate ML](https://img.shields.io/badge/Domain-Real%20Estate%20ML-2E6F95)

IDX Exchange California Home Valuation is a Python and Streamlit machine learning project developed during the IDX Exchange Data Science Internship. The project uses historical CRMLS sold-property data to estimate the ***closed sale*** price of California single-family homes through data cleaning, exploratory analysis, feature engineering, model comparison, XGBoost training, and interactive deployment.

The public repository documents the modeling workflow and includes the exported model artifacts required by the application. Raw MLS exports and processed row-level datasets are excluded to protect restricted real estate data.


## Interactive Streamlit Application

| Application | Purpose | Key inputs and output | Open |
| --- | --- | --- | --- |
| California Home Valuation | Generate an indicative closed-sale price for a California single-family home. | Address or coordinates, property characteristics, market-area fields, amenities, estimated price, and median-error reference band. | [Open Streamlit App](https://tan-idx-exchange-ds.streamlit.app/) |


## Data Source & Privacy

The original working data consists of monthly CRMLS sold-property CSV files covering January 2022 through June 2026. The records contain transaction, property, pricing, location, school district, and amenity fields used during cleaning and feature engineering.

Because MLS data may be restricted or contain sensitive information, this repository does **not** include raw CSV exports, cleaned row-level datasets, credentials, or private acquisition utilities. It includes only non-sensitive source code, notebooks, tests, documentation, and the trained model bundle.


## Reproducibility

The deployed application and local inference workflow are reproducible from the included artifacts in `artifacts/xgboost_final/`. Rebuilding the full dataset or retraining the model requires authorized CRMLS sold-property files with a compatible schema.

The project uses chronological data splits rather than a random split. The model trains on earlier sales and is evaluated on later months, which more closely represents real-world valuation use.


## Key Features

- Cleans and combines monthly CRMLS sold-property data into a consistent modeling dataset.
- Explores baseline models, feature engineering choices, and advanced models through Jupyter notebooks.
- Trains a native-categorical `XGBRegressor` for California single-family-home valuation.
- Uses property size, age, layout, geography, school district, and amenity information.
- Resolves California addresses through the U.S. Census Geocoder or accepts manual coordinates.
- Produces an indicative closed-sale price and a reference band based on median test error.
- Validates versioned model, preprocessing, schema, metric, manifest, and checksum artifacts before inference.


## Feature Engineering

The final inference pipeline creates 11 derived numerical features from information available at valuation time. These features help the model describe the relationship between a home's age, usable space, lot size, amenities, and season rather than relying only on raw values.

| Feature group | Engineered features | Purpose |
| --- | --- | --- |
| Property age | `PropertyAge` | Calculates the home's age from `ValuationDate` and `YearBuilt`, so age changes correctly with the date of the estimate. |
| Space and layout ratios | `BathBedRatio`, `LivingAreaPerBedroom`, `LivingAreaPerBathroom`, `LotToLivingRatio` | Describes how the home's interior space, rooms, and lot size relate to one another. |
| Log-transformed size | `LogLivingArea`, `LogLotSize` | Represents highly skewed living-area and lot-size values on a compressed scale. |
| Amenity summary | `AmenityCount`, `AmenityKnownCount` | Summarizes the presence and availability of pool, view, attached garage, new construction, and fireplace information. |
| Seasonal pattern | `ValuationMonthSin`, `ValuationMonthCos` | Encodes the valuation month as a continuous annual cycle without creating an artificial break between December and January. |
| Geographic enrichment | `UnifiedSchoolDistrict` | Uses property coordinates and California school-district boundaries to add local school-market context as a categorical feature. |

The engineered fields are combined with the original property characteristics, latitude and longitude, and native categorical features such as city, postal code, county, MLS area, property level, and amenity flags.

During training, each historical property's `CloseDate` is copied to `ValuationDate` before these features are calculated. At inference time, the user supplies `ValuationDate`, while `CloseDate` is rejected to prevent post-sale information from leaking into the estimate.


## External Geographic Data

### Address Geocoding

Addresses entered in the Streamlit app are resolved with the **[U.S. Census Geocoder](https://geocoding.geo.census.gov/geocoder/)**. The application uses the matched geographic data returned by the service to populate the standardized address, latitude, longitude, city, postal code, county, and Unified School District. The valuation model itself does not generate these location values.

### School-District Boundaries

During feature engineering, each property point is spatially joined to a California school-district polygon using its latitude and longitude. Only unified school districts are retained, and the resulting district name is stored as `UnifiedSchoolDistrict`.

- Boundary dataset: California School District Areas 2025–26
- Dataset page documented in Notebook 06: [California School District Areas 2025–26](https://data.ca.gov/dataset/california-school-district-areas-2025-26)

## Model Performance

| Metric | Current result |
| --- | ---: |
| Training period | January 2024 – April 2026 |
| Training records | 304,822 |
| Validation period | May 2026 |
| Holdout test period | June 2026 |
| Test $R^2$ | 0.904 |
| Test MAE | $155,260.63 |
| Test MAPE | 11.81% |
| Test MdAPE | 8.01% |


## Future Work

- Add a reproducible hyperparameter-tuning stage for the final XGBoost model. Compare Grid Search with **Random Search** under chronological validation, tuning parameters such as tree depth, learning rate, number of estimators, subsampling, column sampling, and regularization.
- Replace the median-error reference band with a calibrated prediction interval that better represents uncertainty for each property.
- Automate monthly model refreshes and monitor performance drift as new CRMLS sales become available.


## Tech Stack

| Area | Tools |
| --- | --- |
| Data processing | Python, Pandas, NumPy |
| Exploratory analysis | Jupyter Notebook |
| Machine learning | scikit-learn, XGBoost |
| Application | Streamlit |


## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Run the automated tests with:

```powershell
python -m unittest discover -s tests -v
```

## AI-Assisted Development

This portfolio project was developed with AI assistance. AI tools were used to support the Streamlit interface, code refactoring, debugging, testing, and documentation. The project scope, data preparation, feature engineering, model selection, evaluation, and final validation were directed and reviewed by me.


## Business Value

Property valuation often starts with repetitive manual work: reviewing comparable sales, checking location details, and comparing the physical characteristics of a home. This project turns that first review into a faster and more consistent workflow by providing a data-backed estimate from the information available at valuation time.

For a real estate team, the application can support early property screening, internal pricing discussions, and client conversations. The estimate and reference band are decision-support signals, not a replacement for local market knowledge, a comparative market analysis, or a licensed appraisal.
