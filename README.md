## Rice Yield Prediction (Albay) - Regression Pipelines

This repository trains three standalone regressors to predict quarterly rice_yield (t/ha) for Albay using AGRICULTURE.csv, CLIMATE.csv, and DISASTERS.csv.

### Inputs and locations
- data/AGRICULTURE.csv
- data/CLIMATE.csv
- data/DISASTERS.csv

Join keys: year, quarter (all rows are Albay-wide and temporally aligned).

### Training/Test split
- Train: 2000–2018
- Test: 2019–2023

### Enhanced Features
- rainfall, min_temperature, max_temperature, relative_humidity, wind_speed
- wind_dir_sin, wind_dir_cos (from cleaned circular wind_direction)
- temp_range, temp_mean (enhanced temperature features)
- quarter dummies (configurable)
- typhoon_impact (basic STY flag)
- Enhanced typhoon features: sty_count, storm_count, msw_max
- Multi-quarter lags (1, 2, 3 quarters) for climate features
- Rolling aggregates (2Q and 3Q means) for rainfall, humidity, temperature
- All features aligned to avoid temporal leakage

Target: rice_yield = produced_rice / area_harvested (t/ha), validated against provided column when close.

### How to run
- Python 3.13.5 recommended with modern NumPy/Pandas/scikit-learn pins.
- Install dependencies from requirements.txt.
- Train/evaluate models:
  - `python src/mlr_model.py --config config.yaml`
  - `python src/rf_model.py --config config.yaml`
  - `python src/gbr_model.py --config config.yaml`

### Outputs
- Predictions (test rows only): outputs/predictions_[model].csv with columns [year, quarter, produced_rice, area_harvested, rice_yield]
- Per-model summary CSV: outputs/summary_[model].csv with [min_pred_yield, mean_pred_yield, max_pred_yield]
- Detailed metrics: outputs/metrics_[model].csv with per-quarter and per-typhoon diagnostics
- Saved models: models/*.joblib
- Synthetic self-check outputs: outputs/sample/sample_predictions_[model].csv and sample_summary_[model].csv

### Configuration
- See config.yaml. Environment overrides via .env (.env.example provided) for DATA_DIR, OUTPUT_DIR, MODELS_DIR.

### Notes
- STY-only typhoon_impact enforced.
- Wind direction outside [0,360) treated as missing, imputed by circular mean before sin/cos.
- Enhanced feature engineering with multi-quarter lags and rolling aggregates.
- Time-series cross-validation and hyperparameter tuning implemented.
