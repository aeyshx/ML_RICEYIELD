## Rice Yield Prediction (Albay) - Regression Pipelines

This repository trains three standalone regressors to predict quarterly rice_yield (t/ha) for Albay using AGRICULTURE.csv, CLIMATE.csv, and DISASTERS.csv.

### Inputs and locations
- data/AGRICULTURE.csv
- data/CLIMATE.csv
- data/DISASTERS.csv

Join keys: year, quarter (all rows are Albay-wide and temporally aligned).

### Training/Test split
- Train: 2000–2018
- Test: 2006–2018

### Features
- rainfall, min_temperature, max_temperature, relative_humidity, wind_speed
- wind_dir_sin, wind_dir_cos (from cleaned circular wind_direction)
- quarter dummies (configurable)
- typhoon_impact (1 if typhoon present based on typhoon_impact_rule)
- optional one-quarter lag for climate features

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
- Saved models: models/*.joblib
- Synthetic self-check outputs: outputs/sample/sample_predictions_[model].csv and sample_summary_[model].csv

### Configuration
- See config.yaml. Environment overrides via .env (.env.example provided) for DATA_DIR, OUTPUT_DIR, MODELS_DIR.
- Control model saving and sample creation via `output.save_model` and `output.create_sample` settings.

### Notes
- Typhoon impact configurable via typhoon_impact_rule (STY_only or all_types).
- Wind direction outside [0,360) treated as missing, imputed by circular mean before sin/cos.
