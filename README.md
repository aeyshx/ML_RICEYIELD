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

### Features
- rainfall, min_temperature, max_temperature, relative_humidity
- typhoon_impact (1 if any storm event exists in that quarter, else 0)
- msw_max_per_quarter (max msw for that quarter, 0 if no events, computed from all storm types TD/TS/STS/TY/STY)

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
- Typhoon impact configurable via typhoon_impact_rule (any_event uses all storm types TD/TS/STS/TY/STY).
- msw_max_per_quarter is computed from all storm types, unit km/h, zero when no event.
- Wind features, quarter dummies, and lag features are disabled for this iteration.

### Storm features
- **typhoon_impact**: Binary flag (1 if any storm event present in that quarter, 0 otherwise)
- **msw_max_per_quarter**: Maximum sustained wind speed (km/h) for that quarter, 0 when no events occur
