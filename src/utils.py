import os
import sys
import math
import time
import joblib
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from dotenv import load_dotenv


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    load_dotenv(override=True)
    # Resolve paths with environment overrides if provided
    cfg_paths = cfg.get('paths', {})
    cfg_paths['data_dir'] = os.getenv('DATA_DIR', cfg_paths.get('data_dir', 'data'))
    cfg_paths['output_dir'] = os.getenv('OUTPUT_DIR', cfg_paths.get('output_dir', 'outputs'))
    cfg_paths['models_dir'] = os.getenv('MODELS_DIR', cfg_paths.get('models_dir', 'models'))
    cfg['paths'] = cfg_paths
    return cfg


def ensure_directories(output_dir: str, models_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sample'), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_index: pd.DataFrame  # year, quarter, produced_rice, area_harvested for outputs


class Preprocessor:
    def __init__(self, use_quarter_dummies: bool = True, use_one_quarter_lag: bool = True, typhoon_impact_rule: str = 'STY_only'):
        self.use_quarter_dummies = use_quarter_dummies
        self.use_one_quarter_lag = use_one_quarter_lag
        self.typhoon_impact_rule = typhoon_impact_rule
        self.feature_columns_: List[str] = []

    @staticmethod
    def _clean_numeric_series(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce')

    @staticmethod
    def _validate_and_compute_target(df: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        produced = df['produced_rice']
        area = df['area_harvested']
        computed = produced / area
        target = df.get('rice_yield')
        if target is None or target.isna().all():
            return computed
        # Where close, use computed to ensure consistency
        mask_close = (target - computed).abs() <= tol
        result = target.copy()
        result[mask_close] = computed[mask_close]
        # Fill remaining missing with computed
        result = result.fillna(computed)
        return result

    def fit(self, df_joined: pd.DataFrame) -> 'Preprocessor':
        # No fitting needed for the simplified feature set
        # Storm features are computed in read_and_join_data
        # Climate features are handled directly in transform
        return self

    def transform(self, df_joined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df_joined.copy()

        # Clean numeric columns
        for col in ['produced_rice', 'area_harvested']:
            df[col] = self._clean_numeric_series(df[col])

        # Target
        y = self._validate_and_compute_target(df)

        # Climate base features
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
        df['min_temperature'] = pd.to_numeric(df['min_temperature'], errors='coerce')
        df['max_temperature'] = pd.to_numeric(df['max_temperature'], errors='coerce')
        df['relative_humidity'] = pd.to_numeric(df['relative_humidity'], errors='coerce')

        # Storm features are already computed in read_and_join_data
        # typhoon_impact: 1 if any storm event exists in that quarter, else 0
        # msw_max_per_quarter: max msw for that quarter, 0 if no events

        # Final feature set - exactly as specified
        feature_cols = [
            'rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 
            'typhoon_impact', 'msw_max_per_quarter'
        ]

        X = df[feature_cols].copy()
        # Simple imputation for any remaining missing values
        X = X.fillna(X.mean(numeric_only=True))

        self.feature_columns_ = list(X.columns)
        return X, y


def read_and_join_data(data_dir: str, typhoon_impact_rule: str = 'STY_only') -> pd.DataFrame:
    agri_path = os.path.join(data_dir, 'AGRICULTURE.csv')
    climate_path = os.path.join(data_dir, 'CLIMATE.csv')
    disasters_path = os.path.join(data_dir, 'DISASTERS.csv')

    for p in [agri_path, climate_path, disasters_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}")

    agri = pd.read_csv(agri_path)
    climate = pd.read_csv(climate_path)
    disasters = pd.read_csv(disasters_path)

    # Standardize key columns
    for df in [agri, climate, disasters]:
        if 'year' not in df.columns or 'quarter' not in df.columns:
            raise ValueError('All input files must contain year and quarter columns')

    # Process disasters to create quarter-level storm features
    disasters['type'] = disasters['type'].astype(str).str.upper()
    
    if typhoon_impact_rule == 'any_event':
        # Group by [year, quarter] across all storm types (TD/TS/STS/TY/STY)
        # Create msw_max_per_quarter = max(msw) for that quarter
        # Create typhoon_impact = 1 if any event exists in that quarter, else 0
        storm_features = (
            disasters.groupby(['year', 'quarter'], as_index=False)
            .agg({
                'msw': 'max',  # max msw per quarter
                'type': lambda x: 1 if len(x) > 0 else 0  # any event flag
            })
            .rename(columns={
                'msw': 'msw_max_per_quarter',
                'type': 'typhoon_impact'
            })
        )
    elif typhoon_impact_rule == 'STY_only':
        # Legacy behavior for STY_only
        storm_features = (
            disasters.assign(typhoon_flag=lambda d: (d['type'] == 'STY').astype(int))
                     .groupby(['year', 'quarter'], as_index=False)
                     .agg({
                         'typhoon_flag': 'max',
                         'msw': 'max'
                     })
                     .rename(columns={
                         'typhoon_flag': 'typhoon_impact',
                         'msw': 'msw_max_per_quarter'
                     })
        )
    else:
        raise ValueError(f"Unknown typhoon_impact_rule: {typhoon_impact_rule}")

    # Join on year and quarter
    joined = (
        agri.merge(climate, on=['year', 'quarter'], how='inner', suffixes=('', '_clim'))
            .merge(storm_features, on=['year', 'quarter'], how='left')
    )

    # Fill missing msw_max_per_quarter with 0 for quarters without any events
    joined['msw_max_per_quarter'] = joined['msw_max_per_quarter'].fillna(0)
    joined['typhoon_impact'] = joined['typhoon_impact'].fillna(0)

    # Ensure quarter is integer 1-4
    joined['quarter'] = pd.to_numeric(joined['quarter'], errors='coerce').astype('Int64')

    return joined


def chronological_split(df: pd.DataFrame, train_start: int, train_end: int, test_start: int, test_end: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(['year', 'quarter']).reset_index(drop=True)
    train = df_sorted[(df_sorted['year'] >= train_start) & (df_sorted['year'] <= train_end)]
    test = df_sorted[(df_sorted['year'] >= test_start) & (df_sorted['year'] <= test_end)]
    return train, test


def prepare_datasets(cfg: Dict) -> Tuple[DatasetSplit, Preprocessor, pd.DataFrame, pd.DataFrame]:
    paths = cfg['paths']
    split = cfg['split_years']
    typhoon_impact_rule = cfg['features'].get('typhoon_impact_rule', 'STY_only')
    data = read_and_join_data(paths['data_dir'], typhoon_impact_rule)

    train_df, test_df = chronological_split(data, split['train_start'], split['train_end'], split['test_start'], split['test_end'])

    pre = Preprocessor(
        use_quarter_dummies=cfg['features'].get('use_quarter_dummies', True),
        use_one_quarter_lag=cfg['features'].get('use_one_quarter_lag', True),
        typhoon_impact_rule=typhoon_impact_rule
    )
    pre.fit(train_df)

    X_train, y_train = pre.transform(train_df)
    X_test, y_test = pre.transform(test_df)

    test_index = test_df[['year', 'quarter', 'produced_rice', 'area_harvested']].copy()

    return DatasetSplit(X_train, y_train, X_test, y_test, test_index), pre, train_df, test_df


def save_model(model, models_dir: str, model_name: str) -> str:
    ts = time.strftime('%Y%m%d_%H%M%S')
    path = os.path.join(models_dir, f"{model_name}_{ts}.joblib")
    joblib.dump(model, path)
    return path


def write_predictions_and_summary(test_index: pd.DataFrame, preds: np.ndarray, output_dir: str, model_name: str) -> Tuple[str, str]:
    out_pred = test_index.copy()
    out_pred['rice_yield'] = preds
    pred_path = os.path.join(output_dir, f"predictions_{model_name}.csv")
    out_pred.to_csv(pred_path, index=False)

    summary = pd.DataFrame({
        'min_pred_yield': [float(np.min(preds))],
        'mean_pred_yield': [float(np.mean(preds))],
        'max_pred_yield': [float(np.max(preds))]
    })
    summary_path = os.path.join(output_dir, f"summary_{model_name}.csv")
    summary.to_csv(summary_path, index=False)
    return pred_path, summary_path


def build_synthetic_dataframe() -> pd.DataFrame:
    # Minimal synthetic with edge cases: invalid wind_direction values and an STY quarter
    data = pd.DataFrame({
        'year': [2024, 2024, 2025, 2025],
        'quarter': [1, 2, 3, 4],
        'produced_rice': [10000, 12000, 8000, 9000],  # metric tons
        'area_harvested': [2000, 2400, 1600, 1800],   # hectares
        'rice_yield': [5.0, 5.0, 5.0, 5.0],           # t/ha (will be validated)
        'rainfall': [300, 500, 200, 100],
        'min_temperature': [22, 23, 21, 20],
        'max_temperature': [30, 32, 29, 28],
        'relative_humidity': [80, 85, 78, 75],
        'typhoon_impact': [0, 1, 0, 0],
        'msw_max_per_quarter': [0, 200, 0, 0]
    })
    return data


def run_synthetic_self_check(pre: Preprocessor, cfg: Dict, model, model_name: str) -> Tuple[str, str]:
    sample_dir = os.path.join(cfg['paths']['output_dir'], 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    df = build_synthetic_dataframe()
    X, _ = pre.transform(df)
    preds = model.predict(X)

    out_pred = df[['year', 'quarter', 'produced_rice', 'area_harvested']].copy()
    out_pred['rice_yield'] = preds
    pred_path = os.path.join(sample_dir, f"sample_predictions_{model_name}.csv")
    out_pred.to_csv(pred_path, index=False)

    summary = pd.DataFrame({
        'min_pred_yield': [float(np.min(preds))],
        'mean_pred_yield': [float(np.mean(preds))],
        'max_pred_yield': [float(np.max(preds))]
    })
    summary_path = os.path.join(sample_dir, f"sample_summary_{model_name}.csv")
    summary.to_csv(summary_path, index=False)
    return pred_path, summary_path


def write_readme(cfg: Dict) -> None:
    paths = cfg['paths']
    split = cfg['split_years']
    content = f"""
## Rice Yield Prediction (Albay) - Regression Pipelines

This repository trains three standalone regressors to predict quarterly rice_yield (t/ha) for Albay using AGRICULTURE.csv, CLIMATE.csv, and DISASTERS.csv.

### Inputs and locations
- data/AGRICULTURE.csv
- data/CLIMATE.csv
- data/DISASTERS.csv

Join keys: year, quarter (all rows are Albay-wide and temporally aligned).

### Training/Test split
- Train: {split['train_start']}–{split['train_end']}
- Test: {split['test_start']}–{split['test_end']}

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
- Predictions (test rows only): {paths['output_dir']}/predictions_[model].csv with columns [year, quarter, produced_rice, area_harvested, rice_yield]
- Per-model summary CSV: {paths['output_dir']}/summary_[model].csv with [min_pred_yield, mean_pred_yield, max_pred_yield]
- Saved models: {paths['models_dir']}/*.joblib
- Synthetic self-check outputs: {paths['output_dir']}/sample/sample_predictions_[model].csv and sample_summary_[model].csv

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
""".strip() + "\n"
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
