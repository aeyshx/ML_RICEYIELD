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
    def __init__(self, use_quarter_dummies: bool = True, use_one_quarter_lag: bool = True):
        self.use_quarter_dummies = use_quarter_dummies
        self.use_one_quarter_lag = use_one_quarter_lag
        self.circular_mean_deg_: Optional[float] = None
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

    @staticmethod
    def _circular_mean_degrees(valid_angles_deg: np.ndarray) -> float:
        # valid_angles_deg must already be in [0, 360)
        if valid_angles_deg.size == 0:
            # default to 0 degrees if no valid data; sin=0, cos=1
            return 0.0
        radians = np.deg2rad(valid_angles_deg)
        sin_sum = np.sin(radians).sum()
        cos_sum = np.cos(radians).sum()
        mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))
        if mean_angle < 0:
            mean_angle += 360.0
        return mean_angle

    @staticmethod
    def _sanitize_wind_direction(series: pd.Series) -> pd.Series:
        # Treat outside [0,360) as missing
        s = pd.to_numeric(series, errors='coerce')
        s = s.where((s >= 0) & (s < 360))
        return s

    def fit(self, df_joined: pd.DataFrame) -> 'Preprocessor':
        # Compute circular mean on training set's valid wind directions
        wind_dir = self._sanitize_wind_direction(df_joined['wind_direction'])
        valid_angles = wind_dir.dropna().to_numpy(dtype=float)
        self.circular_mean_deg_ = self._circular_mean_degrees(valid_angles)
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
        df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')

        # Wind direction cleaning and circular encoding with imputation
        wind_dir = self._sanitize_wind_direction(df['wind_direction'])
        if self.circular_mean_deg_ is None:
            self.circular_mean_deg_ = 0.0
        wind_dir = wind_dir.fillna(self.circular_mean_deg_)
        radians = np.deg2rad(wind_dir.astype(float))
        df['wind_dir_sin'] = np.sin(radians)
        df['wind_dir_cos'] = np.cos(radians)

        # Typhoon impact: 1 if STY appears in that quarter, else 0
        df['typhoon_impact'] = (df['type'].astype(str).str.upper() == 'STY').astype(int)

        # Quarter features
        if self.use_quarter_dummies:
            quarter_dummies = pd.get_dummies(df['quarter'], prefix='q', dtype=int)
            df = pd.concat([df, quarter_dummies], axis=1)

        # Optional one-quarter lag of climate features to reflect harvest timing
        if self.use_one_quarter_lag:
            lag_cols = ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos']
            # Sort chronologically and shift for lag features
            df = df.sort_values(['year', 'quarter']).reset_index(drop=True)
            for col in lag_cols:
                df[f'{col}_lag1'] = df[col].shift(1)

        # Final feature set
        feature_cols = [
            'rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed',
            'wind_dir_sin', 'wind_dir_cos', 'typhoon_impact'
        ]
        if self.use_quarter_dummies:
            feature_cols.extend([c for c in df.columns if c.startswith('q_')])
        if self.use_one_quarter_lag:
            feature_cols.extend([f'{c}_lag1' for c in ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos']])

        X = df[feature_cols].copy()
        # Simple imputation for any remaining missing values (e.g., first lag)
        X = X.fillna(X.mean(numeric_only=True))

        self.feature_columns_ = list(X.columns)
        return X, y


def read_and_join_data(data_dir: str) -> pd.DataFrame:
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

    # Reduce disasters to per-quarter STY presence
    disasters['type'] = disasters['type'].astype(str).str.upper()
    sty_flags = (
        disasters.assign(sty_flag=lambda d: (d['type'] == 'STY').astype(int))
                 .groupby(['year', 'quarter'], as_index=False)['sty_flag']
                 .max()
    )

    # Join on year and quarter
    joined = (
        agri.merge(climate, on=['year', 'quarter'], how='inner', suffixes=('', '_clim'))
            .merge(sty_flags, on=['year', 'quarter'], how='left')
    )

    # Use original disaster type column if present, else construct from flag for downstream processing
    if 'type' not in joined.columns:
        joined['type'] = np.where(joined['sty_flag'] == 1, 'STY', 'NA')
    joined['type'] = joined['type'].fillna('NA')

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
    data = read_and_join_data(paths['data_dir'])

    train_df, test_df = chronological_split(data, split['train_start'], split['train_end'], split['test_start'], split['test_end'])

    pre = Preprocessor(
        use_quarter_dummies=cfg['features'].get('use_quarter_dummies', True),
        use_one_quarter_lag=cfg['features'].get('use_one_quarter_lag', True)
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
        'wind_speed': [3.5, 4.0, 2.5, 3.0],
        'wind_direction': [360, -273, 90, 270],  # invalid then valid
        'type': ['NA', 'STY', 'NA', 'NA'],
        'msw': [0, 200, 0, 0]
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
- rainfall, min_temperature, max_temperature, relative_humidity, wind_speed
- wind_dir_sin, wind_dir_cos (from cleaned circular wind_direction)
- quarter dummies (configurable)
- typhoon_impact (1 only if STY present)
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
- Predictions (test rows only): {paths['output_dir']}/predictions_[model].csv with columns [year, quarter, produced_rice, area_harvested, rice_yield]
- Per-model summary CSV: {paths['output_dir']}/summary_[model].csv with [min_pred_yield, mean_pred_yield, max_pred_yield]
- Saved models: {paths['models_dir']}/*.joblib
- Synthetic self-check outputs: {paths['output_dir']}/sample/sample_predictions_[model].csv and sample_summary_[model].csv

### Configuration
- See config.yaml. Environment overrides via .env (.env.example provided) for DATA_DIR, OUTPUT_DIR, MODELS_DIR.

### Notes
- STY-only typhoon_impact enforced.
- Wind direction outside [0,360) treated as missing, imputed by circular mean before sin/cos.
""".strip() + "\n"
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
