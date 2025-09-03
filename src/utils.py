import os
import sys
import math
import time
import joblib
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Generator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        self.circular_mean_deg_: Optional[float] = None
        self.feature_columns_: List[str] = []
        # Stats computed on training data for fold-safe transforms
        self._quarter_means_: Dict[str, pd.Series] = {}
        self._quarter_stds_: Dict[str, pd.Series] = {}
        self._winsor_bounds_: Dict[str, Tuple[float, float]] = {}

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
        # Compute per-quarter means/stds for anomalies
        q_groups = df_joined.copy()
        q_groups['quarter'] = pd.to_numeric(q_groups['quarter'], errors='coerce').astype('Int64')
        for col in ['rainfall', 'min_temperature', 'max_temperature']:
            series = pd.to_numeric(q_groups[col], errors='coerce')
            stats = series.groupby(q_groups['quarter'])
            self._quarter_means_[col] = stats.mean()
            self._quarter_stds_[col] = stats.std(ddof=0).replace(0, np.nan)
        # Winsorization bounds from training distribution
        for col in ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed']:
            s = pd.to_numeric(df_joined[col], errors='coerce')
            self._winsor_bounds_[col] = (float(s.quantile(0.01)), float(s.quantile(0.99)))
        return self

    def transform(self, df_joined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df_joined.copy()

        # Clean numeric columns
        for col in ['produced_rice', 'area_harvested']:
            df[col] = self._clean_numeric_series(df[col])

        # Target
        y = self._validate_and_compute_target(df)

        # Climate base features with sanitization
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
        df['min_temperature'] = pd.to_numeric(df['min_temperature'], errors='coerce')
        df['max_temperature'] = pd.to_numeric(df['max_temperature'], errors='coerce')
        df['relative_humidity'] = pd.to_numeric(df['relative_humidity'], errors='coerce')
        # RH sentinel and range check
        df['relative_humidity'] = df['relative_humidity'].where((df['relative_humidity'] >= 0) & (df['relative_humidity'] <= 100))
        df.loc[df['relative_humidity'] == -999, 'relative_humidity'] = np.nan
        df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
        # Winsorize extremes using training bounds
        for col in ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed']:
            if col in self._winsor_bounds_:
                low, high = self._winsor_bounds_[col]
                df[col] = df[col].clip(lower=low, upper=high)

        # Wind direction cleaning and circular encoding with imputation
        wind_dir = self._sanitize_wind_direction(df['wind_direction'])
        if self.circular_mean_deg_ is None:
            self.circular_mean_deg_ = 0.0
        wind_dir = wind_dir.fillna(self.circular_mean_deg_)
        radians = np.deg2rad(wind_dir.astype(float))
        df['wind_dir_sin'] = np.sin(radians)
        df['wind_dir_cos'] = np.cos(radians)

        # Typhoon impact: 1 if typhoon appears in that quarter based on rule, else 0
        if self.typhoon_impact_rule == 'STY_only':
            df['typhoon_impact'] = (df['type'].astype(str).str.upper() == 'STY').astype(int)
        elif self.typhoon_impact_rule == 'all_types':
            # Include all typhoon types: TD, TS, STS, TY, STY
            typhoon_types = ['TD', 'TS', 'STS', 'TY', 'STY']
            df['typhoon_impact'] = df['type'].astype(str).str.upper().isin(typhoon_types).astype(int)
        else:
            raise ValueError(f"Unknown typhoon_impact_rule: {self.typhoon_impact_rule}")

        # Quarter features
        if self.use_quarter_dummies:
            quarter_dummies = pd.get_dummies(df['quarter'], prefix='q', dtype=int)
            df = pd.concat([df, quarter_dummies], axis=1)
        # Cyclical quarter encoding
        q_numeric = pd.to_numeric(df['quarter'], errors='coerce').astype(float)
        df['quarter_sin'] = np.sin(2 * np.pi * (q_numeric - 1) / 4.0)
        df['quarter_cos'] = np.cos(2 * np.pi * (q_numeric - 1) / 4.0)

        # Climate anomalies per quarter (z-scores using training means/stds)
        for col in ['rainfall', 'min_temperature', 'max_temperature']:
            means = self._quarter_means_.get(col, pd.Series(dtype=float))
            stds = self._quarter_stds_.get(col, pd.Series(dtype=float))
            df = df.merge(means.rename(f'{col}_q_mean'), left_on='quarter', right_index=True, how='left')
            df = df.merge(stds.rename(f'{col}_q_std'), left_on='quarter', right_index=True, how='left')
            df[f'{col}_anomaly'] = (df[col] - df[f'{col}_q_mean']) / df[f'{col}_q_std']
            df = df.drop(columns=[f'{col}_q_mean', f'{col}_q_std'])

        # Optional lags of climate and cyclical wind features
        if self.use_one_quarter_lag:
            lag_cols = ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'rainfall_anomaly', 'min_temperature_anomaly', 'max_temperature_anomaly', 'quarter_sin', 'quarter_cos', 'quarter_max_msw', 'quarter_event_count']
            # Sort chronologically and shift for lag features
            df = df.sort_values(['year', 'quarter']).reset_index(drop=True)
            for col in lag_cols:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag2'] = df[col].shift(2)

        # Final feature set
        feature_cols = [
            'rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed',
            'wind_dir_sin', 'wind_dir_cos', 'typhoon_impact',
            'quarter_sin', 'quarter_cos',
            'rainfall_anomaly', 'min_temperature_anomaly', 'max_temperature_anomaly',
            'quarter_max_msw', 'quarter_event_count'
        ]
        if self.use_quarter_dummies:
            feature_cols.extend([c for c in df.columns if c.startswith('q_')])
        if self.use_one_quarter_lag:
            feature_cols.extend([f'{c}_lag1' for c in ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'rainfall_anomaly', 'min_temperature_anomaly', 'max_temperature_anomaly', 'quarter_sin', 'quarter_cos', 'quarter_max_msw', 'quarter_event_count']])
            feature_cols.extend([f'{c}_lag2' for c in ['rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'rainfall_anomaly', 'min_temperature_anomaly', 'max_temperature_anomaly', 'quarter_sin', 'quarter_cos', 'quarter_max_msw', 'quarter_event_count']])

        X = df[feature_cols].copy()
        # Simple imputation for any remaining missing values (e.g., first lag)
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

    # Reduce disasters to per-quarter typhoon presence based on rule and derive intensity/count features
    disasters['type'] = disasters['type'].astype(str).str.upper()
    # Ensure msw exists (if missing, fill with 0)
    if 'msw' not in disasters.columns:
        disasters['msw'] = 0
    disasters['event_id'] = 1
    
    if typhoon_impact_rule == 'STY_only':
        typhoon_flags = (
            disasters.assign(typhoon_flag=lambda d: (d['type'] == 'STY').astype(int))
                     .groupby(['year', 'quarter'], as_index=False)['typhoon_flag']
                     .max()
        )
    elif typhoon_impact_rule == 'all_types':
        # Include all typhoon types: TD, TS, STS, TY, STY
        typhoon_types = ['TD', 'TS', 'STS', 'TY', 'STY']
        typhoon_flags = (
            disasters.assign(typhoon_flag=lambda d: d['type'].isin(typhoon_types).astype(int))
                     .groupby(['year', 'quarter'], as_index=False)['typhoon_flag']
                     .max()
        )
    else:
        raise ValueError(f"Unknown typhoon_impact_rule: {typhoon_impact_rule}")

    # Disaster intensity and frequency features
    per_quarter_intensity = disasters.groupby(['year', 'quarter'], as_index=False).agg(
        quarter_max_msw=('msw', 'max'),
        quarter_event_count=('event_id', 'count')
    )

    # Join on year and quarter
    joined = (
        agri.merge(climate, on=['year', 'quarter'], how='inner', suffixes=('', '_clim'))
            .merge(typhoon_flags, on=['year', 'quarter'], how='left')
            .merge(per_quarter_intensity, on=['year', 'quarter'], how='left')
    )

    # Use original disaster type column if present, else construct from flag for downstream processing
    if 'type' not in joined.columns:
        if typhoon_impact_rule == 'STY_only':
            joined['type'] = np.where(joined['typhoon_flag'] == 1, 'STY', 'NA')
        else:  # all_types
            # For all_types, we need to preserve the actual typhoon type
            # This is a simplified approach - in practice, you might want to handle multiple types per quarter
            typhoon_quarters = disasters[disasters['type'].isin(['TD', 'TS', 'STS', 'TY', 'STY'])]
            if not typhoon_quarters.empty:
                # Get the strongest typhoon type per quarter
                type_priority = {'STY': 5, 'TY': 4, 'STS': 3, 'TS': 2, 'TD': 1}
                typhoon_quarters['priority'] = typhoon_quarters['type'].map(type_priority)
                strongest_per_quarter = typhoon_quarters.groupby(['year', 'quarter'])['priority'].idxmax()
                strongest_types = typhoon_quarters.loc[strongest_per_quarter, ['year', 'quarter', 'type']].rename(columns={'type': 'typhoon_type'})
                joined = joined.merge(strongest_types, on=['year', 'quarter'], how='left')
                joined['type'] = joined['typhoon_type'].fillna('NA')
                joined = joined.drop('typhoon_type', axis=1)
            else:
                joined['type'] = 'NA'
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
    # Deprecated: retained for backward-compatibility with older scripts.
    paths = cfg['paths']
    split = cfg.get('split_years', {})
    typhoon_impact_rule = cfg['features'].get('typhoon_impact_rule', 'STY_only')
    data = read_and_join_data(paths['data_dir'], typhoon_impact_rule)

    if not split:
        # Fallback: use full data as both train and test (no real split)
        df_sorted = data.sort_values(['year', 'quarter']).reset_index(drop=True)
        pre = Preprocessor(
            use_quarter_dummies=cfg['features'].get('use_quarter_dummies', True),
            use_one_quarter_lag=cfg['features'].get('use_one_quarter_lag', True),
            typhoon_impact_rule=typhoon_impact_rule
        )
        pre.fit(df_sorted)
        X_all, y_all = pre.transform(df_sorted)
        return DatasetSplit(X_all, y_all, X_all, y_all, df_sorted[['year', 'quarter', 'produced_rice', 'area_harvested']].copy()), pre, df_sorted, df_sorted

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


# --- Time-aware cross-validation utilities ---

def get_cv_config(cfg: Dict) -> Dict:
    cv_cfg = cfg.get('cv', {}) or {}
    return {
        'k_folds': int(cv_cfg.get('k_folds', 5)),
        'strategy': str(cv_cfg.get('strategy', 'expanding')),
        'gap': int(cv_cfg.get('gap', 0)),
        'max_train_size': cv_cfg.get('max_train_size', None),
        'refit_full': bool(cv_cfg.get('refit_full', True)),
    }


def generate_time_series_folds(n_samples: int, k_folds: int = 5, strategy: str = 'expanding', gap: int = 0, max_train_size: Optional[int] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    if strategy not in ('expanding', 'rolling'):
        raise ValueError("strategy must be 'expanding' or 'rolling'")
    effective_max_train = max_train_size if strategy == 'rolling' else None
    tss = TimeSeriesSplit(n_splits=k_folds, gap=gap, max_train_size=effective_max_train)
    indices = np.arange(n_samples)
    for train_idx, val_idx in tss.split(indices):
        yield train_idx, val_idx


def fit_transform_for_fold(pre: Preprocessor, df_sorted: pd.DataFrame, train_idx: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
    # Fit preprocessor on training subset to compute circular mean, etc., then transform full sorted df
    pre.fit(df_sorted.iloc[train_idx].reset_index(drop=True))
    X_all, y_all = pre.transform(df_sorted)
    return X_all, y_all


def write_oof_predictions(oof_rows: pd.DataFrame, output_dir: str, model_name: str) -> str:
    pred_path = os.path.join(output_dir, f"predictions_oof_{model_name}.csv")
    oof_rows.to_csv(pred_path, index=False)
    return pred_path


def write_cv_metrics(per_fold: List[Dict], output_dir: str, model_name: str) -> str:
    metrics_df = pd.DataFrame(per_fold)
    agg = {
        'fold': ['aggregate'],
        'n_points': [int(metrics_df['n_points'].sum()) if 'n_points' in metrics_df else 0],
        'rmse': [float(metrics_df['rmse'].mean())],
        'mae': [float(metrics_df['mae'].mean())],
        'r2': [float(metrics_df['r2'].mean())],
        'std_rmse': [float(metrics_df['rmse'].std(ddof=0))],
        'std_mae': [float(metrics_df['mae'].std(ddof=0))],
        'std_r2': [float(metrics_df['r2'].std(ddof=0))],
    }
    agg_df = pd.DataFrame(agg)
    # Ensure consistent column order
    cols = ['fold', 'n_points', 'rmse', 'mae', 'r2', 'std_rmse', 'std_mae', 'std_r2']
    out_df = pd.concat([metrics_df, agg_df], ignore_index=True)[cols]
    path = os.path.join(output_dir, f"metrics_cv_{model_name}.csv")
    out_df.to_csv(path, index=False)
    return path


def write_cv_summary(all_preds: np.ndarray, output_dir: str, model_name: str) -> str:
    summary = pd.DataFrame({
        'min_pred_yield': [float(np.min(all_preds))],
        'mean_pred_yield': [float(np.mean(all_preds))],
        'max_pred_yield': [float(np.max(all_preds))]
    })
    summary_path = os.path.join(output_dir, f"summary_cv_{model_name}.csv")
    summary.to_csv(summary_path, index=False)
    return summary_path


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
    cv_cfg = get_cv_config(cfg)
    content = f"""
## Rice Yield Prediction (Albay) - Regression Pipelines

This repository trains three standalone regressors to predict quarterly rice_yield (t/ha) for Albay using AGRICULTURE.csv, CLIMATE.csv, and DISASTERS.csv.

### Inputs and locations
- data/AGRICULTURE.csv
- data/CLIMATE.csv
- data/DISASTERS.csv

Join keys: year, quarter (all rows are Albay-wide and temporally aligned).

### Cross-validation
- Time-aware K-fold CV with scikit-learn TimeSeriesSplit
- Folds: {cv_cfg['k_folds']} | Strategy: {cv_cfg['strategy']} | Gap: {cv_cfg['gap']}
- OOF predictions are concatenated validation predictions across folds

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
- OOF predictions: {paths['output_dir']}/predictions_oof_[model].csv with [year, quarter, produced_rice, area_harvested, rice_yield_pred, fold]
- CV metrics: {paths['output_dir']}/metrics_cv_[model].csv with per-fold rows and aggregate stats
- CV summary: {paths['output_dir']}/summary_cv_[model].csv with [min_pred_yield, mean_pred_yield, max_pred_yield]
- Saved models: {paths['models_dir']}/*.joblib
- Synthetic self-check outputs: {paths['output_dir']}/sample/sample_predictions_[model].csv and sample_summary_[model].csv

### Configuration
- See config.yaml. Environment overrides via .env (.env.example provided) for DATA_DIR, OUTPUT_DIR, MODELS_DIR.
- Control model saving and sample creation via `output.save_model` and `output.create_sample` settings.
- CV block: k_folds, strategy (expanding|rolling), gap, max_train_size, refit_full

### Notes
- Typhoon impact configurable via typhoon_impact_rule (STY_only or all_types).
- Wind direction outside [0,360) treated as missing, imputed by circular mean before sin/cos.
""".strip() + "\n"
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
