import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_config, ensure_directories, read_and_join_data, Preprocessor, save_model, run_synthetic_self_check, write_readme, get_cv_config, generate_time_series_folds, fit_transform_for_fold, write_oof_predictions, write_cv_metrics, write_cv_summary


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_directories(cfg['paths']['output_dir'], cfg['paths']['models_dir'])

    print(f"Python: {sys.version.split()[0]}")
    import sklearn, pandas, numpy
    print(f"numpy: {numpy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"random_state: {cfg.get('random_state', 42)}")

    # Data and preprocessing
    data = read_and_join_data(cfg['paths']['data_dir'], cfg['features'].get('typhoon_impact_rule', 'STY_only'))
    data = data.sort_values(['year', 'quarter']).reset_index(drop=True)
    pre = Preprocessor(
        use_quarter_dummies=cfg['features'].get('use_quarter_dummies', True),
        use_one_quarter_lag=cfg['features'].get('use_one_quarter_lag', True),
        typhoon_impact_rule=cfg['features'].get('typhoon_impact_rule', 'STY_only')
    )

    # CV config
    cv_cfg = get_cv_config(cfg)
    k = cv_cfg['k_folds']
    gap = cv_cfg['gap']
    strategy = cv_cfg['strategy']
    max_train_size = cv_cfg['max_train_size']

    X_all, y_all = None, None
    oof_records = []
    per_fold_metrics = []
    fold_idx = 0
    for train_idx, val_idx in generate_time_series_folds(len(data), k_folds=k, strategy=strategy, gap=gap, max_train_size=max_train_size):
        X_all, y_all = fit_transform_for_fold(pre, data, train_idx)
        X_train, y_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        model = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('reg', Ridge(alpha=1.0, random_state=cfg.get('random_state', 42)))
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae = float(mean_absolute_error(y_val, preds))
        r2 = float(r2_score(y_val, preds))
        per_fold_metrics.append({'fold': fold_idx, 'n_points': int(len(val_idx)), 'rmse': rmse, 'mae': mae, 'r2': r2})

        idx_meta = data.iloc[val_idx][['year', 'quarter', 'produced_rice', 'area_harvested']].reset_index(drop=True)
        idx_meta['rice_yield_pred'] = preds
        idx_meta['fold'] = fold_idx
        oof_records.append(idx_meta)
        fold_idx += 1

    oof_df = pd.concat(oof_records, ignore_index=True)
    pred_path = write_oof_predictions(oof_df, cfg['paths']['output_dir'], 'mlr')
    metrics_path = write_cv_metrics(per_fold_metrics, cfg['paths']['output_dir'], 'mlr')
    summary_path = write_cv_summary(oof_df['rice_yield_pred'].to_numpy(), cfg['paths']['output_dir'], 'mlr')
    print(f"Wrote OOF predictions: {pred_path}")
    print(f"Wrote CV metrics: {metrics_path}")
    print(f"Wrote CV summary: {summary_path}")

    # Optional refit on full data and save
    if cv_cfg['refit_full'] and cfg.get('output', {}).get('save_model', True):
        # Fit preprocessor on all data and refit model
        pre.fit(data)
        X_all, y_all = pre.transform(data)
        final_model = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('reg', Ridge(alpha=1.0, random_state=cfg.get('random_state', 42)))
        ])
        final_model.fit(X_all, y_all)
        model_path = save_model(final_model, cfg['paths']['models_dir'], 'mlr')
        print(f"Saved model to: {model_path}")
        # Synthetic self-check
        if cfg.get('output', {}).get('create_sample', True):
            sp_pred, sp_sum = run_synthetic_self_check(pre, cfg, final_model, 'mlr')
            print(f"Synthetic predictions: {sp_pred}")
            print(f"Synthetic summary: {sp_sum}")
    else:
        print("Model saving disabled by configuration or refit_full is False")

    write_readme(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multiple Linear Regression for rice yield prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.config)
