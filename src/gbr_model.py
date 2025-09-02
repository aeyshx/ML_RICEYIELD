import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_config, ensure_directories, prepare_datasets, save_model, write_predictions_and_summary, run_synthetic_self_check, write_readme, write_metrics_and_diagnostics


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_directories(cfg['paths']['output_dir'], cfg['paths']['models_dir'])

    print(f"Python: {sys.version.split()[0]}")
    import sklearn, pandas, numpy
    print(f"numpy: {numpy.__version__}")
    print(f"pandas: {pandas.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"random_state: {cfg.get('random_state', 42)}")

    ds, pre, train_df, test_df = prepare_datasets(cfg)

    # Get GBR-specific configuration
    gbr_config = cfg.get('models', {}).get('gbr', {})
    
    # Check if hyperparameter tuning is enabled
    use_tuning = gbr_config.get('use_hyperparameter_tuning', False)
    
    if use_tuning:
        print("Using TimeSeriesSplit with GridSearchCV for hyperparameter tuning...")
        
        # Create base model
        base_model = GradientBoostingRegressor(
            min_samples_split=gbr_config.get('default_min_samples_split', 2),
            min_samples_leaf=gbr_config.get('default_min_samples_leaf', 1),
            random_state=cfg.get('random_state', 42)
        )
        
        # Define parameter grid (avoiding None values that cause validation errors)
        param_grid = {
            'n_estimators': gbr_config.get('n_estimators', [50, 100, 200]),
            'learning_rate': gbr_config.get('learning_rate', [0.05, 0.1, 0.2]),
            'max_depth': gbr_config.get('max_depth', [3, 5, 7]),
            'subsample': gbr_config.get('subsample', [0.8, 1.0]),
            'loss': gbr_config.get('loss', ['squared_error', 'huber'])
        }
        
        # Only add max_features if it's not None
        max_features_values = gbr_config.get('max_features', ['sqrt', 'log2'])
        if max_features_values and not all(v is None for v in max_features_values):
            param_grid['max_features'] = max_features_values
        
        # Create TimeSeriesSplit
        n_splits = gbr_config.get('n_splits', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            error_score='raise'  # Raise errors to debug issues
        )
        
        # Fit the grid search
        print(f"Performing grid search with {n_splits} time-series splits...")
        print(f"Parameter grid: {param_grid}")
        grid_search.fit(ds.X_train, ds.y_train)
        
        # Get best model
        model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score (RMSE): {np.sqrt(-grid_search.best_score_):.4f}")
        
    else:
        print("Using default hyperparameters...")
        
        # Create model with default hyperparameters
        model = GradientBoostingRegressor(
            n_estimators=gbr_config.get('default_n_estimators', 100),
            learning_rate=gbr_config.get('default_learning_rate', 0.1),
            max_depth=gbr_config.get('default_max_depth', 3),
            min_samples_split=gbr_config.get('default_min_samples_split', 2),
            min_samples_leaf=gbr_config.get('default_min_samples_leaf', 1),
            subsample=gbr_config.get('default_subsample', 1.0),
            max_features=gbr_config.get('default_max_features', None),
            loss=gbr_config.get('default_loss', 'squared_error'),
            random_state=cfg.get('random_state', 42)
        )
        
        print(f"Training GBR with config: n_estimators={gbr_config.get('default_n_estimators', 100)}, "
              f"learning_rate={gbr_config.get('default_learning_rate', 0.1)}, "
              f"max_depth={gbr_config.get('default_max_depth', 3)}")
        
        model.fit(ds.X_train, ds.y_train)

    # Make predictions
    preds = model.predict(ds.X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ds.y_test, preds))
    mae = mean_absolute_error(ds.y_test, preds)
    r2 = r2_score(ds.y_test, preds)
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # Check if model should be saved
    should_save = gbr_config.get('save_model', cfg.get('models', {}).get('save_model', True))
    if should_save:
        # Save the best model from grid search if tuning was used
        model_to_save = grid_search if use_tuning else model
        model_path = save_model(model_to_save, cfg['paths']['models_dir'], 'gbr')
        print(f"Saved model to: {model_path}")
    else:
        print("Model saving skipped (configured to not save)")

    # Write predictions and summary
    pred_path, summary_path = write_predictions_and_summary(ds.test_index, preds, cfg['paths']['output_dir'], 'gbr')
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote summary: {summary_path}")

    # Write detailed metrics and diagnostics
    metrics_path = write_metrics_and_diagnostics(ds.y_test, preds, ds.test_index, cfg['paths']['output_dir'], 'gbr')
    print(f"Wrote detailed metrics: {metrics_path}")

    # Run synthetic self-check
    sp_pred, sp_sum = run_synthetic_self_check(pre, cfg, model, 'gbr')
    print(f"Synthetic predictions: {sp_pred}")
    print(f"Synthetic summary: {sp_sum}")

    write_readme(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GradientBoostingRegressor for rice yield prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.config)
