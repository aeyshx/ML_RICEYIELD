import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils import load_config, ensure_directories, prepare_datasets, save_model, write_predictions_and_summary, run_synthetic_self_check, write_readme


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
    
    # Create model with configured hyperparameters
    model = GradientBoostingRegressor(
        n_estimators=gbr_config.get('n_estimators', 100),
        learning_rate=gbr_config.get('learning_rate', 0.1),
        max_depth=gbr_config.get('max_depth', 3),
        min_samples_split=gbr_config.get('min_samples_split', 2),
        min_samples_leaf=gbr_config.get('min_samples_leaf', 1),
        subsample=gbr_config.get('subsample', 1.0),
        max_features=gbr_config.get('max_features', None),
        loss=gbr_config.get('loss', 'squared_error'),
        random_state=cfg.get('random_state', 42)
    )
    
    print(f"Training GBR with config: n_estimators={gbr_config.get('n_estimators', 100)}, "
          f"learning_rate={gbr_config.get('learning_rate', 0.1)}, "
          f"max_depth={gbr_config.get('max_depth', 3)}")
    
    model.fit(ds.X_train, ds.y_train)

    preds = model.predict(ds.X_test)

    rmse = np.sqrt(mean_squared_error(ds.y_test, preds))
    mae = mean_absolute_error(ds.y_test, preds)
    r2 = r2_score(ds.y_test, preds)
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # Check if model should be saved
    should_save = gbr_config.get('save_model', cfg.get('models', {}).get('save_model', True))
    if should_save:
        model_path = save_model(model, cfg['paths']['models_dir'], 'gbr')
        print(f"Saved model to: {model_path}")
    else:
        print("Model saving skipped (configured to not save)")

    pred_path, summary_path = write_predictions_and_summary(ds.test_index, preds, cfg['paths']['output_dir'], 'gbr')
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote summary: {summary_path}")

    sp_pred, sp_sum = run_synthetic_self_check(pre, cfg, model, 'gbr')
    print(f"Synthetic predictions: {sp_pred}")
    print(f"Synthetic summary: {sp_sum}")

    write_readme(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GradientBoostingRegressor for rice yield prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.config)
