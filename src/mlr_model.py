import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

    # Get MLR-specific configuration
    mlr_config = cfg.get('models', {}).get('mlr', {})
    
    # Check if regularization is enabled
    use_regularization = mlr_config.get('use_regularization', False)
    
    if use_regularization:
        print("Using regularized linear regression with StandardScaler...")
        
        # Get regularization parameters
        reg_type = mlr_config.get('regularization_type', 'ridge')
        alpha = mlr_config.get('alpha', 1.0)
        l1_ratio = mlr_config.get('l1_ratio', 0.5)
        
        # Create the appropriate regularized model
        if reg_type == 'ridge':
            regressor = Ridge(alpha=alpha, random_state=cfg.get('random_state', 42))
        elif reg_type == 'lasso':
            regressor = Lasso(alpha=alpha, random_state=cfg.get('random_state', 42))
        elif reg_type == 'elasticnet':
            regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=cfg.get('random_state', 42))
        else:
            print(f"Unknown regularization type: {reg_type}. Using Ridge.")
            regressor = Ridge(alpha=alpha, random_state=cfg.get('random_state', 42))
        
        # Create pipeline with StandardScaler
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])
        
        print(f"Training {reg_type.upper()} regression with alpha={alpha}")
        if reg_type == 'elasticnet':
            print(f"L1 ratio: {l1_ratio}")
        
    else:
        print("Using standard LinearRegression...")
        
        # Create standard linear regression model
        model = LinearRegression()
        
        print("Training Multiple Linear Regression")

    # Fit the model
    model.fit(ds.X_train, ds.y_train)

    # Make predictions
    preds = model.predict(ds.X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(ds.y_test, preds))
    mae = mean_absolute_error(ds.y_test, preds)
    r2 = r2_score(ds.y_test, preds)
    print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    # Check if model should be saved
    should_save = mlr_config.get('save_model', cfg.get('models', {}).get('save_model', True))
    if should_save:
        model_path = save_model(model, cfg['paths']['models_dir'], 'mlr')
        print(f"Saved model to: {model_path}")
    else:
        print("Model saving skipped (configured to not save)")

    # Write predictions and summary
    pred_path, summary_path = write_predictions_and_summary(ds.test_index, preds, cfg['paths']['output_dir'], 'mlr')
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote summary: {summary_path}")

    # Write detailed metrics and diagnostics
    metrics_path = write_metrics_and_diagnostics(ds.y_test, preds, ds.test_index, cfg['paths']['output_dir'], 'mlr')
    print(f"Wrote detailed metrics: {metrics_path}")

    # Run synthetic self-check
    sp_pred, sp_sum = run_synthetic_self_check(pre, cfg, model, 'mlr')
    print(f"Synthetic predictions: {sp_pred}")
    print(f"Synthetic summary: {sp_sum}")

    write_readme(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Multiple Linear Regression for rice yield prediction')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()
    main(args.config)
