#!/usr/bin/env python3
"""
Rice Yield Prediction Visualization and Diagnostics

This script creates comprehensive visualizations for analyzing model performance,
residuals, feature importance, and temporal patterns in rice yield predictions.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import load_config, prepare_datasets

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_predictions(output_dir: str, model_name: str) -> pd.DataFrame:
    """Load predictions for a specific model"""
    pred_path = os.path.join(output_dir, f"predictions_{model_name}.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    return pd.read_csv(pred_path)


def load_metrics(output_dir: str, model_name: str) -> pd.DataFrame:
    """Load metrics for a specific model"""
    metrics_path = os.path.join(output_dir, f"metrics_{model_name}.csv")
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return None
    return pd.read_csv(metrics_path)


def plot_yield_parity(y_true: pd.Series, y_pred: np.ndarray, model_name: str, output_dir: str):
    """Plot actual vs predicted yield parity plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Create scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Add metrics text
    ax.text(0.05, 0.95, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Actual Yield (t/ha)')
    ax.set_ylabel('Predicted Yield (t/ha)')
    ax.set_title(f'{model_name.upper()} - Actual vs Predicted Yield')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = os.path.join(output_dir, f"viz_yield_parity_{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved parity plot: {plot_path}")


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, test_index: pd.DataFrame, 
                  model_name: str, output_dir: str):
    """Plot residuals analysis"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Residuals vs predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Yield')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[0, 1].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals by quarter
    quarter_residuals = []
    quarter_labels = []
    for quarter in [1, 2, 3, 4]:
        mask = test_index['quarter'] == quarter
        if mask.sum() > 0:
            quarter_residuals.append(residuals[mask])
            quarter_labels.append(f'Q{quarter}')
    
    if quarter_residuals:
        axes[1, 0].boxplot(quarter_residuals, labels=quarter_labels)
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals by Quarter')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals by typhoon impact
    if 'typhoon_impact' in test_index.columns:
        typhoon_residuals = []
        typhoon_labels = []
        for typhoon_flag in [0, 1]:
            mask = test_index['typhoon_impact'] == typhoon_flag
            if mask.sum() > 0:
                typhoon_residuals.append(residuals[mask])
                typhoon_labels.append('No Typhoon' if typhoon_flag == 0 else 'Typhoon')
        
        if typhoon_residuals:
            axes[1, 1].boxplot(typhoon_residuals, labels=typhoon_labels)
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals by Typhoon Impact')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"viz_yield_residuals_{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved residuals plot: {plot_path}")


def plot_temporal_analysis(test_index: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, 
                         model_name: str, output_dir: str):
    """Plot temporal analysis of predictions"""
    # Create time index
    test_index['date'] = pd.to_datetime(test_index['year'].astype(str) + '-' + 
                                       (test_index['quarter'] * 3).astype(str) + '-15')
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Time series plot
    axes[0].plot(test_index['date'], y_true, 'o-', label='Actual', linewidth=2, markersize=6)
    axes[0].plot(test_index['date'], y_pred, 's-', label='Predicted', linewidth=2, markersize=6)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Yield (t/ha)')
    axes[0].set_title(f'{model_name.upper()} - Yield Predictions Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals over time
    residuals = y_true - y_pred
    axes[1].plot(test_index['date'], residuals, 'o-', color='red', linewidth=2, markersize=6)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"viz_yield_timeseries_{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal analysis plot: {plot_path}")


def plot_seasonal_analysis(test_index: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, 
                          model_name: str, output_dir: str):
    """Plot seasonal analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted by quarter
    quarters = [1, 2, 3, 4]
    actual_means = []
    pred_means = []
    actual_stds = []
    pred_stds = []
    
    for quarter in quarters:
        mask = test_index['quarter'] == quarter
        if mask.sum() > 0:
            actual_means.append(y_true[mask].mean())
            pred_means.append(y_pred[mask].mean())
            actual_stds.append(y_true[mask].std())
            pred_stds.append(y_pred[mask].std())
        else:
            actual_means.append(0)
            pred_means.append(0)
            actual_stds.append(0)
            pred_stds.append(0)
    
    x = np.arange(len(quarters))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, actual_means, width, label='Actual', alpha=0.8)
    axes[0, 0].bar(x + width/2, pred_means, width, label='Predicted', alpha=0.8)
    axes[0, 0].set_xlabel('Quarter')
    axes[0, 0].set_ylabel('Mean Yield (t/ha)')
    axes[0, 0].set_title('Mean Yield by Quarter')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f'Q{q}' for q in quarters])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Standard deviation by quarter
    axes[0, 1].bar(x - width/2, actual_stds, width, label='Actual', alpha=0.8)
    axes[0, 1].bar(x + width/2, pred_stds, width, label='Predicted', alpha=0.8)
    axes[0, 1].set_xlabel('Quarter')
    axes[0, 1].set_ylabel('Std Dev Yield (t/ha)')
    axes[0, 1].set_title('Yield Variability by Quarter')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f'Q{q}' for q in quarters])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Box plot by quarter
    quarter_data = []
    quarter_labels = []
    for quarter in quarters:
        mask = test_index['quarter'] == quarter
        if mask.sum() > 0:
            quarter_data.append(y_true[mask])
            quarter_labels.append(f'Actual Q{quarter}')
            quarter_data.append(y_pred[mask])
            quarter_labels.append(f'Pred Q{quarter}')
    
    if quarter_data:
        axes[1, 0].boxplot(quarter_data, labels=quarter_labels)
        axes[1, 0].set_ylabel('Yield (t/ha)')
        axes[1, 0].set_title('Yield Distribution by Quarter')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Year-over-year comparison
    years = sorted(test_index['year'].unique())
    year_actual = []
    year_pred = []
    year_labels = []
    
    for year in years:
        mask = test_index['year'] == year
        if mask.sum() > 0:
            year_actual.append(y_true[mask].mean())
            year_pred.append(y_pred[mask].mean())
            year_labels.append(str(year))
    
    if year_actual:
        x = np.arange(len(years))
        axes[1, 1].bar(x - width/2, year_actual, width, label='Actual', alpha=0.8)
        axes[1, 1].bar(x + width/2, year_pred, width, label='Predicted', alpha=0.8)
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Mean Yield (t/ha)')
        axes[1, 1].set_title('Mean Yield by Year')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(year_labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"viz_yield_seasonal_{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved seasonal analysis plot: {plot_path}")


def plot_feature_importance(model, feature_names: list, model_name: str, output_dir: str):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
        # For pipeline models
        regressor = model.named_steps['regressor']
        if hasattr(regressor, 'coef_'):
            importances = np.abs(regressor.coef_)
        else:
            print(f"No feature importance available for {model_name}")
            return
    else:
        print(f"No feature importance available for {model_name}")
        return
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    top_features = feature_importance_df.head(20)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name.upper()} - Feature Importance')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f"viz_feature_importance_{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot: {plot_path}")


def create_comprehensive_visualizations(config_path: str, model_name: str):
    """Create all visualizations for a specific model"""
    cfg = load_config(config_path)
    output_dir = cfg['paths']['output_dir']
    
    print(f"Creating visualizations for {model_name}...")
    
    # Load predictions
    try:
        predictions_df = load_predictions(output_dir, model_name)
    except FileNotFoundError:
        print(f"Predictions file not found for {model_name}. Skipping visualizations.")
        return
    
    # Load actual data for comparison
    ds, pre, train_df, test_df = prepare_datasets(cfg)
    
    # Extract actual and predicted values
    y_true = ds.y_test
    y_pred = predictions_df['rice_yield'].values
    
    # Create visualizations
    plot_yield_parity(y_true, y_pred, model_name, output_dir)
    plot_residuals(y_true, y_pred, ds.test_index, model_name, output_dir)
    plot_temporal_analysis(ds.test_index, y_true, y_pred, model_name, output_dir)
    plot_seasonal_analysis(ds.test_index, y_true, y_pred, model_name, output_dir)
    
    # Try to load and plot feature importance if model is available
    try:
        # Load the trained model
        import joblib
        model_files = [f for f in os.listdir(cfg['paths']['models_dir']) 
                      if f.startswith(model_name) and f.endswith('.joblib')]
        if model_files:
            # Load the most recent model
            model_files.sort()
            model_path = os.path.join(cfg['paths']['models_dir'], model_files[-1])
            model = joblib.load(model_path)
            
            # For GridSearchCV, extract the best estimator
            if hasattr(model, 'best_estimator_'):
                model = model.best_estimator_
            
            # For pipeline models, get feature names from preprocessor
            feature_names = pre.feature_columns_
            plot_feature_importance(model, feature_names, model_name, output_dir)
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")
    
    print(f"All visualizations completed for {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive visualizations for rice yield predictions')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--model', type=str, choices=['mlr', 'rf', 'gbr', 'all'], 
                       default='all', help='Model to visualize (default: all)')
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['mlr', 'rf', 'gbr']
    else:
        models = [args.model]
    
    for model_name in models:
        try:
            create_comprehensive_visualizations(args.config, model_name)
        except Exception as e:
            print(f"Error creating visualizations for {model_name}: {e}")
    
    print("Visualization process completed!")


if __name__ == '__main__':
    main()
