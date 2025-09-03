#!/usr/bin/env python3
"""
Rice Yield Prediction Visualization Script

This script visualizes quarterly rice yield model predictions versus ground truth
using Matplotlib and Seaborn, saving high-resolution figures and metrics CSV.

Usage:
    python viz_yield_predictions.py --config config.yaml --model_name mlr
    python viz_yield_predictions.py --config config.yaml --model_name rf
    python viz_yield_predictions.py --config config.yaml --model_name gbr
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
import glob

# Try to import seaborn for enhanced styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: Seaborn not available. Using basic matplotlib styling.")

# Try to import statsmodels for LOWESS
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    LOWESS_AVAILABLE = True
except ImportError:
    LOWESS_AVAILABLE = False
    print("Warning: Statsmodels not available. Using simple moving average for trend lines.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize rice yield predictions vs ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python viz_yield_predictions.py --config config.yaml --model_name mlr
    python viz_yield_predictions.py --config config.yaml --model_name rf
    python viz_yield_predictions.py --config config.yaml --model_name gbr
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        choices=["mlr", "rf", "gbr"],
        help="Model name for predictions file"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)


def robust_numeric_parse(value):
    """Robustly parse numeric values, handling thousands separators."""
    if pd.isna(value) or value == '':
        return np.nan
    
    # Convert to string and remove thousands separators
    if isinstance(value, str):
        # Remove quotes and thousands separators
        value = value.replace('"', '').replace(',', '')
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def load_ground_truth(data_dir):
    """Load and validate ground truth data from AGRICULTURE.csv."""
    agriculture_path = os.path.join(data_dir, "AGRICULTURE.csv")
    
    if not os.path.exists(agriculture_path):
        print(f"Error: Ground truth file '{agriculture_path}' not found.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(agriculture_path)
        
        # Parse numeric columns with thousands separators
        df['produced_rice'] = df['produced_rice'].apply(robust_numeric_parse)
        df['area_harvested'] = df['area_harvested'].apply(robust_numeric_parse)
        
        # Compute rice_yield_true for validation
        df['rice_yield_true'] = df['produced_rice'] / np.clip(df['area_harvested'], 1e-6, None)
        
        # Validate against provided rice_yield column
        yield_diff = np.abs(df['rice_yield_true'] - df['rice_yield']).max()
        if yield_diff > 1e-6:
            print(f"Warning: Computed rice_yield differs from provided by max {yield_diff:.6f}")
        
        # Use computed rice_yield_true for consistency
        df['rice_yield_true'] = df['rice_yield_true']
        
        print(f"Loaded ground truth data: {len(df)} rows")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        
        return df[['year', 'quarter', 'rice_yield_true']]
        
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        sys.exit(1)


def load_predictions(output_dir, model_name):
    """Load model predictions from CSV file."""
    predictions_path = os.path.join(output_dir, f"predictions_{model_name}.csv")
    
    if not os.path.exists(predictions_path):
        print(f"Error: Predictions file '{predictions_path}' not found.")
        sys.exit(1)
    
    try:
        df = pd.read_csv(predictions_path)
        
        # Check if rice_yield column exists
        if 'rice_yield' in df.columns:
            # Rename to rice_yield_pred for clarity
            df = df.rename(columns={'rice_yield': 'rice_yield_pred'})
        elif 'rice_yield_pred' in df.columns:
            pass  # Already correctly named
        else:
            print("Error: No rice_yield column found in predictions file.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        
        print(f"Loaded predictions data: {len(df)} rows")
        print(f"Predictions columns: {list(df.columns)}")
        
        return df[['year', 'quarter', 'rice_yield_pred']]
        
    except Exception as e:
        print(f"Error loading predictions data: {e}")
        sys.exit(1)


def load_disasters(data_dir):
    """Load disasters data and create STY quarter flags."""
    disasters_path = os.path.join(data_dir, "DISASTERS.csv")
    
    if not os.path.exists(disasters_path):
        print(f"Warning: Disasters file '{disasters_path}' not found. STY highlighting disabled.")
        return None
    
    try:
        df = pd.read_csv(disasters_path)
        
        # Create STY-only binary flag
        sty_quarters = df[df['type'] == 'STY'][['year', 'quarter']].drop_duplicates()
        
        # Create a flag for all quarters
        all_quarters = pd.DataFrame([
            (year, quarter) 
            for year in range(2000, 2024) 
            for quarter in range(1, 5)
        ], columns=['year', 'quarter'])
        
        # Mark STY quarters
        all_quarters['is_sty_quarter'] = all_quarters.set_index(['year', 'quarter']).index.isin(
            sty_quarters.set_index(['year', 'quarter']).index
        ).astype(int)
        
        print(f"Loaded disasters data: {len(df)} events")
        print(f"STY quarters found: {sty_quarters.to_dict('records')}")
        
        return all_quarters
        
    except Exception as e:
        print(f"Warning: Error loading disasters data: {e}. STY highlighting disabled.")
        return None


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'mape': np.nan,
            'n_points': 0
        }
    
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # MAPE with clipping to avoid division by zero
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.clip(y_true_clean, 1e-6, None))) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'n_points': len(y_true_clean)
    }


def setup_plotting_style():
    """Setup matplotlib and seaborn styling."""
    if SEABORN_AVAILABLE:
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # Fallback for newer seaborn versions
            sns.set_style("whitegrid")
    else:
        # Basic matplotlib styling
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'lines.linewidth': 2,
            'axes.grid': True,
            'grid.alpha': 0.3
        })


def create_time_series_plot(data, model_name, output_dir, test_start, test_end):
    """Create time series overlay plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time index
    data['time_index'] = data['year'] + (data['quarter'] - 1) / 4
    
    # Filter to test period
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()
    
    # Plot true values
    ax.plot(test_data['time_index'], test_data['rice_yield_true'], 
            'o-', linewidth=2, markersize=6, label='True', color='blue')
    
    # Plot predicted values
    ax.plot(test_data['time_index'], test_data['rice_yield_pred'], 
            's--', linewidth=2, markersize=6, label='Predicted', color='red')
    
    # Highlight STY quarters if available
    if 'is_sty_quarter' in test_data.columns:
        sty_data = test_data[test_data['is_sty_quarter'] == 1]
        if len(sty_data) > 0:
            ax.scatter(sty_data['time_index'], sty_data['rice_yield_true'], 
                      s=100, facecolors='none', edgecolors='orange', 
                      linewidth=2, label='STY Quarter', zorder=5)
    
    # Formatting
    ax.set_title(f'Rice Yield (t/ha) — {model_name.upper()} — Test {test_start}-{test_end}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Rice Yield (t/ha)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis ticks
    years = range(test_start, test_end + 1)
    ax.set_xticks([y + 0.5 for y in years])
    ax.set_xticklabels([str(y) for y in years])
    
    # Annotate end-of-year points
    for year in years:
        year_data = test_data[test_data['year'] == year]
        if len(year_data) > 0:
            q4_data = year_data[year_data['quarter'] == 4]
            if len(q4_data) > 0:
                ax.annotate(f'{year} Q4', 
                           xy=(q4_data['time_index'].iloc[0], q4_data['rice_yield_true'].iloc[0]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'viz_yield_timeseries_{model_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved time series plot: {output_path}")


def create_parity_plot(data, model_name, output_dir, test_start, test_end):
    """Create parity (y=x) scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Filter to test period
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()
    
    # Create color map for quarters
    colors = plt.cm.Set1(np.linspace(0, 1, 4))
    quarter_colors = {1: colors[0], 2: colors[1], 3: colors[2], 4: colors[3]}
    
    # Scatter plot by quarter
    for quarter in range(1, 5):
        quarter_data = test_data[test_data['quarter'] == quarter]
        if len(quarter_data) > 0:
            ax.scatter(quarter_data['rice_yield_true'], quarter_data['rice_yield_pred'],
                      c=[quarter_colors[quarter]], s=60, alpha=0.7, 
                      label=f'Q{quarter}', edgecolors='black', linewidth=0.5)
    
    # Add y=x reference line
    min_val = min(test_data['rice_yield_true'].min(), test_data['rice_yield_pred'].min())
    max_val = max(test_data['rice_yield_true'].max(), test_data['rice_yield_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    # Compute and display metrics
    metrics = compute_metrics(test_data['rice_yield_true'], test_data['rice_yield_pred'])
    
    # Add metrics text box
    metrics_text = f'RMSE: {metrics["rmse"]:.3f}\nMAE: {metrics["mae"]:.3f}\nR²: {metrics["r2"]:.3f}\nMAPE: {metrics["mape"]:.1f}%'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_title(f'Rice Yield Parity Plot — {model_name.upper()} — Test {test_start}-{test_end}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('True Rice Yield (t/ha)')
    ax.set_ylabel('Predicted Rice Yield (t/ha)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'viz_yield_parity_{model_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parity plot: {output_path}")


def create_residuals_plot(data, model_name, output_dir, test_start, test_end):
    """Create residual diagnostics plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter to test period
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()
    
    # Compute residuals
    test_data['residuals'] = test_data['rice_yield_pred'] - test_data['rice_yield_true']
    test_data['time_index'] = test_data['year'] + (test_data['quarter'] - 1) / 4
    
    # Subplot A: Residuals vs Time
    ax1.plot(test_data['time_index'], test_data['residuals'], 'o-', linewidth=1.5, markersize=4)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Annotate largest absolute residuals
    largest_residuals = test_data.loc[test_data['residuals'].abs().nlargest(3).index]
    for _, row in largest_residuals.iterrows():
        ax1.annotate(f'{row["year"]}-Q{row["quarter"]}', 
                    xy=(row['time_index'], row['residuals']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    ax1.set_title('Residuals vs Time')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Residuals (t/ha)')
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis
    years = range(test_start, test_end + 1)
    ax1.set_xticks([y + 0.5 for y in years])
    ax1.set_xticklabels([str(y) for y in years])
    
    # Subplot B: Residuals vs Predicted
    ax2.scatter(test_data['rice_yield_pred'], test_data['residuals'], alpha=0.6, s=50)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add trend line
    if LOWESS_AVAILABLE:
        # Use LOWESS smoothing
        lowess_result = lowess(test_data['residuals'], test_data['rice_yield_pred'], 
                              frac=0.3, it=3)
        ax2.plot(lowess_result[:, 0], lowess_result[:, 1], 'g-', linewidth=2, label='LOWESS')
    else:
        # Use simple moving average
        sorted_data = test_data.sort_values('rice_yield_pred')
        window_size = max(3, len(sorted_data) // 10)
        ma_residuals = sorted_data['residuals'].rolling(window=window_size, center=True).mean()
        ax2.plot(sorted_data['rice_yield_pred'], ma_residuals, 'g-', linewidth=2, label='Moving Avg')
    
    ax2.set_title('Residuals vs Predicted Values')
    ax2.set_xlabel('Predicted Rice Yield (t/ha)')
    ax2.set_ylabel('Residuals (t/ha)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Residual Diagnostics — {model_name.upper()} — Test {test_start}-{test_end}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'viz_yield_residuals_{model_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved residuals plot: {output_path}")


def create_seasonal_plot(data, model_name, output_dir, test_start, test_end):
    """Create seasonal distributions plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter to test period
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()
    
    # Prepare data for boxplot
    plot_data = []
    labels = []
    colors = []
    
    for quarter in range(1, 5):
        quarter_data = test_data[test_data['quarter'] == quarter]
        if len(quarter_data) > 0:
            # True values
            plot_data.append(quarter_data['rice_yield_true'].values)
            labels.append(f'Q{quarter} True')
            colors.append('lightblue')
            
            # Predicted values
            plot_data.append(quarter_data['rice_yield_pred'].values)
            labels.append(f'Q{quarter} Pred')
            colors.append('lightcoral')
    
    # Create boxplot
    bp = ax.boxplot(plot_data, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add median markers
    for i, data_values in enumerate(plot_data):
        median_val = np.median(data_values)
        ax.plot([i+1], [median_val], 'ko', markersize=6, zorder=5)
    
    # Add per-quarter MAE annotations
    for quarter in range(1, 5):
        quarter_data = test_data[test_data['quarter'] == quarter]
        if len(quarter_data) > 0:
            mae = mean_absolute_error(quarter_data['rice_yield_true'], quarter_data['rice_yield_pred'])
            ax.text(quarter * 2 - 0.5, ax.get_ylim()[1] * 0.95, f'MAE: {mae:.3f}', 
                   ha='center', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_title(f'Seasonal Rice Yield Distributions — {model_name.upper()} — Test {test_start}-{test_end}', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Rice Yield (t/ha)')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'viz_yield_seasonal_{model_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved seasonal plot: {output_path}")


def save_metrics(data, model_name, output_dir, test_start, test_end):
    """Save metrics to CSV file."""
    # Filter to test period
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()
    
    # Compute metrics
    metrics = compute_metrics(test_data['rice_yield_true'], test_data['rice_yield_pred'])
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame([{
        'model_name': model_name,
        'test_start': test_start,
        'test_end': test_end,
        'n_points': metrics['n_points'],
        'rmse': metrics['rmse'],
        'mae': metrics['mae'],
        'r2': metrics['r2'],
        'mape': metrics['mape']
    }])
    
    # Save to CSV
    output_path = os.path.join(output_dir, f'metrics_{model_name}.csv')
    metrics_df.to_csv(output_path, index=False)
    
    print(f"Saved metrics: {output_path}")
    print(f"Metrics for {model_name.upper()}:")
    print(f"  RMSE: {metrics['rmse']:.3f} t/ha")
    print(f"  MAE: {metrics['mae']:.3f} t/ha")
    print(f"  R²: {metrics['r2']:.3f}")
    print(f"  MAPE: {metrics['mape']:.1f}%")


def load_trained_model(models_dir, model_name):
    """Load the most recent trained model for the given model name."""
    # Find the most recent model file
    pattern = os.path.join(models_dir, f"{model_name}_*.joblib")
    model_files = glob.glob(pattern)
    
    if not model_files:
        print(f"Warning: No trained model found for {model_name}")
        return None
    
    # Get the most recent model file
    latest_model = max(model_files, key=os.path.getctime)
    
    try:
        model = joblib.load(latest_model)
        print(f"Loaded model: {latest_model}")
        return model
    except Exception as e:
        print(f"Error loading model {latest_model}: {e}")
        return None


def create_feature_importance_plot(model, model_name, output_dir, data_dir):
    """Create feature importance plot for tree-based models."""
    # Check if model has feature_importances_ attribute
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {model_name} does not support feature importance (e.g., Linear Regression)")
        return
    
    # Get feature names from the model or construct them
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # Construct feature names based on the preprocessing logic
        feature_names = [
            'rainfall', 'min_temperature', 'max_temperature', 'relative_humidity', 'wind_speed',
            'wind_dir_sin', 'wind_dir_cos', 'typhoon_impact',
            'q_1', 'q_2', 'q_3', 'q_4',  # quarter dummies
            'rainfall_lag1', 'min_temperature_lag1', 'max_temperature_lag1', 
            'relative_humidity_lag1', 'wind_speed_lag1', 'wind_dir_sin_lag1', 'wind_dir_cos_lag1'
        ]
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(importance_df)), importance_df['importance'], 
                   color='steelblue', alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Feature Importance — {model_name.upper()} Model', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center', fontsize=9)
    
    # All features use consistent color (no highlighting)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'viz_feature_importance_{model_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance plot: {output_path}")
    
    # Print top 5 most important features
    top_features = importance_df.tail(5)
    print(f"Top 5 most important features for {model_name.upper()}:")
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")


def check_sample_predictions(output_dir, model_name):
    """Check if sample predictions exist and create quick visualizations."""
    sample_path = os.path.join(output_dir, "sample", f"sample_predictions_{model_name}.csv")
    
    if not os.path.exists(sample_path):
        return
    
    print(f"Found sample predictions: {sample_path}")
    print("Creating sample visualizations...")
    
    try:
        # Load sample data
        sample_data = pd.read_csv(sample_path)
        
        if 'rice_yield' in sample_data.columns:
            sample_data = sample_data.rename(columns={'rice_yield': 'rice_yield_pred'})
        
        # Create quick time series plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        if 'year' in sample_data.columns and 'quarter' in sample_data.columns:
            sample_data['time_index'] = sample_data['year'] + (sample_data['quarter'] - 1) / 4
            
            if 'rice_yield_true' in sample_data.columns:
                ax.plot(sample_data['time_index'], sample_data['rice_yield_true'], 
                       'o-', linewidth=2, markersize=6, label='True', color='blue')
            
            ax.plot(sample_data['time_index'], sample_data['rice_yield_pred'], 
                   's--', linewidth=2, markersize=6, label='Predicted', color='red')
        
        ax.set_title(f'Sample Rice Yield Predictions — {model_name.upper()}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Rice Yield (t/ha)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save sample plot
        sample_output_path = os.path.join(output_dir, "sample", f"sample_viz_yield_timeseries_{model_name}.png")
        plt.savefig(sample_output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"Saved sample time series plot: {sample_output_path}")
        
    except Exception as e:
        print(f"Warning: Error creating sample visualizations: {e}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract paths and parameters
    data_dir = config['paths']['data_dir']
    output_dir = config['paths']['output_dir']
    models_dir = config['paths']['models_dir']
    test_start = config['split_years']['test_start']
    test_end = config['split_years']['test_end']
    model_name = args.model_name
    
    # Print configuration
    print("=" * 60)
    print("RICE YIELD PREDICTION VISUALIZATION")
    print("=" * 60)
    print(f"Configuration file: {args.config}")
    print(f"Model name: {model_name}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test period: {test_start}-{test_end}")
    print(f"Seaborn available: {SEABORN_AVAILABLE}")
    print(f"LOWESS available: {LOWESS_AVAILABLE}")
    print("=" * 60)
    
    # Setup plotting style
    setup_plotting_style()
    
    # Load data
    print("\nLoading data...")
    ground_truth = load_ground_truth(data_dir)
    predictions = load_predictions(output_dir, model_name)
    disasters = load_disasters(data_dir)
    
    # Merge data
    print("\nMerging data...")
    data = ground_truth.merge(predictions, on=['year', 'quarter'], how='inner')
    
    if disasters is not None:
        data = data.merge(disasters, on=['year', 'quarter'], how='left')
        data['is_sty_quarter'] = data['is_sty_quarter'].fillna(0)
    
    # Filter to test period
    test_data = data[(data['year'] >= test_start) & (data['year'] <= test_end)].copy()
    
    print(f"Test data points: {len(test_data)}")
    print(f"Test data range: {test_data['year'].min()}-{test_data['year'].max()}")
    
    if len(test_data) == 0:
        print("Error: No test data points found.")
        sys.exit(1)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Time series plot
    create_time_series_plot(data, model_name, output_dir, test_start, test_end)
    
    # Parity plot
    create_parity_plot(data, model_name, output_dir, test_start, test_end)
    
    # Residuals plot
    create_residuals_plot(data, model_name, output_dir, test_start, test_end)
    
    # Seasonal plot
    create_seasonal_plot(data, model_name, output_dir, test_start, test_end)
    
    # Save metrics
    save_metrics(data, model_name, output_dir, test_start, test_end)
    
    # Create feature importance plot
    print("\nCreating feature importance plot...")
    trained_model = load_trained_model(models_dir, model_name)
    if trained_model is not None:
        create_feature_importance_plot(trained_model, model_name, output_dir, data_dir)
    
    # Check for sample predictions
    check_sample_predictions(output_dir, model_name)
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"All figures saved to: {output_dir}")
    print(f"Metrics saved to: {output_dir}/metrics_{model_name}.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
