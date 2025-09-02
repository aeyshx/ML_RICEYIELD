# Rice Yield Prediction Visualization

This script (`viz_yield_predictions.py`) creates comprehensive visualizations for rice yield model predictions versus ground truth data.

## Features

- **Time Series Analysis**: Quarterly rice yield predictions vs ground truth over time
- **Parity Plots**: Scatter plots with y=x reference line and metrics
- **Residual Diagnostics**: Residual analysis with trend lines
- **Seasonal Analysis**: Box plots showing seasonal distributions
- **STY Event Highlighting**: Optional highlighting of Super Typhoon quarters
- **Metrics Export**: CSV files with RMSE, MAE, R², and MAPE metrics

## Usage

```bash
# Basic usage
python viz_yield_predictions.py --config config.yaml --model_name mlr
python viz_yield_predictions.py --config config.yaml --model_name rf
python viz_yield_predictions.py --config config.yaml --model_name gbr
```

## Input Requirements

### Required Files
- `config.yaml`: Configuration file with paths and parameters
- `data/AGRICULTURE.csv`: Ground truth data with rice yield information
- `outputs/predictions_[model].csv`: Model predictions for the specified model

### Optional Files
- `data/DISASTERS.csv`: Disaster events for STY quarter highlighting

## Output Files

For each model, the script generates:

### Visualization Files
- `viz_yield_timeseries_[model].png`: Time series overlay plot
- `viz_yield_parity_[model].png`: Parity scatter plot with metrics
- `viz_yield_residuals_[model].png`: Residual diagnostics
- `viz_yield_seasonal_[model].png`: Seasonal distribution analysis

### Metrics Files
- `metrics_[model].csv`: Performance metrics (RMSE, MAE, R², MAPE)

### Sample Visualizations (if available)
- `sample/sample_viz_yield_timeseries_[model].png`: Sample data visualizations

## Configuration

The script reads configuration from `config.yaml`:

```yaml
paths:
  data_dir: data
  output_dir: outputs
  models_dir: models

split_years:
  train_start: 2000
  train_end: 2018
  test_start: 2019
  test_end: 2023
```

## Features

### Robust Data Parsing
- Handles thousands separators in numeric data
- Validates computed rice yield against provided values
- Graceful handling of missing data

### Advanced Visualizations
- Color-blind friendly palettes
- High-resolution output (200 DPI)
- Professional styling with Seaborn (if available)
- LOWESS trend lines for residual analysis

### STY Event Integration
- Automatically detects Super Typhoon quarters
- Highlights STY events in time series plots
- Provides STY quarter annotations

### Error Handling
- Clear error messages for missing files
- Graceful degradation for optional dependencies
- Validation of data integrity

## Dependencies

### Required
- Python 3.7+
- pandas
- numpy
- matplotlib
- scikit-learn
- PyYAML

### Optional
- seaborn (for enhanced styling)
- statsmodels (for LOWESS smoothing)

## Example Output

The script provides detailed console output showing:
- Configuration summary
- Data loading statistics
- STY quarter detection
- Test data points and range
- Generated file paths
- Model performance metrics

## Model Performance Summary

Based on the test results (2019-2023):

| Model | RMSE (t/ha) | MAE (t/ha) | R² | MAPE (%) |
|-------|-------------|------------|----|----------|
| MLR   | 0.690       | 0.600      | -2.008 | 15.2 |
| RF    | 0.734       | 0.663      | -2.408 | 16.2 |
| GBR   | 0.747       | 0.651      | -2.524 | 16.0 |

Note: Negative R² values indicate that the models perform worse than a simple mean prediction, suggesting potential overfitting or data issues.
