# Rice Yield Prediction Visualization and Diagnostics

This document describes the comprehensive visualization and diagnostic capabilities added to the rice yield prediction system.

## Overview

The enhanced system now includes sophisticated visualization tools to analyze model performance, understand feature importance, and diagnose prediction patterns across different temporal and seasonal contexts.

## Key Improvements Implemented

### 1. Enhanced Feature Engineering
- **Temperature Range Features**: Added `temp_range` (max - min) and `temp_mean` for better heat stress modeling
- **Multi-Quarter Lags**: Implemented 1, 2, and 3-quarter lags for climate variables to capture cumulative effects
- **Rolling Aggregates**: Added 2Q and 3Q rolling means for rainfall, humidity, and temperature
- **Enhanced Typhoon Features**: Extended from binary STY flag to include `sty_count`, `storm_count`, and `msw_max`

### 2. Time-Series Cross-Validation
- **TimeSeriesSplit**: Implemented proper time-series cross-validation to prevent data leakage
- **GridSearchCV**: Added hyperparameter tuning for Random Forest and Gradient Boosting models
- **Configurable Parameters**: All tuning parameters configurable via `config.yaml`

### 3. Model Regularization
- **Ridge/Lasso/ElasticNet**: Added regularization options for linear models
- **StandardScaler**: Implemented preprocessing pipeline for better coefficient stability
- **Configurable Alpha**: Regularization strength configurable via config

### 4. Comprehensive Diagnostics
- **Per-Quarter Metrics**: Performance analysis by quarter to identify seasonal patterns
- **Per-Typhoon Metrics**: Performance analysis for typhoon vs non-typhoon quarters
- **Detailed Residual Analysis**: Comprehensive residual diagnostics and visualization

## Visualization Types

### 1. Yield Parity Plots (`viz_yield_parity_*.png`)
- **Purpose**: Compare actual vs predicted yields
- **Features**: 
  - Scatter plot with perfect prediction line
  - RMSE, MAE, and R² metrics displayed
  - Clear visualization of prediction accuracy

### 2. Residual Analysis (`viz_yield_residuals_*.png`)
- **Purpose**: Diagnose prediction errors and patterns
- **Features**:
  - Residuals vs predicted values
  - Residual distribution histogram
  - Residuals by quarter (seasonal patterns)
  - Residuals by typhoon impact (shock analysis)

### 3. Temporal Analysis (`viz_yield_timeseries_*.png`)
- **Purpose**: Analyze predictions over time
- **Features**:
  - Time series of actual vs predicted yields
  - Residuals over time to identify trends
  - Temporal pattern identification

### 4. Seasonal Analysis (`viz_yield_seasonal_*.png`)
- **Purpose**: Understand seasonal patterns and variability
- **Features**:
  - Mean yield by quarter (actual vs predicted)
  - Yield variability by quarter
  - Box plots showing distribution by quarter
  - Year-over-year comparison

### 5. Feature Importance (`viz_feature_importance_*.png`)
- **Purpose**: Understand which features drive predictions
- **Features**:
  - Top 20 most important features
  - Importance scores for tree-based models
  - Coefficient magnitudes for linear models

## Usage

### Running Visualizations

```bash
# Create visualizations for all models
python viz_yield_predictions.py --config config.yaml --model all

# Create visualizations for specific model
python viz_yield_predictions.py --config config.yaml --model rf
python viz_yield_predictions.py --config config.yaml --model gbr
python viz_yield_predictions.py --config config.yaml --model mlr
```

### Configuration Options

The visualization system respects all configuration options in `config.yaml`:

```yaml
features:
  use_quarter_dummies: true
  use_multi_quarter_lags: true
  use_rolling_features: true
  use_enhanced_typhoon: true

models:
  rf:
    use_hyperparameter_tuning: true
    n_splits: 5
    n_estimators: [100, 200, 300, 500]
    max_depth: [None, 5, 10, 15]
  
  gbr:
    use_hyperparameter_tuning: true
    n_splits: 5
    loss: ['squared_error', 'huber']
    
  mlr:
    use_regularization: true
    regularization_type: 'ridge'
    alpha: 1.0
```

## Output Files

### Predictions and Metrics
- `predictions_[model].csv`: Test period predictions with actual values
- `summary_[model].csv`: Basic summary statistics
- `metrics_[model].csv`: Detailed metrics including per-quarter and per-typhoon analysis

### Visualizations
- `viz_yield_parity_[model].png`: Actual vs predicted comparison
- `viz_yield_residuals_[model].png`: Comprehensive residual analysis
- `viz_yield_timeseries_[model].png`: Temporal analysis
- `viz_yield_seasonal_[model].png`: Seasonal pattern analysis
- `viz_feature_importance_[model].png`: Feature importance ranking

## Diagnostic Insights

### 1. Seasonal Performance Analysis
The system automatically analyzes performance by quarter to identify:
- Which quarters are hardest to predict
- Seasonal bias in predictions
- Impact of growing season vs harvest season

### 2. Typhoon Impact Analysis
Enhanced typhoon features help identify:
- Performance during typhoon quarters
- Whether typhoon intensity features improve predictions
- Residual patterns during extreme weather events

### 3. Feature Importance Insights
Feature importance analysis reveals:
- Which climate variables are most predictive
- Impact of lag features on prediction accuracy
- Relative importance of typhoon features vs climate features

### 4. Temporal Stability
Time series analysis shows:
- Whether model performance degrades over time
- Systematic over/under-prediction patterns
- Impact of changing climate patterns

## Best Practices

### 1. Model Comparison
- Compare visualizations across all three models
- Look for consistent patterns vs model-specific issues
- Use feature importance to understand model differences

### 2. Seasonal Analysis
- Pay special attention to Q2-Q3 (growing season) vs Q4-Q1 (harvest season)
- Check if rolling features improve seasonal predictions
- Verify that lag features align with crop calendar

### 3. Typhoon Analysis
- Compare performance during typhoon vs non-typhoon quarters
- Check if enhanced typhoon features improve predictions
- Look for systematic bias during extreme weather

### 4. Feature Engineering Validation
- Use feature importance to validate new features
- Check if multi-quarter lags provide value
- Verify that rolling aggregates capture meaningful patterns

## Troubleshooting

### Common Issues

1. **Missing Predictions File**: Ensure models have been trained first
2. **No Feature Importance**: Some models (e.g., standard LinearRegression) don't provide feature importance
3. **Empty Visualizations**: Check if test period has sufficient data
4. **Memory Issues**: Large datasets may require reducing plot resolution

### Performance Tips

1. **Parallel Processing**: Use `n_jobs=-1` for faster hyperparameter tuning
2. **Reduced Grid Search**: Start with smaller parameter grids for faster iteration
3. **Selective Visualization**: Use `--model` flag to create only needed visualizations

## Future Enhancements

### Planned Features
1. **Interactive Dashboards**: Web-based interactive visualizations
2. **Automated Reporting**: PDF reports with key insights
3. **Model Comparison Tools**: Side-by-side model performance comparison
4. **Anomaly Detection**: Automatic identification of prediction anomalies

### Research Directions
1. **Ensemble Methods**: Combine predictions from multiple models
2. **Deep Learning**: Explore neural network approaches
3. **Causal Inference**: Understand causal relationships between features and yield
4. **Uncertainty Quantification**: Provide prediction confidence intervals

## Conclusion

The enhanced visualization and diagnostic system provides comprehensive tools for understanding model performance, identifying improvement opportunities, and validating feature engineering decisions. The combination of enhanced features, proper time-series validation, and detailed diagnostics creates a robust framework for rice yield prediction.
