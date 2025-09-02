# Configuration Guide

This guide explains the new configuration options available in `config.yaml` for controlling model behavior and hyperparameters.

## Model Configuration Options

### Global Model Settings

The `models` section in `config.yaml` now includes:

- **`save_model`**: Global setting to control whether models are saved as joblib files
  - `true` (default): Save all models
  - `false`: Don't save any models

### Individual Model Configurations

Each model type has its own configuration section that can override global settings:

#### Multiple Linear Regression (MLR)
```yaml
models:
  mlr:
    save_model: true  # Override global setting for this model
```

#### Random Forest (RF)
```yaml
models:
  rf:
    save_model: true      # Override global setting
    n_estimators: 300     # Number of trees
    max_depth: null       # None for unlimited depth
    min_samples_split: 2  # Minimum samples to split
    min_samples_leaf: 1   # Minimum samples per leaf
    max_features: 'sqrt'  # Feature selection method
    bootstrap: true       # Use bootstrapping
    n_jobs: -1           # Use all CPU cores
    oob_score: false      # Out-of-bag score
```

#### Gradient Boosting (GBR)
```yaml
models:
  gbr:
    save_model: true           # Override global setting
    n_estimators: 100          # Number of boosting stages
    learning_rate: 0.1         # Learning rate
    max_depth: 3               # Maximum depth of trees
    min_samples_split: 2       # Minimum samples to split
    min_samples_leaf: 1        # Minimum samples per leaf
    subsample: 1.0             # Fraction of samples for fitting
    max_features: null         # None for all features
    loss: 'squared_error'       # Loss function
```

## Configuration Hierarchy

The save_model setting follows this hierarchy:
1. Model-specific setting (e.g., `models.rf.save_model`)
2. Global setting (`models.save_model`)
3. Default value (`true`)

## Example Configurations

### Don't Save Any Models
```yaml
models:
  save_model: false
```

### Save Only Specific Models
```yaml
models:
  save_model: false      # Global: don't save
  mlr:
    save_model: true     # Override: save MLR
  rf:
    save_model: false    # Override: don't save RF
  gbr: {}               # Use global: don't save GBR
```

### Custom Hyperparameters
```yaml
models:
  save_model: true
  rf:
    n_estimators: 500    # More trees
    max_depth: 10       # Limit depth
    max_features: 'log2' # Use log2 features
  gbr:
    n_estimators: 200    # More estimators
    learning_rate: 0.05  # Lower learning rate
    max_depth: 5         # Deeper trees
```

## Usage Examples

Run models with default settings:
```bash
python src/mlr_model.py --config config.yaml
python src/rf_model.py --config config.yaml
python src/gbr_model.py --config config.yaml
```

Run models without saving:
```bash
python src/mlr_model.py --config test_config_no_save.yaml
python src/rf_model.py --config test_config_no_save.yaml
python src/gbr_model.py --config test_config_no_save.yaml
```

Run models with custom hyperparameters:
```bash
python src/rf_model.py --config test_config_custom_params.yaml
python src/gbr_model.py --config test_config_custom_params.yaml
```

## Benefits

1. **Flexible Model Saving**: Control whether to save models based on your needs
2. **Hyperparameter Tuning**: Easily experiment with different model configurations
3. **Resource Management**: Save disk space by not saving models when not needed
4. **Reproducibility**: All model settings are version-controlled in config files
5. **Easy Experimentation**: Create multiple config files for different experiments
