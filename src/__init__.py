"""
Car Price Prediction Package

A production-ready ML pipeline for predicting used car prices.

Modules:
    - data: Data loading and preprocessing
    - features: Feature engineering
    - models: Model training and prediction
    - visualization: Plotting and reporting
"""

from src.config import get_config, config
from src.data import (
    load_raw_data,
    clean_data,
    split_data,
    run_preprocessing_pipeline,
)
from src.features import (
    FeatureEngineer,
    build_features,
)
from src.models import (
    train_all_models,
    select_best_model,
    save_model,
    load_model,
    predict_from_raw,
    batch_predict,
)
from src.visualization import (
    generate_eda_report,
    generate_model_report,
)

__version__ = "1.0.0"
__author__ = "3rd Year Data Science Student"

__all__ = [
    # Config
    "get_config",
    "config",
    # Data
    "load_raw_data",
    "clean_data",
    "split_data",
    "run_preprocessing_pipeline",
    # Features
    "FeatureEngineer",
    "build_features",
    # Models
    "train_all_models",
    "select_best_model",
    "save_model",
    "load_model",
    "predict_from_raw",
    "batch_predict",
    # Visualization
    "generate_eda_report",
    "generate_model_report",
]