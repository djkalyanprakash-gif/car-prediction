"""
Model training module for Car Price Prediction.
Handles model selection, hyperparameter tuning, and evaluation.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.base import BaseEstimator
import joblib
import warnings

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.config import get_config


config = get_config()


def get_models() -> Dict[str, BaseEstimator]:
    """Get dictionary of models to train."""
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    if HAS_XGB:
        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=config.get("model.xgboost.n_estimators", 500),
            max_depth=config.get("model.xgboost.max_depth", 6),
            learning_rate=config.get("model.xgboost.learning_rate", 0.1),
            subsample=config.get("model.xgboost.subsample", 0.8),
            colsample_bytree=config.get("model.xgboost.colsample_bytree", 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if HAS_LGB:
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=config.get("model.lightgbm.n_estimators", 500),
            max_depth=config.get("model.lightgbm.max_depth", 6),
            learning_rate=config.get("model.lightgbm.learning_rate", 0.1),
            num_leaves=config.get("model.lightgbm.num_leaves", 31),
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    if HAS_CAT:
        models["catboost"] = cb.CatBoostRegressor(
            iterations=config.get("model.catboost.iterations", 500),
            depth=config.get("model.catboost.depth", 6),
            learning_rate=config.get("model.catboost.learning_rate", 0.1),
            random_state=42,
            verbose=False,
            thread_count=-1,
        )

    return models


def evaluate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> Dict[str, float]:
    """
    Evaluate model using cross-validation.

    Returns:
        Dictionary of metric scores (mean ± std)
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Negative RMSE for sklearn (higher is better)
    rmse_scores = -cross_val_score(
        model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    mae_scores = -cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)

    return {
        "rmse_mean": rmse_scores.mean(),
        "rmse_std": rmse_scores.std(),
        "mae_mean": mae_scores.mean(),
        "mae_std": mae_scores.std(),
        "r2_mean": r2_scores.mean(),
        "r2_std": r2_scores.std(),
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> BaseEstimator:
    """Train a single model."""
    model.fit(X_train, y_train)

    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val.values, y_pred)
        print(f"  Validation metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.4f}")

    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    cv_folds: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models and return results.

    Returns:
        Dictionary with model names as keys, containing:
        - 'model': fitted model
        - 'cv_results': cross-validation results
        - 'val_metrics': validation metrics (if val set provided)
    """
    if cv_folds is None:
        cv_folds = config.get("model.cv_folds", 5)

    models = get_models()
    results = {}

    print(f"Training {len(models)} models with {cv_folds}-fold CV...")
    print("-" * 60)

    for name, model in models.items():
        print(f"\n{name}:")

        # Cross-validation
        cv_results = evaluate_model(model, X_train, y_train, cv_folds)
        print(f"  CV RMSE: {cv_results['rmse_mean']:.2f} ± {cv_results['rmse_std']:.2f}")
        print(f"  CV MAE:  {cv_results['mae_mean']:.2f} ± {cv_results['mae_std']:.2f}")
        print(f"  CV R²:   {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")

        # Train on full training data
        model.fit(X_train, y_train)

        val_metrics = None
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            val_metrics = calculate_metrics(y_val.values, y_pred)
            print(f"  Val RMSE: {val_metrics['rmse']:.2f}, MAE: {val_metrics['mae']:.2f}, R²: {val_metrics['r2']:.4f}")

        results[name] = {
            "model": model,
            "cv_results": cv_results,
            "val_metrics": val_metrics,
        }

    return results


def optimize_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.

    Args:
        model_name: Name of model to optimize
        X_train: Training features
        y_train: Training target
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds

    Returns:
        Best parameters dict
    """
    if not HAS_OPTUNA:
        raise ImportError("Optuna not installed. Install with: pip install optuna")

    if n_trials is None:
        n_trials = config.get("model.optuna_trials", 50)
    if timeout is None:
        timeout = config.get("model.optuna_timeout", 1800)

    def objective(trial):
        if model_name == "xgboost" and HAS_XGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            }
            model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)

        elif model_name == "lightgbm" and HAS_LGB:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            }
            model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbosity=-1)

        elif model_name == "catboost" and HAS_CAT:
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
                "border_count": trial.suggest_int("border_count", 32, 255),
            }
            model = cb.CatBoostRegressor(**params, random_state=42, verbose=False, thread_count=-1)

        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)

        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Cross-validation score
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print(f"Best {model_name} params: {study.best_params}")
    print(f"Best CV RMSE: {study.best_value:.4f}")

    return study.best_params


def train_with_optimization(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """Train model with hyperparameter optimization."""
    if not config.get("model.use_optuna", True):
        # Use default params
        models = get_models()
        model = models[model_name]
        model.fit(X_train, y_train)
        return model, {}

    print(f"\nOptimizing {model_name} hyperparameters...")
    best_params = optimize_hyperparameters(model_name, X_train, y_train)

    # Train with best params
    if model_name == "xgboost" and HAS_XGB:
        model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    elif model_name == "lightgbm" and HAS_LGB:
        model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=-1)
    elif model_name == "catboost" and HAS_CAT:
        model = cb.CatBoostRegressor(**best_params, random_state=42, verbose=False, thread_count=-1)
    elif model_name == "random_forest":
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)

    val_metrics = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        val_metrics = calculate_metrics(y_val.values, y_pred)
        print(f"Validation: RMSE={val_metrics['rmse']:.2f}, MAE={val_metrics['mae']:.2f}, R²={val_metrics['r2']:.4f}")

    return model, {"best_params": best_params, "val_metrics": val_metrics}


def select_best_model(
    results: Dict[str, Dict[str, Any]],
    metric: str = "rmse_mean",
) -> Tuple[str, BaseEstimator]:
    """
    Select best model based on CV metric.

    Args:
        results: Output from train_all_models
        metric: Metric to optimize (lower is better for rmse/mae, higher for r2)

    Returns:
        (best_model_name, best_model)
    """
    if metric in ["rmse_mean", "mae_mean"]:
        best_name = min(results.keys(), key=lambda k: results[k]["cv_results"][metric])
    else:  # r2_mean
        best_name = max(results.keys(), key=lambda k: results[k]["cv_results"][metric])

    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} ({metric}={results[best_name]['cv_results'][metric]:.4f})")
    return best_name, best_model


def save_model(model: BaseEstimator, name: str, path: Optional[Path] = None) -> None:
    """Save trained model."""
    if path is None:
        path = Path(config.get("output.model_dir", "models")) / f"{name}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Saved model to {path}")


def load_model(name: str, path: Optional[Path] = None) -> BaseEstimator:
    """Load trained model."""
    if path is None:
        path = Path(config.get("output.model_dir", "models")) / f"{name}.joblib"
    return joblib.load(path)


if __name__ == "__main__":
    # Quick test
    from src.data.preprocessing import run_preprocessing_pipeline
    from src.features.build_features import build_features

    train_df, test_df = run_preprocessing_pipeline()
    X_train, X_test, y_train, y_test = build_features(train_df, test_df)

    results = train_all_models(X_train, y_train, X_test, y_test)
    best_name, best_model = select_best_model(results)
    save_model(best_model, "best_model")