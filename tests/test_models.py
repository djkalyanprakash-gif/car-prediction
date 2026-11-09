"""
Tests for model training module.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.train import (
    get_models,
    evaluate_model,
    calculate_metrics,
    train_model,
    train_all_models,
    select_best_model,
)
from src.models.predict import evaluate_predictions


class TestModelTraining:
    """Tests for model training functions."""

    def create_sample_data(self, n=100):
        """Create sample features and target."""
        np.random.seed(42)
        X = pd.DataFrame({
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randint(0, 5, n),
            "feature_4": np.random.randn(n),
        })
        # Create target with some relationship to features
        y = pd.Series(
            100000 + 50000 * X["feature_1"] - 30000 * X["feature_2"] + 10000 * X["feature_3"] + np.random.randn(n) * 10000,
            name="selling_price"
        )
        return X, y

    def test_get_models(self):
        """Test get_models returns expected models."""
        models = get_models()

        assert "linear_regression" in models
        assert "ridge" in models
        assert "random_forest" in models

        # Check they're sklearn estimators
        from sklearn.base import BaseEstimator
        for name, model in models.items():
            assert isinstance(model, BaseEstimator)

    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])

        metrics = calculate_metrics(y_true, y_pred)

        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics

        # Check values are reasonable
        assert metrics["rmse"] > 0
        assert metrics["mae"] > 0
        assert metrics["r2"] <= 1.0
        assert metrics["mape"] >= 0

    def test_evaluate_model_cv(self):
        """Test cross-validation evaluation."""
        from sklearn.linear_model import LinearRegression

        X, y = self.create_sample_data(50)
        model = LinearRegression()

        results = evaluate_model(model, X, y, cv_folds=3)

        assert "rmse_mean" in results
        assert "rmse_std" in results
        assert "mae_mean" in results
        assert "mae_std" in results
        assert "r2_mean" in results
        assert "r2_std" in results

        # Std should be non-negative
        assert results["rmse_std"] >= 0
        assert results["mae_std"] >= 0
        assert results["r2_std"] >= 0

    def test_train_model(self):
        """Test single model training."""
        from sklearn.linear_model import LinearRegression

        X, y = self.create_sample_data(50)
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]

        model = LinearRegression()
        trained = train_model(model, X_train, y_train, X_val, y_val)

        # Model should be fitted
        assert hasattr(trained, "coef_")

    def test_train_all_models(self):
        """Test training all models."""
        X, y = self.create_sample_data(80)
        X_train, X_val = X[:60], X[60:]
        y_train, y_val = y[:60], y[60:]

        results = train_all_models(X_train, y_train, X_val, y_val, cv_folds=3)

        assert len(results) >= 3  # At least LR, Ridge, RF

        for name, result in results.items():
            assert "model" in result
            assert "cv_results" in result
            assert "val_metrics" in result
            assert result["val_metrics"] is not None

    def test_select_best_model(self):
        """Test best model selection."""
        X, y = self.create_sample_data(80)
        X_train, X_val = X[:60], X[60:]
        y_train, y_val = y[:60], y[60:]

        results = train_all_models(X_train, y_train, X_val, y_val, cv_folds=3)

        best_name, best_model = select_best_model(results, metric="rmse_mean")

        assert best_name in results
        assert best_model is results[best_name]["model"]

        # Best should have lowest RMSE
        best_rmse = results[best_name]["cv_results"]["rmse_mean"]
        for name, result in results.items():
            assert result["cv_results"]["rmse_mean"] >= best_rmse - 1e-10

    def test_select_best_model_r2(self):
        """Test best model selection with R2 (higher is better)."""
        X, y = self.create_sample_data(80)
        X_train, X_val = X[:60], X[60:]
        y_train, y_val = y[:60], y[60:]

        results = train_all_models(X_train, y_train, X_val, y_val, cv_folds=3)

        best_name, best_model = select_best_model(results, metric="r2_mean")

        # Best should have highest R2
        best_r2 = results[best_name]["cv_results"]["r2_mean"]
        for name, result in results.items():
            assert result["cv_results"]["r2_mean"] <= best_r2 + 1e-10


class TestEvaluatePredictions:
    """Tests for prediction evaluation."""

    def test_evaluate_predictions_output(self):
        """Test evaluate_predictions returns metrics dict."""
        y_true = np.array([100000, 200000, 300000, 400000, 500000])
        y_pred = np.array([110000, 190000, 310000, 390000, 510000])

        metrics = evaluate_predictions(y_true, y_pred, "TestModel")

        assert isinstance(metrics, dict)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mape" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])