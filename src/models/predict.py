"""
Prediction module for Car Price Prediction.
Handles inference on new data.
"""

from pathlib import Path
from typing import Union, Optional, List
import pandas as pd
import numpy as np
import joblib

from src.config import get_config
from src.features.build_features import FeatureEngineer, load_feature_engineer
from src.models.train import load_model, calculate_metrics


config = get_config()


def predict(
    model,
    X: pd.DataFrame,
    feature_engineer: Optional[FeatureEngineer] = None,
) -> np.ndarray:
    """
    Make predictions on new data.

    Args:
        model: Trained model
        X: Feature DataFrame (raw or preprocessed)
        feature_engineer: Fitted FeatureEngineer (required if X is raw)

    Returns:
        Array of predictions
    """
    # Transform features if feature engineer provided
    if feature_engineer is not None:
        X = feature_engineer.transform(X)

    preds = model.predict(X)
    return preds


def predict_from_raw(
    raw_df: pd.DataFrame,
    model_name: str = "best_model",
    model_dir: Optional[Path] = None,
    fe_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Predict on raw data (with name column etc).

    Args:
        raw_df: Raw DataFrame with all original columns
        model_name: Name of saved model
        model_dir: Directory containing models
        fe_path: Path to feature engineer

    Returns:
        DataFrame with predictions added
    """
    from src.data.preprocessing import clean_data

    # Clean raw data
    df_clean = clean_data(raw_df.copy())

    # Load feature engineer and model
    if fe_path is None:
        fe_path = Path(config.get("output.model_dir", "models")) / "feature_engineer.joblib"
    fe = load_feature_engineer(fe_path)

    if model_dir is None:
        model_dir = Path(config.get("output.model_dir", "models"))
    model = load_model(model_name, model_dir)

    # Separate target if present
    target_col = config.get("data.target_column", "selling_price")
    has_target = target_col in df_clean.columns
    y_true = df_clean[target_col] if has_target else None
    X = df_clean.drop(columns=[target_col], errors="ignore")

    # Transform and predict
    X_transformed = fe.transform(X)
    preds = model.predict(X_transformed)

    # Create output
    output = raw_df.copy()
    output["predicted_price"] = preds

    if has_target:
        output["actual_price"] = y_true.values
        output["residual"] = output["actual_price"] - output["predicted_price"]
        output["abs_error"] = np.abs(output["residual"])
        output["pct_error"] = (output["abs_error"] / output["actual_price"]) * 100

    return output


def batch_predict(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model_name: str = "best_model",
    model_dir: Optional[Path] = None,
    fe_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Batch prediction from CSV file.

    Args:
        input_path: Path to input CSV
        output_path: Path to save predictions CSV
        model_name: Model to use
        model_dir: Model directory
        fe_path: Feature engineer path

    Returns:
        Predictions DataFrame
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load data
    raw_df = pd.read_csv(input_path)
    print(f"Loaded {len(raw_df)} rows from {input_path}")

    # Predict
    results = predict_from_raw(raw_df, model_name, model_dir, fe_path)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    return results


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Evaluate and print prediction metrics."""
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"  RMSE:  {metrics['rmse']:.2f}")
    print(f"  MAE:   {metrics['mae']:.2f}")
    print(f"  R²:    {metrics['r2']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    return metrics


def predict_price(
    year: int,
    km_driven: int,
    fuel: str,
    seller_type: str,
    transmission: str,
    owner: str,
    mileage: float,
    engine: int,
    max_power: float,
    seats: int,
    brand: str,
    model_name: str = "best_model",
    model_dir: Optional[Path] = None,
    fe_path: Optional[Path] = None,
) -> float:
    """
    Predict price for a single car (convenience function).

    Args:
        Car specifications
        model_name: Model to use
        model_dir: Model directory
        fe_path: Feature engineer path

    Returns:
        Predicted price
    """
    # Create single-row DataFrame
    raw_data = pd.DataFrame([{
        "name": f"{brand} Car",
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "mileage": f"{mileage} kmpl",
        "engine": f"{engine} CC",
        "max_power": f"{max_power} bhp",
        "seats": seats,
    }])

    result = predict_from_raw(raw_data, model_name, model_dir, fe_path)
    return float(result["predicted_price"].iloc[0])


if __name__ == "__main__":
    # Example usage
    print("Prediction module ready.")
    print("Use predict_from_raw() for DataFrames or batch_predict() for CSV files.")