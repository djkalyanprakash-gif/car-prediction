#!/usr/bin/env python
"""
Entry point for training pipeline.
Run: python train.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.preprocessing import run_preprocessing_pipeline
from src.features.build_features import build_features
from src.models.train import train_all_models, select_best_model, train_with_optimization, save_model
from src.models.predict import evaluate_predictions


def main():
    print("=" * 60)
    print("CAR PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Preprocess
    print("\n[1/4] Preprocessing data...")
    train_df, test_df = run_preprocessing_pipeline()

    # 2. Feature engineering
    print("\n[2/4] Building features...")
    X_train, X_test, y_train, y_test = build_features(train_df, test_df)

    # 3. Train all models
    print("\n[3/4] Training models...")
    results = train_all_models(X_train, y_train, X_test, y_test)
    best_name, best_model = select_best_model(results)

    # 4. Optimize best model
    print(f"\n[4/4] Optimizing {best_name}...")
    optimized_model, opt_info = train_with_optimization(best_name, X_train, y_train, X_test, y_test)

    # Use optimized if better
    default_rmse = results[best_name]['val_metrics']['rmse']
    opt_rmse = opt_info.get('val_metrics', {}).get('rmse', float('inf'))

    if opt_rmse < default_rmse:
        best_model = optimized_model
        print(f"Using optimized model (RMSE: {opt_rmse:.0f} vs {default_rmse:.0f})")
    else:
        print(f"Keeping default model (RMSE: {default_rmse:.0f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    y_pred = best_model.predict(X_test)
    evaluate_predictions(y_test.values, y_pred, f"{best_name} (final)")

    # Save
    save_model(best_model, "best_model")
    print("\n✅ Training complete! Model saved to models/best_model.joblib")


if __name__ == "__main__":
    main()