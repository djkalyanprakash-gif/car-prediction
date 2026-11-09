"""
Visualization module for Car Price Prediction.
Creates publication-quality plots for EDA, model evaluation, and reporting.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import seaborn as sns
import warnings

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.config import get_config


config = get_config()

# Style setup
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Color palette
COLORS = {
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "accent": "#3498DB",
    "success": "#27AE60",
    "warning": "#F39C12",
    "background": "#ECF0F1",
    "text": "#2C3E50",
}

def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    """Save figure to reports/figures directory."""
    figures_dir = Path(config.get("output.figures_dir", "reports/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved figure: {path}")


# ============================================================
# EDA Visualizations
# ============================================================

def plot_target_distribution(
    y: pd.Series,
    title: str = "Target Distribution (Selling Price)",
    save: bool = True,
) -> plt.Figure:
    """Plot distribution of target variable."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(y, bins=50, edgecolor="white", alpha=0.8, color=COLORS["primary"])
    axes[0].set_xlabel("Selling Price (INR)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram")

    # Box plot
    axes[1].boxplot(y, vert=True, patch_artist=True,
                    boxprops=dict(facecolor=COLORS["accent"], alpha=0.7))
    axes[1].set_ylabel("Selling Price (INR)")
    axes[1].set_title("Box Plot")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_figure(fig, "target_distribution")
    return fig


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: List[str],
    save: bool = True,
) -> plt.Figure:
    """Plot distributions of categorical features."""
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        value_counts = df[col].value_counts()
        value_counts.plot(kind="bar", ax=ax, color=COLORS["primary"], edgecolor="white", alpha=0.8)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_figure(fig, "categorical_distributions")
    return fig


def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: List[str],
    save: bool = True,
) -> plt.Figure:
    """Plot distributions of numerical features."""
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        df[col].hist(bins=30, ax=ax, edgecolor="white", alpha=0.8, color=COLORS["accent"])
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_figure(fig, "numerical_distributions")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str = "selling_price",
    save: bool = True,
) -> plt.Figure:
    """Plot correlation heatmap with target."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Feature Correlation Heatmap (Target: {target_col})", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_figure(fig, "correlation_heatmap")
    return fig


def plot_target_vs_features(
    df: pd.DataFrame,
    target_col: str = "selling_price",
    features: Optional[List[str]] = None,
    save: bool = True,
) -> plt.Figure:
    """Plot target vs numerical features (scatter plots)."""
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in features if f != target_col][:6]  # Limit to 6

    n_cols = min(3, len(features))
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if len(features) > 1 else [axes]

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.scatter(df[feat], df[target_col], alpha=0.4, s=10, color=COLORS["primary"])
        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("Selling Price (INR)")
        ax.set_title(f"Price vs {feat.replace('_', ' ').title()}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Target vs Numerical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_figure(fig, "target_vs_features")
    return fig


def plot_missing_values(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot missing values heatmap and bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap
    ax = axes[0]
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap")

    # Bar chart
    ax = axes[1]
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing.plot(kind="bar", ax=ax, color=COLORS["secondary"], edgecolor="white", alpha=0.8)
        ax.set_title("Missing Values Count")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Missing Values Count")

    plt.tight_layout()

    if save:
        save_figure(fig, "missing_values")
    return fig


# ============================================================
# Model Evaluation Visualizations
# ============================================================

def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = "rmse_mean",
    save: bool = True,
) -> plt.Figure:
    """Compare models on CV metric."""
    model_names = list(results.keys())
    means = [results[m]["cv_results"][metric] for m in model_names]
    stds = [results[m]["cv_results"][metric.replace("mean", "std")] for m in model_names]

    # Sort by metric
    sorted_idx = np.argsort(means)
    model_names = [model_names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(model_names, means, xerr=stds, capsize=5,
                   color=COLORS["primary"], edgecolor="white", alpha=0.8)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Best at top

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_width() + std + 0.01 * max(means), bar.get_y() + bar.get_height()/2,
                f"{mean:.2f} ± {std:.2f}", va="center", fontsize=10)

    plt.tight_layout()

    if save:
        save_figure(fig, f"model_comparison_{metric}")
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Plot predicted vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10, color=COLORS["primary"], label="Predictions")
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (INR)")
    ax.set_ylabel("Predicted Price (INR)")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.legend()

    # Residual plot
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=10, color=COLORS["secondary"])
    ax.axhline(y=0, color="red", linestyle="--", lw=2)
    ax.set_xlabel("Predicted Price (INR)")
    ax.set_ylabel("Residual (INR)")
    ax.set_title(f"{model_name}: Residuals")

    plt.tight_layout()

    if save:
        save_figure(fig, f"predicted_vs_actual_{model_name.lower().replace(' ', '_')}")
    return fig


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Plot residual distribution."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor="white", alpha=0.8, color=COLORS["secondary"])
    axes[0].axvline(x=0, color="red", linestyle="--", lw=2)
    axes[0].set_xlabel("Residual (INR)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Residual Distribution")

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normality Check)")

    fig.suptitle(f"{model_name}: Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        save_figure(fig, f"residuals_{model_name.lower().replace(' ', '_')}")
    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    save: bool = True,
) -> plt.Figure:
    """Plot feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        print(f"Model {model_name} doesn't have feature_importances_")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices], color=COLORS["primary"], edgecolor="white", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"{model_name}: Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        save_figure(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}")
    return fig


def plot_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    top_n: int = 20,
    n_repeats: int = 10,
    save: bool = True,
) -> plt.Figure:
    """Plot permutation importance."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    std = result.importances_std
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[indices], xerr=std[indices], capsize=3,
            color=COLORS["accent"], edgecolor="white", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([X.columns[i] for i in indices])
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"{model_name}: Top {top_n} Permutation Importances", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        save_figure(fig, f"permutation_importance_{model_name.lower().replace(' ', '_')}")
    return fig


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    model_name: str = "Model",
    save: bool = True,
) -> Optional[plt.Figure]:
    """Plot SHAP summary for tree-based models."""
    if not HAS_SHAP:
        print("SHAP not installed. Install with: pip install shap")
        return None

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        ax.set_title(f"{model_name}: SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            save_figure(fig, f"shap_summary_{model_name.lower().replace(' ', '_')}")
        return fig
    except Exception as e:
        print(f"SHAP plotting failed: {e}")
        return None


def plot_learning_curves(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Plot learning curves."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, "o-", color=COLORS["primary"], label="Training Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color=COLORS["primary"])
    ax.plot(train_sizes, val_mean, "o-", color=COLORS["secondary"], label="Validation Score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color=COLORS["secondary"])

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score (Negative RMSE)")
    ax.set_title(f"{model_name}: Learning Curves", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        save_figure(fig, f"learning_curves_{model_name.lower().replace(' ', '_')}")
    return fig


# ============================================================
# Report Generation
# ============================================================

def generate_eda_report(
    df: pd.DataFrame,
    target_col: str = "selling_price",
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """Generate complete EDA report with all plots."""
    print("Generating EDA report...")

    figures = {}

    # Missing values
    figures["missing"] = plot_missing_values(df)

    # Target distribution
    figures["target_dist"] = plot_target_distribution(df[target_col])

    # Categorical
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        figures["categorical"] = plot_categorical_distributions(df, cat_cols)

    # Numerical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != target_col]
    if num_cols:
        figures["numerical"] = plot_numerical_distributions(df, num_cols)

    # Correlations
    figures["correlation"] = plot_correlation_heatmap(df, target_col)

    # Target vs features
    figures["target_vs_features"] = plot_target_vs_features(df, target_col)

    print("EDA report complete.")
    return figures


def generate_model_report(
    results: Dict[str, Dict[str, Any]],
    best_model_name: str,
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """Generate complete model evaluation report."""
    print("Generating model evaluation report...")

    figures = {}

    # Model comparison
    figures["comparison_rmse"] = plot_model_comparison(results, "rmse_mean")
    figures["comparison_r2"] = plot_model_comparison(results, "r2_mean")

    # Best model detailed analysis
    y_pred = best_model.predict(X_test)

    figures["pred_vs_actual"] = plot_predicted_vs_actual(y_test.values, y_pred, best_model_name)
    figures["residuals"] = plot_residual_distribution(y_test.values, y_pred, best_model_name)

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        figures["feature_importance"] = plot_feature_importance(best_model, feature_names, best_model_name)

    # Permutation importance
    if config.get("evaluation.permutation_importance", True):
        figures["perm_importance"] = plot_permutation_importance(best_model, X_test, y_test, best_model_name)

    # SHAP
    if config.get("evaluation.shap_analysis", True) and HAS_SHAP:
        figures["shap"] = plot_shap_summary(best_model, X_test, best_model_name)

    print("Model report complete.")
    return figures


if __name__ == "__main__":
    print("Visualization module ready. Import functions to create plots.")