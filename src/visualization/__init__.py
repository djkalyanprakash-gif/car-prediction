"""Visualization module."""

from src.visualization.visualize import (
    plot_target_distribution,
    plot_categorical_distributions,
    plot_numerical_distributions,
    plot_correlation_heatmap,
    plot_target_vs_features,
    plot_missing_values,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_residual_distribution,
    plot_feature_importance,
    plot_permutation_importance,
    plot_shap_summary,
    plot_learning_curves,
    generate_eda_report,
    generate_model_report,
    save_figure,
)

__all__ = [
    "plot_target_distribution",
    "plot_categorical_distributions",
    "plot_numerical_distributions",
    "plot_correlation_heatmap",
    "plot_target_vs_features",
    "plot_missing_values",
    "plot_model_comparison",
    "plot_predicted_vs_actual",
    "plot_residual_distribution",
    "plot_feature_importance",
    "plot_permutation_importance",
    "plot_shap_summary",
    "plot_learning_curves",
    "generate_eda_report",
    "generate_model_report",
    "save_figure",
]