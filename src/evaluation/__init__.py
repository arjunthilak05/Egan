"""Evaluation module for QGAN performance assessment."""

from .metrics import (
    mode_coverage_score,
    sample_diversity_score,
    pairwise_distance_diversity,
    mode_collapse_metric,
    inception_score,
    frechet_inception_distance,
    quality_diversity_tradeoff,
    entropy_diversity_correlation,
    MetricsTracker
)

from .visualization import (
    plot_training_curves,
    plot_2d_distribution,
    plot_mode_coverage_comparison,
    plot_entropy_distribution,
    plot_ablation_study,
    plot_image_grid,
    plot_entropy_diversity_correlation,
    create_comparison_table
)

__all__ = [
    # Metrics
    'mode_coverage_score',
    'sample_diversity_score',
    'pairwise_distance_diversity',
    'mode_collapse_metric',
    'inception_score',
    'frechet_inception_distance',
    'quality_diversity_tradeoff',
    'entropy_diversity_correlation',
    'MetricsTracker',
    # Visualization
    'plot_training_curves',
    'plot_2d_distribution',
    'plot_mode_coverage_comparison',
    'plot_entropy_distribution',
    'plot_ablation_study',
    'plot_image_grid',
    'plot_entropy_diversity_correlation',
    'create_comparison_table',
]
