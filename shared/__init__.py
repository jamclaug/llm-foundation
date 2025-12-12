"""Shared utilities for all models."""

# Core imports (always available)
from .config import Config, MambaConfig
from .dataset import TinyStoriesDataset, WikiTextDataset, PG19Dataset, get_dataset
from .trainer import Trainer, TrainerConfig, create_trainer
from .utils import count_parameters

# Evaluation and visualization (new)
from .evaluation import (
    EvaluationResult,
    compute_metrics,
    create_evaluation_result,
    compute_mse,
    compute_rmse,
    compute_mae,
    compute_r2,
    compute_pearson_r,
    compute_skill_score,
)
from .visualize import (
    get_backend,
    list_backends,
    VisualizationBackend,
    PlotextBackend,
    MatplotlibBackend,
)
from .diagnostics import (
    run_diagnostics,
    plot_sequence,
    plot_residuals,
    plot_xy_scatter,
    plot_error_distribution,
    plot_lag,
    plot_metric_comparison,
    plot_loss_curve,
    print_metrics_table,
)

# Optional imports (may have additional dependencies)
try:
    from .benchmark import benchmark
except ImportError:
    benchmark = None
