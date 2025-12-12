#!/usr/bin/env python3
"""
Diagnostic Plots for Model Evaluation.

Provides visualization functions that work with EvaluationResult objects
and the pluggable visualization backend system. Supports both terminal
(plotext) and file (matplotlib) output.

Plot Types
----------
**Regression/Time Series:**
- `plot_sequence`: Overlay actual vs predicted over time
- `plot_residuals`: Error over time (shows drift)
- `plot_xy_scatter`: Predicted vs actual (45° = perfect)
- `plot_error_distribution`: Histogram of residuals (should be normal)
- `plot_lag`: Prediction vs previous actual (detects lagging)

**LLM:**
- `plot_loss_curve`: Training/validation loss over time
- `plot_token_probabilities`: Distribution of predicted probabilities

**Generic:**
- `plot_metric_comparison`: Bar chart comparing metrics
- `plot_metric_history`: Line plot of metric over training

Usage Example
-------------
    from shared.evaluation import create_evaluation_result
    from shared.diagnostics import run_diagnostics, plot_sequence
    from shared.visualize import get_backend
    
    # Create evaluation result
    result = create_evaluation_result(predictions, targets, domain="timeseries")
    
    # Run all diagnostic plots (terminal)
    run_diagnostics(result)
    
    # Or individual plots with custom backend
    backend = get_backend("matplotlib")
    plot_sequence(result, backend)
    backend.save("output/sequence_plot.png")

Design Notes
------------
- All plot functions take `result: EvaluationResult` and `backend: VisualizationBackend`
- Functions handle their own axis labels, titles, etc.
- `run_diagnostics` is the main entry point for standard diagnostic suite
- Plots are designed to be informative in both terminal and graphical output
"""

from typing import Optional, List, Dict, Any, Literal
import numpy as np

from .evaluation import EvaluationResult
from .visualize import VisualizationBackend, get_backend


# =============================================================================
# REGRESSION / TIME SERIES PLOTS
# =============================================================================

def plot_sequence(
    result: EvaluationResult,
    backend: Optional[VisualizationBackend] = None,
    *,
    max_points: int = 200,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot actual and predicted values overlaid on same time axis.
    
    Good for visually assessing:
    - Overall fit quality
    - Phase alignment (is prediction leading/lagging?)
    - Amplitude matching
    - Pattern capture (does model follow trends/cycles?)
    
    Args:
        result: EvaluationResult with predictions and targets.
        backend: Visualization backend (auto-detected if None).
        max_points: Maximum points to plot (subsample if more).
        title: Custom title (default: "Sequence Plot: Actual vs Predicted").
        show: Whether to display the plot.
    """
    if backend is None:
        backend = get_backend()
    
    pred = result.predictions.flatten()
    targ = result.targets.flatten()
    
    # Subsample if too many points
    if len(pred) > max_points:
        step = len(pred) // max_points
        pred = pred[::step]
        targ = targ[::step]
    
    x = np.arange(len(pred))
    
    backend.clear()
    backend.plot_line(targ, x, label="Actual", color="blue")
    backend.plot_line(pred, x, label="Predicted", color="red")
    
    plot_title = title or "Sequence Plot: Actual vs Predicted"
    backend.set_title(plot_title)
    backend.set_labels(xlabel="Time Step", ylabel="Value")
    backend.legend()
    
    if show:
        backend.show()


def plot_residuals(
    result: EvaluationResult,
    backend: Optional[VisualizationBackend] = None,
    *,
    max_points: int = 200,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot residuals (prediction - actual) over time.
    
    Good for detecting:
    - **Drift**: Systematic trend in residuals over time
    - **Heteroscedasticity**: Changing variance over time
    - **Bias**: Residuals consistently above/below zero
    - **Outliers**: Large spikes in error
    
    Ideal: Residuals should be random noise around zero.
    
    Args:
        result: EvaluationResult with predictions and targets.
        backend: Visualization backend (auto-detected if None).
        max_points: Maximum points to plot.
        title: Custom title.
        show: Whether to display.
    """
    if backend is None:
        backend = get_backend()
    
    residuals = result.residuals.flatten()
    
    # Subsample if needed
    if len(residuals) > max_points:
        step = len(residuals) // max_points
        residuals = residuals[::step]
    
    x = np.arange(len(residuals))
    
    backend.clear()
    backend.plot_line(residuals, x, label="Residuals", color="purple")
    backend.plot_hline(0, color="gray", linestyle="--")
    
    plot_title = title or "Residual Plot: Error Over Time"
    backend.set_title(plot_title)
    backend.set_labels(xlabel="Time Step", ylabel="Residual (Pred - Actual)")
    
    if show:
        backend.show()


def plot_xy_scatter(
    result: EvaluationResult,
    backend: Optional[VisualizationBackend] = None,
    *,
    max_points: int = 1000,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Scatter plot of predicted vs actual values.
    
    Perfect predictions lie on the 45° diagonal (y=x).
    
    Good for detecting:
    - **Systematic bias**: Points shifted above/below diagonal
    - **Heteroscedasticity**: Spread changes with value magnitude
    - **Non-linearity**: Curved pattern instead of linear
    - **Outliers**: Points far from the diagonal
    
    Args:
        result: EvaluationResult with predictions and targets.
        backend: Visualization backend (auto-detected if None).
        max_points: Maximum points to plot.
        title: Custom title.
        show: Whether to display.
    """
    if backend is None:
        backend = get_backend()
    
    pred = result.predictions.flatten()
    targ = result.targets.flatten()
    
    # Subsample if needed
    if len(pred) > max_points:
        indices = np.random.choice(len(pred), max_points, replace=False)
        pred = pred[indices]
        targ = targ[indices]
    
    backend.clear()
    backend.plot_scatter(targ, pred, label="Predictions", color="blue")
    
    # Add 45° reference line
    min_val = min(targ.min(), pred.min())
    max_val = max(targ.max(), pred.max())
    backend.plot_line(
        [min_val, max_val], 
        [min_val, max_val], 
        label="Perfect (y=x)", 
        color="gray"
    )
    
    plot_title = title or f"XY Scatter: R² = {result.metrics.get('r2', 0):.4f}"
    backend.set_title(plot_title)
    backend.set_labels(xlabel="Actual", ylabel="Predicted")
    backend.legend()
    
    if show:
        backend.show()


def plot_error_distribution(
    result: EvaluationResult,
    backend: Optional[VisualizationBackend] = None,
    *,
    bins: int = 50,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Histogram of residuals (prediction errors).
    
    Ideal: Normal distribution centered at zero.
    
    Good for detecting:
    - **Bias**: Distribution not centered at zero
    - **Heavy tails**: More extreme errors than expected (kurtosis)
    - **Skewness**: Asymmetric distribution
    - **Multi-modality**: Multiple peaks (different error modes)
    
    Args:
        result: EvaluationResult with predictions and targets.
        backend: Visualization backend (auto-detected if None).
        bins: Number of histogram bins.
        title: Custom title.
        show: Whether to display.
    """
    if backend is None:
        backend = get_backend()
    
    residuals = result.residuals.flatten()
    
    backend.clear()
    backend.plot_histogram(residuals, bins=bins, color="green")
    backend.plot_hline(0, color="red", linestyle="--")  # Reference at 0
    
    # Compute stats for title
    mean_err = np.mean(residuals)
    std_err = np.std(residuals)
    
    plot_title = title or f"Error Distribution: μ={mean_err:.4f}, σ={std_err:.4f}"
    backend.set_title(plot_title)
    backend.set_labels(xlabel="Residual (Pred - Actual)", ylabel="Count")
    
    if show:
        backend.show()


def plot_lag(
    result: EvaluationResult,
    backend: Optional[VisualizationBackend] = None,
    *,
    lag: int = 1,
    max_points: int = 1000,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Lag plot: prediction[t] vs actual[t-lag].
    
    **Critical for detecting "lazy" models** that just predict the previous value.
    
    If predictions are highly correlated with lagged actuals (points on diagonal),
    the model may be "cheating" by copying the previous timestep instead of
    actually predicting future values.
    
    Good models should show:
    - For predictable data: Some correlation, but not on diagonal
    - For random walks: Predictions should NOT match lagged actuals perfectly
    
    Args:
        result: EvaluationResult with predictions and targets.
        backend: Visualization backend (auto-detected if None).
        lag: Number of timesteps to lag (default: 1).
        max_points: Maximum points to plot.
        title: Custom title.
        show: Whether to display.
    """
    if backend is None:
        backend = get_backend()
    
    pred = result.predictions.flatten()
    targ = result.targets.flatten()
    
    if len(pred) <= lag:
        print(f"Warning: Not enough points for lag={lag}")
        return
    
    # Align: prediction[t] vs actual[t-lag]
    pred_aligned = pred[lag:]
    targ_lagged = targ[:-lag]
    
    # Subsample if needed
    if len(pred_aligned) > max_points:
        indices = np.random.choice(len(pred_aligned), max_points, replace=False)
        pred_aligned = pred_aligned[indices]
        targ_lagged = targ_lagged[indices]
    
    # Compute lag correlation
    lag_corr = result.metrics.get("lag_correlation", np.corrcoef(pred_aligned, targ_lagged)[0, 1])
    
    backend.clear()
    backend.plot_scatter(targ_lagged, pred_aligned, label="Pred[t] vs Actual[t-1]", color="orange")
    
    # Add diagonal reference
    min_val = min(targ_lagged.min(), pred_aligned.min())
    max_val = max(targ_lagged.max(), pred_aligned.max())
    backend.plot_line(
        [min_val, max_val],
        [min_val, max_val],
        label="Lagging (y=x)",
        color="red"
    )
    
    plot_title = title or f"Lag Plot (lag={lag}): Correlation = {lag_corr:.4f}"
    backend.set_title(plot_title)
    backend.set_labels(xlabel=f"Actual[t-{lag}]", ylabel="Prediction[t]")
    backend.legend()
    
    if show:
        backend.show()


# =============================================================================
# GENERIC PLOTS
# =============================================================================

def plot_metric_comparison(
    metrics: Dict[str, float],
    backend: Optional[VisualizationBackend] = None,
    *,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Bar chart comparing different metrics.
    
    Useful for comparing model performance across metrics
    or comparing multiple models on same metric.
    
    Args:
        metrics: Dict of metric_name -> value.
        backend: Visualization backend.
        title: Custom title.
        show: Whether to display.
    """
    if backend is None:
        backend = get_backend()
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    backend.clear()
    backend.plot_bar(labels, values, color="steelblue")
    
    plot_title = title or "Metric Comparison"
    backend.set_title(plot_title)
    backend.set_labels(xlabel="Metric", ylabel="Value")
    
    if show:
        backend.show()


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    backend: Optional[VisualizationBackend] = None,
    *,
    title: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot training (and optionally validation) loss over time.
    
    Good for detecting:
    - **Overfitting**: Val loss increases while train loss decreases
    - **Underfitting**: Both losses plateau at high value
    - **Learning rate issues**: Erratic loss curves
    
    Args:
        train_losses: Training loss values.
        val_losses: Optional validation loss values.
        backend: Visualization backend.
        title: Custom title.
        show: Whether to display.
    """
    if backend is None:
        backend = get_backend()
    
    epochs = np.arange(1, len(train_losses) + 1)
    
    backend.clear()
    backend.plot_line(train_losses, epochs, label="Train Loss", color="blue")
    
    if val_losses is not None:
        val_epochs = np.arange(1, len(val_losses) + 1)
        backend.plot_line(val_losses, val_epochs, label="Val Loss", color="orange")
    
    plot_title = title or "Loss Curve"
    backend.set_title(plot_title)
    backend.set_labels(xlabel="Epoch", ylabel="Loss")
    backend.legend()
    
    if show:
        backend.show()


# =============================================================================
# DIAGNOSTIC SUITE
# =============================================================================

def run_diagnostics(
    result: EvaluationResult,
    backend: Optional[VisualizationBackend] = None,
    *,
    plots: Optional[List[str]] = None,
    interactive: bool = True,
    save_dir: Optional[str] = None,
) -> None:
    """
    Run standard diagnostic plots for an evaluation result.
    
    Automatically selects appropriate plots based on the domain.
    
    Args:
        result: EvaluationResult to visualize.
        backend: Visualization backend (auto-detected if None).
        plots: List of specific plots to run. If None, runs domain-appropriate suite.
            Options: "sequence", "residuals", "scatter", "distribution", "lag", "metrics"
        interactive: If True, show each plot with pause. If False, display sequentially.
        save_dir: If provided, save plots to this directory instead of displaying.
    
    Example:
        >>> result = create_evaluation_result(preds, targets, domain="timeseries")
        >>> run_diagnostics(result)  # Shows all diagnostic plots
        >>> run_diagnostics(result, plots=["scatter", "lag"])  # Specific plots
        >>> run_diagnostics(result, save_dir="output/plots/")  # Save to files
    """
    if backend is None:
        backend = get_backend()
    
    # Determine which plots to run
    domain = result.domain
    
    if plots is None:
        if domain in ("regression", "timeseries"):
            plots = ["sequence", "residuals", "scatter", "distribution", "lag", "metrics"]
        elif domain == "llm":
            plots = ["metrics"]  # LLM has different plots
        elif domain == "classification":
            plots = ["metrics"]
        else:
            plots = ["metrics"]
    
    # Print summary first
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(result.summary())
    print("=" * 60 + "\n")
    
    # Plot functions mapping
    plot_funcs = {
        "sequence": lambda: plot_sequence(result, backend, show=False),
        "residuals": lambda: plot_residuals(result, backend, show=False),
        "scatter": lambda: plot_xy_scatter(result, backend, show=False),
        "distribution": lambda: plot_error_distribution(result, backend, show=False),
        "lag": lambda: plot_lag(result, backend, show=False),
        "metrics": lambda: plot_metric_comparison(result.metrics, backend, show=False),
    }
    
    for i, plot_name in enumerate(plots, 1):
        if plot_name not in plot_funcs:
            print(f"  ⚠ Unknown plot: {plot_name}")
            continue
        
        print(f"[{i}/{len(plots)}] {plot_name.title()} Plot")
        
        # Generate plot
        plot_funcs[plot_name]()
        
        if save_dir:
            # Save to file
            from pathlib import Path
            save_path = Path(save_dir) / f"{plot_name}_plot.png"
            backend.save(str(save_path))
            backend.clear()
        else:
            # Display
            backend.show()
            
            if interactive and i < len(plots):
                try:
                    input("  Press Enter for next plot...")
                except EOFError:
                    pass  # Non-interactive environment
    
    print("\n✓ Diagnostic plots complete.")


def print_metrics_table(result: EvaluationResult) -> None:
    """
    Print metrics in a formatted table.
    
    Args:
        result: EvaluationResult with computed metrics.
    """
    print("\n" + "=" * 50)
    print(f"{'Metric':<25} {'Value':>20}")
    print("=" * 50)
    
    for name, value in sorted(result.metrics.items()):
        if isinstance(value, float):
            print(f"{name:<25} {value:>20.6f}")
        else:
            print(f"{name:<25} {str(value):>20}")
    
    print("=" * 50)
