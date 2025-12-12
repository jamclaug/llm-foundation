#!/usr/bin/env python3
"""
Unified Evaluation Module for All Models.

Provides metrics computation and evaluation harness that works across
different model types (LLM, time series, classification). Designed to
integrate with the visualization module for diagnostic plots.

Supported Domains
-----------------
- **regression/timeseries**: MSE, MAE, RMSE, R², skill score, lag correlation
- **llm**: Perplexity, token accuracy, BLEU (optional)
- **classification**: Accuracy, precision, recall, F1

Usage Example
-------------
    from shared.evaluation import compute_metrics, EvaluationResult
    
    # Compute metrics for predictions
    metrics = compute_metrics(
        predictions=model_output,
        targets=ground_truth,
        domain="regression",
    )
    print(metrics)  # {'mse': 0.05, 'rmse': 0.22, 'r2': 0.87, ...}
    
    # Full evaluation with result object
    result = EvaluationResult(
        predictions=predictions,
        targets=targets,
        metrics=metrics,
        domain="timeseries",
    )
    
    # Use with diagnostics
    from shared.diagnostics import run_diagnostics
    run_diagnostics(result)

Design Philosophy
-----------------
- **Domain-agnostic core**: Basic metrics work for any prediction task
- **Domain-specific extras**: Additional metrics for LLM, time series, etc.
- **Lightweight**: Minimal dependencies (numpy, optional torch)
- **Composable**: Metrics can be computed individually or as a bundle
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Any
import math
import warnings

import numpy as np

# Optional torch import
try:
    import torch
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Tensor = Any  # Type hint fallback

# Type alias
ArrayLike = Union[List[float], np.ndarray, "Tensor"]
Domain = Literal["regression", "timeseries", "llm", "classification"]


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert array-like data to numpy array."""
    if data is None:
        return None
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif hasattr(data, "numpy"):
        return data.numpy()
    else:
        return np.asarray(data)


# =============================================================================
# EVALUATION RESULT
# =============================================================================

@dataclass
class EvaluationResult:
    """
    Container for evaluation results.
    
    Holds predictions, targets, and computed metrics in a single object
    that can be passed to visualization/diagnostic functions.
    
    Attributes:
        predictions: Model predictions (flattened or structured).
        targets: Ground truth values.
        metrics: Dict of computed metrics.
        domain: Type of task ("regression", "timeseries", "llm", "classification").
        metadata: Additional info (model name, dataset, etc.).
    
    Properties:
        residuals: Predictions - Targets (for regression/timeseries).
        num_samples: Number of prediction samples.
    """
    predictions: np.ndarray
    targets: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)
    domain: Domain = "regression"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert to numpy and validate."""
        self.predictions = _to_numpy(self.predictions)
        self.targets = _to_numpy(self.targets)
        
        if self.predictions.shape != self.targets.shape:
            warnings.warn(
                f"Shape mismatch: predictions {self.predictions.shape} vs "
                f"targets {self.targets.shape}. Will attempt to align."
            )
    
    @property
    def residuals(self) -> np.ndarray:
        """Compute residuals (predictions - targets)."""
        return self.predictions - self.targets
    
    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return self.predictions.size
    
    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            f"Evaluation Results ({self.domain})",
            "=" * 40,
            f"Samples: {self.num_samples:,}",
        ]
        
        for name, value in self.metrics.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.6f}")
            else:
                lines.append(f"  {name}: {value}")
        
        if self.metadata:
            lines.append("-" * 40)
            for key, val in self.metadata.items():
                lines.append(f"  {key}: {val}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"EvaluationResult(domain={self.domain!r}, n={self.num_samples}, metrics={list(self.metrics.keys())})"


# =============================================================================
# CORE METRICS (Domain-Agnostic)
# =============================================================================

def compute_mse(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    Mean Squared Error.
    
    MSE = (1/n) * Σ(pred - target)²
    
    Lower is better. Penalizes large errors more than small ones.
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    return float(np.mean((pred - targ) ** 2))


def compute_rmse(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    Root Mean Squared Error.
    
    RMSE = √MSE
    
    In same units as the data. More interpretable than MSE.
    """
    return math.sqrt(compute_mse(predictions, targets))


def compute_mae(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    Mean Absolute Error.
    
    MAE = (1/n) * Σ|pred - target|
    
    Less sensitive to outliers than MSE. In same units as data.
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    return float(np.mean(np.abs(pred - targ)))


def compute_r2(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    R-squared (Coefficient of Determination).
    
    R² = 1 - SS_res / SS_tot
       = 1 - Σ(target - pred)² / Σ(target - mean(target))²
    
    Range: (-∞, 1]
    - R² = 1: Perfect predictions
    - R² = 0: Model predicts the mean (no skill)
    - R² < 0: Model worse than predicting the mean
    
    Note: This is the same as Pearson correlation squared when predictions
    are unbiased, but differs when there's systematic bias.
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    
    ss_res = np.sum((targ - pred) ** 2)
    ss_tot = np.sum((targ - np.mean(targ)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return float(1 - ss_res / ss_tot)


def compute_pearson_r(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    Pearson Correlation Coefficient.
    
    r = Cov(pred, target) / (std(pred) * std(target))
    
    Range: [-1, 1]
    - r = 1: Perfect positive correlation
    - r = 0: No linear correlation
    - r = -1: Perfect negative correlation
    
    Note: High correlation doesn't mean accurate predictions!
    A model that predicts 2*target has r=1 but is wrong.
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    
    # Use numpy's corrcoef
    corr_matrix = np.corrcoef(pred, targ)
    return float(corr_matrix[0, 1])


def compute_skill_score(
    predictions: ArrayLike,
    targets: ArrayLike,
    baseline_predictions: Optional[ArrayLike] = None,
) -> float:
    """
    Skill Score (relative improvement over baseline).
    
    skill = 1 - MSE_model / MSE_baseline
    
    If baseline not provided, uses "naive" baseline (predict previous value).
    
    Range: (-∞, 1]
    - skill > 0: Better than baseline
    - skill = 0: Same as baseline
    - skill < 0: Worse than baseline
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    
    model_mse = compute_mse(pred, targ)
    
    if baseline_predictions is not None:
        baseline = _to_numpy(baseline_predictions).flatten()
    else:
        # Naive baseline: predict previous value (shift targets by 1)
        # This assumes time-aligned predictions
        baseline = np.concatenate([[targ[0]], targ[:-1]])
    
    baseline_mse = compute_mse(baseline, targ)
    
    if baseline_mse == 0:
        return 1.0 if model_mse == 0 else -float('inf')
    
    return float(1 - model_mse / baseline_mse)


# =============================================================================
# TIME SERIES SPECIFIC METRICS
# =============================================================================

def compute_lag_correlation(predictions: ArrayLike, targets: ArrayLike, lag: int = 1) -> float:
    """
    Correlation between predictions and lagged targets.
    
    Measures if model is "cheating" by predicting the previous value.
    
    lag_corr = corr(predictions[lag:], targets[:-lag])
    
    High lag correlation (close to 1) with low prediction accuracy suggests
    the model has learned to simply copy the previous value.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth.
        lag: Number of steps to lag (default: 1).
    
    Returns:
        Pearson correlation between predictions and lagged targets.
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    
    if len(pred) <= lag:
        return 0.0
    
    # Correlate predictions with lagged targets
    # If pred[t] ≈ target[t-lag], model is just copying with lag
    pred_aligned = pred[lag:]
    targ_lagged = targ[:-lag]
    
    corr_matrix = np.corrcoef(pred_aligned, targ_lagged)
    return float(corr_matrix[0, 1])


def compute_directional_accuracy(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    Fraction of correct direction predictions.
    
    Measures if the model correctly predicts whether the next value
    will go up or down. Important for trading/control applications.
    
    dir_acc = mean(sign(Δpred) == sign(Δtarget))
    
    Returns:
        Fraction in [0, 1], where 0.5 is random chance.
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    
    if len(pred) < 2:
        return 0.5
    
    pred_direction = np.sign(np.diff(pred))
    targ_direction = np.sign(np.diff(targ))
    
    # Handle zeros (no change)
    mask = (pred_direction != 0) & (targ_direction != 0)
    if not np.any(mask):
        return 0.5
    
    return float(np.mean(pred_direction[mask] == targ_direction[mask]))


def compute_mape(predictions: ArrayLike, targets: ArrayLike, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    MAPE = (100/n) * Σ|target - pred| / |target|
    
    Useful when relative error matters more than absolute error.
    Warning: Undefined/infinite when target ≈ 0.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth.
        epsilon: Small constant to avoid division by zero.
    
    Returns:
        MAPE as a percentage (0-100+).
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    
    # Avoid division by zero
    denom = np.maximum(np.abs(targ), epsilon)
    
    return float(100 * np.mean(np.abs(pred - targ) / denom))


# =============================================================================
# LLM SPECIFIC METRICS
# =============================================================================

def compute_perplexity(log_probs: ArrayLike) -> float:
    """
    Perplexity from log probabilities.
    
    PPL = exp(-mean(log_probs))
    
    Lower is better. Measures how "surprised" the model is by the data.
    
    Args:
        log_probs: Log probabilities of correct tokens.
    
    Returns:
        Perplexity (positive float, typically 1-1000+).
    """
    lp = _to_numpy(log_probs).flatten()
    return float(np.exp(-np.mean(lp)))


def compute_perplexity_from_loss(loss: float) -> float:
    """
    Perplexity from cross-entropy loss.
    
    PPL = exp(loss)
    
    Convenient when you have the loss value directly.
    """
    return float(np.exp(loss))


def compute_token_accuracy(
    predicted_tokens: ArrayLike,
    target_tokens: ArrayLike,
    ignore_index: int = -100,
) -> float:
    """
    Token-level accuracy.
    
    acc = mean(pred == target) for non-ignored positions
    
    Args:
        predicted_tokens: Model's predicted token IDs.
        target_tokens: Ground truth token IDs.
        ignore_index: Token ID to ignore (e.g., padding).
    
    Returns:
        Accuracy in [0, 1].
    """
    pred = _to_numpy(predicted_tokens).flatten()
    targ = _to_numpy(target_tokens).flatten()
    
    mask = targ != ignore_index
    if not np.any(mask):
        return 0.0
    
    return float(np.mean(pred[mask] == targ[mask]))


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def compute_accuracy(predictions: ArrayLike, targets: ArrayLike) -> float:
    """
    Classification accuracy.
    
    acc = mean(pred == target)
    """
    pred = _to_numpy(predictions).flatten()
    targ = _to_numpy(targets).flatten()
    return float(np.mean(pred == targ))


def compute_confusion_matrix(
    predictions: ArrayLike,
    targets: ArrayLike,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns:
        2D array where C[i,j] = count of (target=i, pred=j).
    """
    pred = _to_numpy(predictions).flatten().astype(int)
    targ = _to_numpy(targets).flatten().astype(int)
    
    if num_classes is None:
        num_classes = max(pred.max(), targ.max()) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targ, pred):
        cm[t, p] += 1
    
    return cm


# =============================================================================
# BUNDLED METRIC COMPUTATION
# =============================================================================

def compute_metrics(
    predictions: ArrayLike,
    targets: ArrayLike,
    domain: Domain = "regression",
    baseline: Optional[ArrayLike] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Compute all relevant metrics for a domain.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth values.
        domain: Type of task ("regression", "timeseries", "llm", "classification").
        baseline: Optional baseline predictions for skill score.
        **kwargs: Additional domain-specific arguments.
    
    Returns:
        Dict of metric_name -> value.
    
    Example:
        >>> metrics = compute_metrics(preds, targets, domain="timeseries")
        >>> print(metrics)
        {'mse': 0.05, 'rmse': 0.22, 'mae': 0.18, 'r2': 0.87, 'skill_score': 0.45, ...}
    """
    pred = _to_numpy(predictions)
    targ = _to_numpy(targets)
    
    metrics = {}
    
    if domain in ("regression", "timeseries"):
        # Core regression metrics
        metrics["mse"] = compute_mse(pred, targ)
        metrics["rmse"] = compute_rmse(pred, targ)
        metrics["mae"] = compute_mae(pred, targ)
        metrics["r2"] = compute_r2(pred, targ)
        metrics["pearson_r"] = compute_pearson_r(pred, targ)
        metrics["skill_score"] = compute_skill_score(pred, targ, baseline)
        
        if domain == "timeseries":
            # Time series specific
            metrics["lag_correlation"] = compute_lag_correlation(pred, targ)
            metrics["directional_accuracy"] = compute_directional_accuracy(pred, targ)
            
            # MAPE only if data doesn't cross zero (avoids division issues)
            # Check: all values same sign AND far from zero
            min_abs = np.min(np.abs(targ))
            max_abs = np.max(np.abs(targ))
            same_sign = (np.all(targ > 0) or np.all(targ < 0))
            far_from_zero = min_abs > 0.1 * max_abs  # min is at least 10% of max
            
            if same_sign and far_from_zero:
                metrics["mape"] = compute_mape(pred, targ)
    
    elif domain == "llm":
        # LLM metrics - expect different input format
        if "log_probs" in kwargs:
            metrics["perplexity"] = compute_perplexity(kwargs["log_probs"])
        if "loss" in kwargs:
            metrics["perplexity_from_loss"] = compute_perplexity_from_loss(kwargs["loss"])
        
        # Token accuracy if integer predictions
        if pred.dtype in (np.int32, np.int64):
            metrics["token_accuracy"] = compute_token_accuracy(pred, targ)
    
    elif domain == "classification":
        metrics["accuracy"] = compute_accuracy(pred, targ)
        # Could add precision, recall, F1 here
    
    return metrics


def create_evaluation_result(
    predictions: ArrayLike,
    targets: ArrayLike,
    domain: Domain = "regression",
    metadata: Optional[Dict[str, Any]] = None,
    **metric_kwargs,
) -> EvaluationResult:
    """
    Create an EvaluationResult with computed metrics.
    
    Convenience function that computes metrics and packages everything.
    
    Args:
        predictions: Model predictions.
        targets: Ground truth.
        domain: Task domain.
        metadata: Additional info to store.
        **metric_kwargs: Passed to compute_metrics.
    
    Returns:
        EvaluationResult ready for visualization.
    """
    metrics = compute_metrics(predictions, targets, domain=domain, **metric_kwargs)
    
    return EvaluationResult(
        predictions=predictions,
        targets=targets,
        metrics=metrics,
        domain=domain,
        metadata=metadata or {},
    )
