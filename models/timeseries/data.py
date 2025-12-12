"""
Synthetic Time Series Data Generation.

This module provides utilities for generating synthetic time series data
for testing and training the TimeSeriesTransformer model. Each generator
produces data with known, controllable properties.

Why Synthetic Data?
-------------------
1. **Ground Truth**: We know exactly what patterns exist, making evaluation 
   straightforward.
2. **Controlled Complexity**: Start simple (sine waves), increase complexity 
   gradually (multi-scale, regime changes).
3. **Unlimited Supply**: Generate as much training data as needed.
4. **Debugging**: If the model can't learn simple patterns, there's a bug.

Available Generators
--------------------
1. **Sine Waves** (`generate_sine_waves`):
   - Predictable periodic patterns
   - Test: Can model learn periodicity?
   
2. **Random Walk** (`generate_random_walk`):
   - Brownian motion (cumulative random steps)
   - Test: Can model learn momentum/drift?
   
3. **AR Process** (`generate_ar_process`):
   - Autoregressive: x[t] depends on x[t-1], x[t-2], ...
   - Test: Can model learn temporal dependencies?

4. **Trend + Seasonal** (`generate_trend_seasonal`):
   - Linear trend + periodic component + noise
   - Test: Can model decompose signal components?

5. **Regime Switching** (`generate_regime_switching`):
   - Alternates between different behaviors
   - Test: Can model detect regime changes?

Each generator returns (values, metadata) where:
- values: Tensor of shape (num_samples, time_steps, num_streams)
- metadata: Dict with generation parameters for reproducibility

Usage Example
-------------
    from models.timeseries.data import (
        generate_sine_waves,
        TimeSeriesDataset,
    )
    
    # Generate synthetic data
    values, meta = generate_sine_waves(
        num_samples=1000,
        time_steps=200,
        num_streams=10,
    )
    
    # Create dataset for training
    dataset = TimeSeriesDataset(values, window_size=5)
    dataloader = DataLoader(dataset, batch_size=32)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Literal
from pathlib import Path
import gzip
import hashlib
import io
import json
import math
import sys

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
def _progress_bar(iterable, total=None, desc=None, disable=False):
    """Create a progress bar, falling back to simple prints if tqdm unavailable."""
    if HAS_TQDM and not disable:
        return tqdm(iterable, total=total, desc=desc, file=sys.stdout)
    else:
        # Simple fallback - just return iterable with occasional prints
        return iterable


# =============================================================================
# DATA GENERATORS
# =============================================================================

def generate_sine_waves(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    frequency_range: Tuple[float, float] = (0.01, 0.1),
    amplitude_range: Tuple[float, float] = (0.5, 2.0),
    phase_range: Tuple[float, float] = (0.0, 2 * math.pi),
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate multivariate sine wave time series.
    
    Each stream is a sine wave with random frequency, amplitude, and phase.
    Good for testing if the model can learn periodic patterns.
    
    Formula: x[t] = amplitude * sin(2π * frequency * t + phase) + noise
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        frequency_range: (min, max) frequency in cycles per time step.
            0.1 means 10 time steps per cycle.
        amplitude_range: (min, max) amplitude of the sine waves.
        phase_range: (min, max) phase offset in radians.
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters.
    
    Example:
        >>> values, meta = generate_sine_waves(100, 200, 5, noise_std=0.05)
        >>> values.shape  # (100, 200, 5)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random parameters for each (sample, stream)
    freq_min, freq_max = frequency_range
    amp_min, amp_max = amplitude_range
    phase_min, phase_max = phase_range
    
    frequencies = torch.rand(num_samples, num_streams) * (freq_max - freq_min) + freq_min
    amplitudes = torch.rand(num_samples, num_streams) * (amp_max - amp_min) + amp_min
    phases = torch.rand(num_samples, num_streams) * (phase_max - phase_min) + phase_min
    
    # Generate time axis: (1, time_steps, 1)
    t = torch.arange(time_steps, dtype=torch.float32).view(1, time_steps, 1)
    
    # Broadcast: (num_samples, 1, num_streams)
    freq = frequencies.unsqueeze(1)
    amp = amplitudes.unsqueeze(1)
    phase = phases.unsqueeze(1)
    
    # Compute sine waves: (num_samples, time_steps, num_streams)
    values = amp * torch.sin(2 * math.pi * freq * t + phase)
    
    # Add noise
    if noise_std > 0:
        noise = torch.randn_like(values) * noise_std
        values = values + noise
    
    metadata = {
        "generator": "sine_waves",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "frequency_range": frequency_range,
        "amplitude_range": amplitude_range,
        "phase_range": phase_range,
        "noise_std": noise_std,
        "seed": seed,
    }
    
    return values, metadata


def generate_random_walk(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    step_std: float = 0.1,
    drift: float = 0.0,
    initial_range: Tuple[float, float] = (-1.0, 1.0),
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate random walk (Brownian motion) time series.
    
    Each time step: x[t] = x[t-1] + drift + N(0, step_std)
    
    Random walks are the "hardest" baseline for forecasting because
    the best predictor is simply the previous value. If your model
    can't beat random walk, it's not learning anything useful.
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        step_std: Standard deviation of random steps (volatility).
        drift: Constant drift per time step (trend).
        initial_range: (min, max) for initial values.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters.
    
    Example:
        >>> values, meta = generate_random_walk(100, 200, 5, step_std=0.05)
        >>> values.shape  # (100, 200, 5)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Initial values
    init_min, init_max = initial_range
    initial = torch.rand(num_samples, 1, num_streams) * (init_max - init_min) + init_min
    
    # Generate random steps
    steps = torch.randn(num_samples, time_steps - 1, num_streams) * step_std + drift
    
    # Cumulative sum to get random walk
    cumsum = torch.cumsum(steps, dim=1)
    
    # Combine initial value with cumulative steps
    values = torch.cat([initial, initial + cumsum], dim=1)
    
    metadata = {
        "generator": "random_walk",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "step_std": step_std,
        "drift": drift,
        "initial_range": initial_range,
        "seed": seed,
    }
    
    return values, metadata


def generate_ar_process(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    ar_coefficients: Optional[List[float]] = None,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate autoregressive AR(p) time series.
    
    AR(p): x[t] = c1*x[t-1] + c2*x[t-2] + ... + cp*x[t-p] + noise
    
    This tests if the model can learn linear temporal dependencies.
    The coefficients determine the "memory" of the process.
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        ar_coefficients: List of AR coefficients [c1, c2, ...].
            Default: [0.7, -0.2] (AR(2) with some mean reversion).
            Coefficients should sum to < 1 for stability.
        noise_std: Standard deviation of innovation noise.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters.
    
    Example:
        >>> # AR(3) process
        >>> values, meta = generate_ar_process(
        ...     100, 200, 5, ar_coefficients=[0.5, 0.3, -0.1]
        ... )
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if ar_coefficients is None:
        ar_coefficients = [0.7, -0.2]  # AR(2) with mean reversion
    
    ar_order = len(ar_coefficients)
    coeffs = torch.tensor(ar_coefficients, dtype=torch.float32)
    
    # Initialize with small random values
    values = torch.zeros(num_samples, time_steps, num_streams)
    values[:, :ar_order, :] = torch.randn(num_samples, ar_order, num_streams) * 0.1
    
    # Generate AR process
    noise = torch.randn(num_samples, time_steps, num_streams) * noise_std
    
    for t in range(ar_order, time_steps):
        # x[t] = sum(c[i] * x[t-i-1]) + noise
        ar_term = sum(
            coeffs[i] * values[:, t - i - 1, :]
            for i in range(ar_order)
        )
        values[:, t, :] = ar_term + noise[:, t, :]
    
    metadata = {
        "generator": "ar_process",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "ar_coefficients": ar_coefficients,
        "ar_order": ar_order,
        "noise_std": noise_std,
        "seed": seed,
    }
    
    return values, metadata


def generate_trend_seasonal(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    trend_slope_range: Tuple[float, float] = (-0.01, 0.01),
    seasonal_period_range: Tuple[int, int] = (20, 50),
    seasonal_amplitude_range: Tuple[float, float] = (0.5, 1.5),
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate time series with trend + seasonal component + noise.
    
    Formula: x[t] = trend_slope * t + amplitude * sin(2π * t / period) + noise
    
    This is a classic time series decomposition. Tests if the model
    can separate trend from seasonality.
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        trend_slope_range: (min, max) slope for linear trend.
        seasonal_period_range: (min, max) period for seasonal component.
        seasonal_amplitude_range: (min, max) amplitude for seasonal component.
        noise_std: Standard deviation of additive noise.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random parameters
    slope_min, slope_max = trend_slope_range
    period_min, period_max = seasonal_period_range
    amp_min, amp_max = seasonal_amplitude_range
    
    slopes = torch.rand(num_samples, num_streams) * (slope_max - slope_min) + slope_min
    periods = torch.randint(period_min, period_max + 1, (num_samples, num_streams)).float()
    amplitudes = torch.rand(num_samples, num_streams) * (amp_max - amp_min) + amp_min
    
    # Time axis
    t = torch.arange(time_steps, dtype=torch.float32).view(1, time_steps, 1)
    
    # Broadcast parameters
    slope = slopes.unsqueeze(1)  # (num_samples, 1, num_streams)
    period = periods.unsqueeze(1)
    amp = amplitudes.unsqueeze(1)
    
    # Compute components
    trend = slope * t
    seasonal = amp * torch.sin(2 * math.pi * t / period)
    noise = torch.randn(num_samples, time_steps, num_streams) * noise_std
    
    values = trend + seasonal + noise
    
    metadata = {
        "generator": "trend_seasonal",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "trend_slope_range": trend_slope_range,
        "seasonal_period_range": seasonal_period_range,
        "seasonal_amplitude_range": seasonal_amplitude_range,
        "noise_std": noise_std,
        "seed": seed,
    }
    
    return values, metadata


def generate_regime_switching(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    num_regimes: int = 2,
    min_regime_length: int = 20,
    regime_params: Optional[List[Dict]] = None,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate time series with regime switching behavior.
    
    The time series alternates between different "regimes" with
    different statistical properties (mean, volatility, trend).
    
    This tests if the model can detect regime changes - a key
    capability for financial and industrial applications.
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        num_regimes: Number of different regimes.
        min_regime_length: Minimum time steps per regime.
        regime_params: List of dicts with regime parameters:
            - "mean": Mean level in this regime
            - "volatility": Noise std in this regime
            - "drift": Trend in this regime
            Default: alternating calm (low vol) and volatile (high vol) regimes.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters and regime labels.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if regime_params is None:
        # Default: calm vs volatile regimes
        regime_params = [
            {"mean": 0.0, "volatility": 0.05, "drift": 0.001},   # Calm regime
            {"mean": 0.0, "volatility": 0.2, "drift": -0.002},   # Volatile regime
        ]
    
    num_regimes = len(regime_params)
    
    values = torch.zeros(num_samples, time_steps, num_streams)
    regime_labels = torch.zeros(num_samples, time_steps, dtype=torch.long)
    
    # Use progress bar for large datasets
    sample_iter = _progress_bar(
        range(num_samples), 
        total=num_samples, 
        desc="    Generating regime_switching samples",
        disable=(num_samples < 1000)
    )
    
    for sample_idx in sample_iter:
        # Generate regime switch points
        t = 0
        current_regime = torch.randint(0, num_regimes, (1,)).item()
        prev_value = torch.zeros(num_streams)
        
        while t < time_steps:
            # How long to stay in current regime
            regime_length = torch.randint(
                min_regime_length, 
                min(min_regime_length * 3, time_steps - t + 1),
                (1,)
            ).item()
            regime_length = min(regime_length, time_steps - t)
            
            # Get regime parameters
            params = regime_params[current_regime]
            mean = params.get("mean", 0.0)
            vol = params.get("volatility", 0.1)
            drift = params.get("drift", 0.0)
            
            # Generate values for this regime (random walk with regime params)
            for i in range(regime_length):
                step = torch.randn(num_streams) * vol + drift
                new_value = prev_value + step + mean * (1 if i == 0 else 0)
                values[sample_idx, t + i, :] = new_value
                regime_labels[sample_idx, t + i] = current_regime
                prev_value = new_value
            
            t += regime_length
            current_regime = (current_regime + 1) % num_regimes
    
    metadata = {
        "generator": "regime_switching",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "num_regimes": num_regimes,
        "min_regime_length": min_regime_length,
        "regime_params": regime_params,
        "regime_labels": regime_labels,
        "seed": seed,
    }
    
    return values, metadata


def generate_regime_ar(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    num_regimes: int = 3,
    min_regime_length: int = 30,
    regime_ar_params: Optional[List[Dict]] = None,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate time series with regime-switching AR dynamics.
    
    Combines regime switching with autoregressive (AR) processes.
    Each regime has its own AR coefficients and noise level, creating
    time series that switch between different dynamical behaviors.
    
    This is harder than pure AR (which has constant dynamics) and tests
    if the model can both learn AR structure AND detect regime changes.
    
    Real-world examples:
    - Financial markets: Bull vs bear vs sideways regimes
    - Industrial sensors: Normal vs degraded vs fault modes
    - Weather: Different seasonal patterns
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        num_regimes: Number of different AR regimes.
        min_regime_length: Minimum time steps per regime.
        regime_ar_params: List of dicts with regime AR parameters:
            - "ar_coefficients": List of AR coefficients
            - "noise_std": Innovation noise std
            - "mean_level": Mean reversion level (optional)
            Default: 3 regimes with different AR dynamics.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters and regime labels.
    
    Example:
        >>> values, meta = generate_regime_ar(
        ...     100, 200, 5,
        ...     regime_ar_params=[
        ...         {"ar_coefficients": [0.9], "noise_std": 0.05},  # Trending
        ...         {"ar_coefficients": [0.3, -0.5], "noise_std": 0.2},  # Oscillating
        ...     ]
        ... )
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if regime_ar_params is None:
        # Default: 3 distinct AR regimes
        regime_ar_params = [
            # Regime 0: High persistence, low noise (trending)
            {"ar_coefficients": [0.95], "noise_std": 0.05, "mean_level": 0.0},
            # Regime 1: Mean-reverting with oscillation (choppy)
            {"ar_coefficients": [0.4, -0.3], "noise_std": 0.15, "mean_level": 0.0},
            # Regime 2: High volatility, moderate persistence (volatile trending)
            {"ar_coefficients": [0.7], "noise_std": 0.3, "mean_level": 0.0},
        ]
    
    num_regimes = len(regime_ar_params)
    
    values = torch.zeros(num_samples, time_steps, num_streams)
    regime_labels = torch.zeros(num_samples, time_steps, dtype=torch.long)
    
    # Use progress bar for large datasets
    sample_iter = _progress_bar(
        range(num_samples), 
        total=num_samples, 
        desc="    Generating regime_ar samples",
        disable=(num_samples < 1000)  # Only show for large datasets
    )
    
    for sample_idx in sample_iter:
        # Generate regime switch points
        t = 0
        current_regime = torch.randint(0, num_regimes, (1,)).item()
        
        # Initialize history for AR
        max_ar_order = max(len(p["ar_coefficients"]) for p in regime_ar_params)
        history = torch.zeros(max_ar_order, num_streams)
        
        while t < time_steps:
            # How long to stay in current regime
            remaining = time_steps - t
            if remaining <= min_regime_length:
                # Not enough time left, use all remaining
                regime_length = remaining
            else:
                # Random length between min and min*3 (capped by remaining)
                max_len = min(min_regime_length * 3, remaining)
                regime_length = torch.randint(min_regime_length, max_len + 1, (1,)).item()
            
            # Get regime AR parameters
            params = regime_ar_params[current_regime]
            ar_coeffs = torch.tensor(params["ar_coefficients"], dtype=torch.float32)
            noise_std = params.get("noise_std", 0.1)
            mean_level = params.get("mean_level", 0.0)
            ar_order = len(ar_coeffs)
            
            # Generate values for this regime using AR dynamics
            for i in range(regime_length):
                # AR term: sum of coeffs * past values
                ar_term = torch.zeros(num_streams)
                for j in range(ar_order):
                    ar_term += ar_coeffs[j] * history[j]
                
                # Mean reversion toward mean_level
                ar_term += (1 - ar_coeffs.sum()) * mean_level
                
                # Add innovation noise
                noise = torch.randn(num_streams) * noise_std
                new_value = ar_term + noise
                
                values[sample_idx, t + i, :] = new_value
                regime_labels[sample_idx, t + i] = current_regime
                
                # Update history (shift and add new value)
                history = torch.roll(history, 1, dims=0)
                history[0] = new_value
            
            t += regime_length
            current_regime = (current_regime + 1) % num_regimes
    
    metadata = {
        "generator": "regime_ar",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "num_regimes": num_regimes,
        "min_regime_length": min_regime_length,
        "regime_ar_params": regime_ar_params,
        "regime_labels": regime_labels,
        "seed": seed,
    }
    
    return values, metadata


def generate_mixed(
    num_samples: int,
    time_steps: int,
    num_streams: int,
    generators: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    """
    Generate a mixed dataset with different types of time series.
    
    Useful for training a model that needs to handle diverse patterns.
    Each sample is generated by a randomly chosen generator.
    
    Args:
        num_samples: Number of independent time series to generate.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors per sample.
        generators: List of generator names to use. Default: all generators.
        seed: Random seed for reproducibility.
    
    Returns:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Dict with generation parameters and source labels.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    available_generators = {
        "sine_waves": generate_sine_waves,
        "random_walk": generate_random_walk,
        "ar_process": generate_ar_process,
        "trend_seasonal": generate_trend_seasonal,
        "regime_switching": generate_regime_switching,
        "regime_ar": generate_regime_ar,
    }
    
    if generators is None:
        generators = list(available_generators.keys())
    
    # Allocate output
    values = torch.zeros(num_samples, time_steps, num_streams)
    source_labels = []
    
    # Generate each sample from a random generator
    samples_per_gen = num_samples // len(generators)
    
    idx = 0
    for gen_name in generators:
        gen_func = available_generators[gen_name]
        n = samples_per_gen if gen_name != generators[-1] else (num_samples - idx)
        
        gen_values, _ = gen_func(
            num_samples=n,
            time_steps=time_steps,
            num_streams=num_streams,
            seed=seed + idx if seed else None,
        )
        
        values[idx:idx+n] = gen_values
        source_labels.extend([gen_name] * n)
        idx += n
    
    # Shuffle
    perm = torch.randperm(num_samples)
    values = values[perm]
    source_labels = [source_labels[i] for i in perm.tolist()]
    
    metadata = {
        "generator": "mixed",
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "generators_used": generators,
        "source_labels": source_labels,
        "seed": seed,
    }
    
    return values, metadata


# =============================================================================
# DATASET CLASS
# =============================================================================

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.
    
    Wraps generated (or real) time series data and provides
    (input, target) pairs for training. The target is the same
    as input (the model learns to predict shifted values internally).
    
    Attributes:
        values: Tensor of shape (num_samples, time_steps, num_streams).
        metadata: Optional dict with generation info.
    
    Usage:
        >>> values, meta = generate_sine_waves(1000, 200, 10)
        >>> dataset = TimeSeriesDataset(values)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     values = batch["values"]
        ...     outputs = model(values, targets=values)
    """
    
    def __init__(
        self,
        values: Tensor,
        metadata: Optional[Dict] = None,
        normalize: bool = False,
    ):
        """
        Args:
            values: Tensor of shape (num_samples, time_steps, num_streams).
            metadata: Optional generation metadata.
            normalize: If True, normalize each sample to zero mean, unit std.
        """
        self.values = values
        self.metadata = metadata or {}
        
        if normalize:
            # Normalize per-sample, per-stream
            mean = values.mean(dim=1, keepdim=True)
            std = values.std(dim=1, keepdim=True) + 1e-8
            self.values = (values - mean) / std
            self.metadata["normalized"] = True
            self.metadata["normalize_mean"] = mean
            self.metadata["normalize_std"] = std
    
    def __len__(self) -> int:
        return self.values.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Get a single sample.
        
        Returns dict with:
            - "values": (time_steps, num_streams) input time series
            - "targets": same as values (model handles shifting internally)
        """
        values = self.values[idx]  # (time_steps, num_streams)
        
        return {
            "values": values,
            "targets": values,  # Target is same as input; model shifts internally
        }


# =============================================================================
# DISK CACHING
# =============================================================================

def _compute_cache_key(
    generator: str,
    num_samples: int,
    time_steps: int,
    num_streams: int,
    seed: int,
    **kwargs,
) -> str:
    """Compute a deterministic hash for cache key based on generation params."""
    params = {
        "generator": generator,
        "num_samples": num_samples,
        "time_steps": time_steps,
        "num_streams": num_streams,
        "seed": seed,
        **kwargs,
    }
    # Sort keys for determinism, convert to JSON string
    param_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:12]


def _save_cache(values: Tensor, metadata: Dict, cache_path: Path) -> None:
    """Save generated data to compressed cache file."""
    cache_data = {
        "values": values,
        "metadata": {k: v for k, v in metadata.items() if not isinstance(v, Tensor)},
    }
    buffer = io.BytesIO()
    torch.save(cache_data, buffer)
    buffer.seek(0)
    with gzip.open(cache_path, 'wb', compresslevel=6) as f:
        f.write(buffer.read())


def _load_cache(cache_path: Path) -> Tuple[Tensor, Dict]:
    """Load data from compressed cache file."""
    with gzip.open(cache_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    cache_data = torch.load(buffer, weights_only=False)
    return cache_data["values"], cache_data["metadata"]


def create_train_val_datasets(
    generator: str = "mixed",
    train_samples: int = 10000,
    val_samples: int = 1000,
    time_steps: int = 200,
    num_streams: int = 10,
    normalize: bool = True,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    **generator_kwargs,
) -> Tuple[TimeSeriesDataset, TimeSeriesDataset]:
    """
    Create train and validation datasets using a specified generator.
    
    Supports disk caching: if cache_dir is provided, generated data is saved
    to disk and reloaded on subsequent runs with matching parameters. This
    speeds up training restarts and ensures reproducibility.
    
    Args:
        generator: Name of generator to use:
            "sine_waves", "random_walk", "ar_process", 
            "trend_seasonal", "regime_switching", "regime_ar", "mixed"
        train_samples: Number of training samples.
        val_samples: Number of validation samples.
        time_steps: Length of each time series.
        num_streams: Number of streams/sensors.
        normalize: Whether to normalize the data.
        seed: Random seed for reproducibility.
        cache_dir: Directory to cache generated data. If None, no caching.
        **generator_kwargs: Additional arguments for the generator.
    
    Returns:
        train_dataset: TimeSeriesDataset for training.
        val_dataset: TimeSeriesDataset for validation.
    
    Example:
        >>> train_ds, val_ds = create_train_val_datasets(
        ...     generator="sine_waves",
        ...     train_samples=5000,
        ...     val_samples=500,
        ...     num_streams=5,
        ...     cache_dir="output/cache",  # Enable caching
        ... )
    """
    generators = {
        "sine_waves": generate_sine_waves,
        "random_walk": generate_random_walk,
        "ar_process": generate_ar_process,
        "trend_seasonal": generate_trend_seasonal,
        "regime_switching": generate_regime_switching,
        "regime_ar": generate_regime_ar,
        "mixed": generate_mixed,
    }
    
    if generator not in generators:
        raise ValueError(f"Unknown generator: {generator}. Available: {list(generators.keys())}")
    
    gen_func = generators[generator]
    
    # Check for cached data
    train_values, train_meta = None, None
    val_values, val_meta = None, None
    
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Compute cache keys
        train_key = _compute_cache_key(
            generator, train_samples, time_steps, num_streams, seed, **generator_kwargs
        )
        val_key = _compute_cache_key(
            generator, val_samples, time_steps, num_streams, seed + 10000, **generator_kwargs
        )
        
        train_cache = cache_path / f"train_{generator}_{train_key}.pt.gz"
        val_cache = cache_path / f"val_{generator}_{val_key}.pt.gz"
        
        # Try to load from cache
        if train_cache.exists():
            size_mb = train_cache.stat().st_size / (1024 * 1024)
            print(f"  ✓ Loading cached training data: {train_cache.name} ({size_mb:.1f} MB)")
            train_values, train_meta = _load_cache(train_cache)
            print(f"    Loaded {train_values.shape[0]} samples")
        
        if val_cache.exists():
            size_mb = val_cache.stat().st_size / (1024 * 1024)
            print(f"  ✓ Loading cached validation data: {val_cache.name} ({size_mb:.1f} MB)")
            val_values, val_meta = _load_cache(val_cache)
            print(f"    Loaded {val_values.shape[0]} samples")
    
    # Generate if not cached
    if train_values is None:
        print(f"  Generating {train_samples} training samples (seed={seed})...")
        train_values, train_meta = gen_func(
            num_samples=train_samples,
            time_steps=time_steps,
            num_streams=num_streams,
            seed=seed,
            **generator_kwargs,
        )
        print(f"    Generated tensor shape: {tuple(train_values.shape)}")
        if cache_dir:
            print(f"  Saving to cache...")
            _save_cache(train_values, train_meta, train_cache)
            size_mb = train_cache.stat().st_size / (1024 * 1024)
            print(f"  ✓ Cached: {train_cache.name} ({size_mb:.1f} MB)")
    
    if val_values is None:
        print(f"  Generating {val_samples} validation samples (seed={seed + 10000})...")
        val_values, val_meta = gen_func(
            num_samples=val_samples,
            time_steps=time_steps,
            num_streams=num_streams,
            seed=seed + 10000,
            **generator_kwargs,
        )
        print(f"    Generated tensor shape: {tuple(val_values.shape)}")
        if cache_dir:
            print(f"  Saving to cache...")
            _save_cache(val_values, val_meta, val_cache)
            size_mb = val_cache.stat().st_size / (1024 * 1024)
            print(f"  ✓ Cached: {val_cache.name} ({size_mb:.1f} MB)")
    
    train_dataset = TimeSeriesDataset(train_values, train_meta, normalize=normalize)
    val_dataset = TimeSeriesDataset(val_values, val_meta, normalize=normalize)
    
    return train_dataset, val_dataset
