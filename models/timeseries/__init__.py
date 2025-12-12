"""
Time Series Tokenization and Modeling.

This module provides tools for tokenizing multivariate time series data
and training Transformer or Mamba models for time series forecasting.

Key Components
--------------
Tokenizer:
- TimeSeriesTokenizerConfig: Configuration for tokenization
- TimeSeriesTokenizer: nn.Module for tokenizing and embedding time series
- compute_delta_bins_and_levels: Helper for delta/level extraction

Models (both share the same tokenizer):
- TimeSeriesTransformerConfig: Configuration for the transformer
- TimeSeriesTransformer: Decoder-only transformer (O(n²), good for short sequences)
- TimeSeriesMambaConfig: Configuration for Mamba
- TimeSeriesMamba: Mamba SSM model (O(n), good for long sequences)
- create_model: Convenience function for model creation (default: Transformer)
- create_mamba_model: Convenience function for Mamba model creation

Data:
- generate_sine_waves: Periodic pattern generator
- generate_random_walk: Brownian motion generator
- generate_ar_process: Autoregressive process generator
- generate_trend_seasonal: Trend + seasonal pattern generator
- generate_regime_switching: Regime-switching generator
- generate_regime_ar: Regime-switching AR process generator
- generate_mixed: Mixed pattern generator
- TimeSeriesDataset: PyTorch Dataset wrapper
- create_train_val_datasets: Create train/val datasets

Choosing Between Transformer and Mamba
--------------------------------------
- **Transformer**: Better for short sequences (<200 steps), can attend to any position
- **Mamba**: Better for long sequences (200-5000+ steps), O(n) complexity

For industrial processes (minute sampling, hours of history): use Mamba
For high-frequency patterns (small context needed): use Transformer

Example
-------
    from models.timeseries import (
        create_model,          # Transformer
        create_mamba_model,    # Mamba
        create_train_val_datasets,
    )
    
    # Create data
    train_ds, val_ds = create_train_val_datasets(
        generator="regime_ar",
        train_samples=1000,
        num_streams=5,
    )
    
    # Create Transformer model (short context)
    model = create_model(num_streams=5, window_size=10, d_model=128)
    
    # OR create Mamba model (long context)
    model = create_mamba_model(num_streams=5, window_size=100, d_model=192)
    
    # Same training interface for both!
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=32)
    for batch in loader:
        outputs = model(batch["values"], targets=batch["targets"])
        loss = outputs["loss"]
"""

from .tokenizer import (
    TimeSeriesTokenizerConfig,
    TimeSeriesTokenizer,
    compute_delta_bins_and_levels,
    compute_patterns_and_levels,  # backward compatibility
    create_tokenizer,
    create_default_thresholds,
    estimate_thresholds_from_data,
)

from .model import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformer,
    create_model,
    count_parameters,
)

from .model_mamba import (
    TimeSeriesMambaConfig,
    TimeSeriesMamba,
    create_mamba_model,
)

from .data import (
    generate_sine_waves,
    generate_random_walk,
    generate_ar_process,
    generate_trend_seasonal,
    generate_regime_switching,
    generate_regime_ar,
    generate_mixed,
    TimeSeriesDataset,
    create_train_val_datasets,
)

__all__ = [
    # Tokenizer
    "TimeSeriesTokenizerConfig",
    "TimeSeriesTokenizer",
    "compute_delta_bins_and_levels",
    "compute_patterns_and_levels",
    "create_tokenizer",
    "create_default_thresholds",
    "estimate_thresholds_from_data",
    # Transformer Model
    "TimeSeriesTransformerConfig",
    "TimeSeriesTransformer",
    "create_model",
    "count_parameters",
    # Mamba Model
    "TimeSeriesMambaConfig",
    "TimeSeriesMamba",
    "create_mamba_model",
    # Data
    "generate_sine_waves",
    "generate_random_walk",
    "generate_ar_process",
    "generate_trend_seasonal",
    "generate_regime_switching",
    "generate_regime_ar",
    "generate_mixed",
    "TimeSeriesDataset",
    "create_train_val_datasets",
]
