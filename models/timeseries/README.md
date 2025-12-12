# Time Series Tokenization and Modeling

This module provides tools for tokenizing multivariate time series and training Transformer models for forecasting tasks.

## Key Features

- **Percentage-Based Pattern Encoding**: Captures relative changes (scale-invariant)
- **Factorized Embeddings**: Efficient parameter usage, scales to 100+ bins
- **Transformer Model**: Decoder-only transformer for next-window prediction
- **Synthetic Data Generators**: For testing and development
- **MSE Loss with Skill Score**: Compare against random walk baseline

## Quick Start

```python
from models.timeseries import (
    create_model,
    create_train_val_datasets,
)
import torch
from torch.utils.data import DataLoader

# Create synthetic data
train_ds, val_ds = create_train_val_datasets(
    generator="sine_waves",
    train_samples=5000,
    num_streams=10,
)

# Create model
model = create_model(
    num_streams=10,
    window_size=5,
    num_bins=50,
    d_model=128,
    n_layers=4,
    delta_mode="percent",  # Scale-invariant!
)

# Training loop
loader = DataLoader(train_ds, batch_size=32, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for batch in loader:
    outputs = model(batch["values"], targets=batch["targets"])
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Concept Overview

### Pattern Encoding (Percentage-Based)

The tokenizer converts raw values into **percentage changes** (not absolute differences):

```
delta = (x[t] - x[t-1]) / |x[t-1]|
```

This makes the encoding **scale-invariant**:
- 10 → 15 (+50%) and 1000 → 1500 (+50%) produce the **same** pattern bin
- Level features capture absolute scale separately

### Factorized Embeddings

The final token embedding is:

```
token_emb = stream_emb + pattern_emb + time_emb + level_proj(level_features)
```

Where:
- `stream_emb`: Which sensor/variable
- `pattern_emb`: Combined percentage-delta embeddings
- `time_emb`: Window position in sequence
- `level_proj`: Projected continuous statistics (last, mean, std)

### Delta Modes

| Mode | Formula | Use Case |
|------|---------|----------|
| `"percent"` (default) | `(x[t] - x[t-1]) / \|x[t-1]\|` | Most applications, scale-invariant |
| `"absolute"` | `x[t] - x[t-1]` | When absolute changes matter |
| `"log_return"` | `log(x[t] / x[t-1])` | Finance (symmetric, additive) |

## Understanding the Loss Function

We use **Mean Squared Error (MSE)** for next-window prediction:

```python
loss = MSE(predicted_windows, actual_windows)
```

### Why MSE is Good for Time Series

1. **Quadratic Penalty**: Big errors penalized more (desirable for forecasting)
2. **Proper Scoring Rule**: Optimal prediction = true expected value
3. **Interpretable**: RMSE has same units as data
4. **Decomposable**: MSE = Bias² + Variance + Irreducible Error

### Skill Score: Are We Learning?

We compare against a **random walk baseline** (predict previous value):

```
skill_score = 1 - (model_mse / baseline_mse)
```

| Skill Score | Meaning |
|-------------|---------|
| > 0 | Model beats baseline ✓ |
| = 0 | Same as baseline (useless) |
| < 0 | Worse than baseline (broken) |

For different data types:
- **Sine waves**: Skill should be high (~0.9) - patterns are deterministic
- **Random walk**: Skill ≈ 0 - no learnable pattern (expected!)
- **AR process**: Skill > 0 - learnable temporal dependencies

### Interpreting Training Progress

```
Loss: 0.47 | RMSE: 0.68 | Baseline RMSE: 1.47 | Skill: 0.79
```

- Loss decreasing → Model learning ✓
- Skill > 0 → Beating baseline ✓
- RMSE < Baseline RMSE → Useful predictions ✓

## Configuration

### TimeSeriesTokenizerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_streams` | int | required | Number of input streams/sensors |
| `window_size` | int | 5 | Time steps per window |
| `num_bins` | int | 5 | Bins for delta discretization |
| `d_model` | int | 128 | Embedding dimension |
| `delta_mode` | str | "percent" | "absolute", "percent", or "log_return" |
| `default_threshold_scale` | float | 0.1 | Threshold scale (±10% for percent mode) |
| `delta_combine` | str | "mean" | "sum" or "mean" for delta embeddings |

### TimeSeriesTransformerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_streams` | int | required | Number of input streams |
| `d_model` | int | 256 | Hidden dimension |
| `n_layers` | int | 6 | Number of transformer blocks |
| `n_heads` | int | 8 | Attention heads |
| `dropout` | float | 0.1 | Dropout probability |

## Data Generators

Generate synthetic time series for testing:

```python
from models.timeseries import (
    generate_sine_waves,      # Periodic patterns
    generate_random_walk,      # Brownian motion
    generate_ar_process,       # Autoregressive
    generate_trend_seasonal,   # Trend + seasonal
    generate_regime_switching, # Regime changes
    generate_mixed,            # Mix of all above
)

# Example
values, metadata = generate_sine_waves(
    num_samples=1000,
    time_steps=200,
    num_streams=10,
    noise_std=0.1,
)
```

## Training

### Command Line

```bash
# Train on sine waves
python -m models.timeseries.train \
    --generator sine_waves \
    --train_samples 10000 \
    --epochs 10 \
    --d_model 128 \
    --n_layers 4

# Train on mixed data
python -m models.timeseries.train \
    --generator mixed \
    --train_samples 20000 \
    --epochs 20
```

### Python API

```python
from models.timeseries.train import train

model, metrics = train(
    generator="sine_waves",
    train_samples=10000,
    num_streams=10,
    d_model=128,
    n_layers=4,
    epochs=10,
)
```

## Example Output

```
TimeSeriesTransformer initialized:
  Total parameters: 168,709 (0.17M)
  Pattern encoding: delta_mode='percent', num_bins=30
  Architecture: d_model=64, n_layers=2, n_heads=4

Epoch 3/3:
  Train | Loss: 0.210838 | RMSE: 0.4592 | Skill: 0.9033
  Val   | Loss: 0.204143 | RMSE: 0.4518 | Skill: 0.9068

Interpretation:
  ✓ Model significantly beats baseline (skill=0.91)
    The model has learned predictable patterns in the data.
```

## Architecture Details

### Embedding Tables

| Table | Size | Purpose |
|-------|------|---------|
| `stream_emb` | (num_streams, d_model) | Identifies which sensor |
| `delta_emb` | (num_bins, d_model) | Encodes delta magnitude |
| `delta_pos_emb` | (num_deltas, d_model) | Position within window |
| `time_emb` | (max_windows, d_model) | Window position |
| `level_proj` | (3 → d_model) | Projects level features |

### Transformer Components

- **Multi-Head Attention**: Combined QKV projection, causal masking
- **Feed-Forward**: Two-layer MLP with GELU activation
- **Pre-Norm**: LayerNorm before attention/FFN (stable training)
- **Prediction Head**: Linear projection to next window values

## Running Tests

```bash
# Tokenizer tests
python models/timeseries/tests/test_tokenizer.py

# Quick model test
python -c "
from models.timeseries import create_model
model = create_model(num_streams=5, d_model=64)
import torch
x = torch.randn(2, 50, 5)
out = model(x, targets=x)
print('Loss:', out['loss'].item())
"
```
