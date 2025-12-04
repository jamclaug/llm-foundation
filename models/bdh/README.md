# BDH-SLIM Transformer Implementation

**Status**: Phase 1 Implementation Complete  
**Date**: December 2, 2025

## Overview

Implementation of the Baby Dragon Hatchling (BDH-SLIM) transformer architecture with Hebbian learning, based on the paper ["The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"](https://arxiv.org/abs/2509.26507).

## Key Features

### Biological Learning
- **Hebbian Learning**: Local weight updates based on co-activation ("neurons that fire together, wire together")
- **Activity-Dependent Pruning**: Weak synapses are removed over time (biological sparsity)
- **Homeostatic Plasticity**: Neurons regulate their activity to prevent runaway excitation

### Training Modes
1. **Backprop Only**: Standard gradient descent (like GPT-2)
2. **Hebbian Only**: Pure local learning rules (biologically plausible)
3. **Hybrid**: Both mechanisms simultaneously (best of both worlds)

### Architecture
- Standard transformer structure (attention + FFN)
- Hebbian-capable linear layers
- Sparse, interpretable activations
- Compatible with existing training infrastructure

## Files

```
src/
├── hebbian.py          # Hebbian learning primitives
│   ├── HebbianLinear   # Linear layer with local learning
│   ├── HebbianAttention # Attention with Hebbian updates
│   ├── HebbianFFN      # Feed-forward with Hebbian updates
│   └── Biological functions (pruning, homeostatic scaling)
│
├── model_bdh.py        # BDH transformer architecture
│   ├── BDHTransformerLayer
│   ├── BDHTransformer
│   └── create_bdh_config()
│
├── train_bdh.py        # BDH-specific training script
└── config.py           # Updated with BDH parameters
```

## Quick Start

### Training with Hybrid Learning (Recommended)

```bash
# Default 158M model, hybrid learning
python src/train_bdh.py --mode train --max_steps 66000

# Specify learning mode
python src/train_bdh.py --learning_mode hybrid --hebbian_lr 0.01
```

### Training with Pure Hebbian Learning

```bash
# Biological learning only (no backprop)
python src/train_bdh.py --learning_mode hebbian --hebbian_lr 0.05
```

### Training with Pure Backpropagation

```bash
# Standard gradient descent (baseline)
python src/train_bdh.py --learning_mode backprop
```

### Custom Architecture

```bash
# Larger model (fit on 4GB GPU)
python src/train_bdh.py \
  --d_model 512 \
  --n_heads 8 \
  --d_ff 1024 \
  --n_layers 6 \
  --batch_size 4 \
  --max_steps 66000 \
  --output_dir output/bdh_large
```

## Configuration Parameters

### BDH-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_mode` | `hybrid` | Learning paradigm: `backprop`, `hebbian`, or `hybrid` |
| `hebbian_lr` | `0.01` | Hebbian learning rate (local updates) |
| `prune_every` | `1000` | Apply biological pruning every N steps |
| `prune_threshold` | `0.01` | Threshold for weak connection removal |
| `homeostatic_scaling` | `True` | Enable activity regulation |

### Standard Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | `512` | Embedding dimension |
| `n_heads` | `8` | Attention heads |
| `d_ff` | `1024` | FFN hidden dimension |
| `n_layers` | `6` | Transformer layers |
| `dropout` | `0.1` | Dropout rate |
| `max_len` | `256` | Max sequence length |

## How It Works

### Hebbian Learning Rule

```python
# Standard backprop: Δw = -η * ∂L/∂w (global gradient)
loss.backward()
optimizer.step()

# Hebbian learning: Δw = η * x * y (local activation)
model.hebbian_update()
```

### Hybrid Training Loop

```python
for batch in dataloader:
    # Forward pass (caches activations)
    output = model(input_ids, labels=labels)
    loss = output["loss"]
    
    # Backprop update (if enabled)
    if learning_mode in ['backprop', 'hybrid']:
        loss.backward()
        optimizer.step()
    
    # Hebbian update (if enabled)
    if learning_mode in ['hebbian', 'hybrid']:
        model.hebbian_update()
    
    # Biological plasticity (periodic)
    if step % prune_every == 0:
        model.apply_biological_plasticity()
```

### Activity-Dependent Pruning

Weak synapses (low absolute weight) are removed periodically:

```python
# Prune weights below threshold
mask = (weight.abs() > threshold).float()
weight.data *= mask  # Zero out weak connections
```

### Homeostatic Scaling

Neurons regulate their activity to maintain target firing rate:

```python
# Scale weights to achieve target activity
current_activity = weight.abs().mean()
scale_factor = target_activity / current_activity
weight.data *= scale_factor
```

## Expected Results

### Comparison with Standard Transformer

| Metric | Standard (Backprop) | BDH (Hybrid) | BDH (Hebbian Only) |
|--------|---------------------|--------------|-------------------|
| Val Loss @ 5K steps | ~2.2 | TBD | TBD |
| Training Speed | 1.0x | ~0.95x | ~1.05x |
| Memory Usage | 2.77GB | ~2.8GB | ~2.6GB |
| Sparsity | 0% | TBD | TBD |

### Biological Properties

- **Sparsity**: Weight sparsity increases over training (pruning)
- **Monosemanticity**: Individual neurons respond to specific concepts
- **Interpretability**: Activations are sparse and positive
- **Plasticity**: Model adapts quickly to new patterns (Hebbian)

## Research Questions

1. **Learning Efficiency**: Does Hebbian learning improve sample efficiency?
2. **Generalization**: Do biological constraints help or hurt generalization?
3. **Sparsity**: How does learned sparsity compare to architectural sparsity (MoE)?
4. **Interpretability**: Are BDH activations more interpretable than standard transformers?
5. **Hybrid Benefits**: Does combining backprop + Hebbian outperform either alone?

## Limitations

### Current Implementation

- **Not yet tested**: No training results yet (just implemented)
- **Simplified**: Doesn't include all BDH paper features (dendrites, spiking neurons)
- **Single GPU**: Designed for 4GB VRAM (no distributed training)

### Future Work

- Add dendrite-inspired computations
- Implement spiking neuron dynamics
- Add graph-based connectivity (scale-free network)
- Multi-GPU support for larger models
- Comprehensive evaluation benchmarks

## Debugging & Analysis

### Check Sparsity

```python
from model_bdh import BDHTransformer

model = BDHTransformer(config)
# ... train model ...

stats = model.get_sparsity_stats()
print(f"Weight sparsity: {stats['weight_sparsity']:.2%}")
print(f"Active params: {stats['active_params']:,}")
```

### Visualize Hebbian Updates

```python
# Track weight changes over time
initial_weights = model.layers[0].attn.w_q.weight.clone()

# ... train for N steps with Hebbian learning ...

final_weights = model.layers[0].attn.w_q.weight
delta_w = final_weights - initial_weights

# Visualize: which connections strengthened?
import matplotlib.pyplot as plt
plt.hist(delta_w.flatten().cpu(), bins=100)
plt.xlabel("Weight Change (Δw)")
plt.ylabel("Count")
plt.title("Hebbian Weight Updates Distribution")
plt.show()
```

### Monitor Activity

```python
# Check if neurons are firing
hidden_states = output["hidden"]
activity = (hidden_states > 0).float().mean()
print(f"Neuron activity: {activity:.2%}")
```

## Integration with Meta-Architecture

This BDH model is designed to be compatible with the future meta-architecture vision:

```python
# Future: Mix BDH with standard transformers
from model_bdh import BDHTransformer
from model import SparseMoETransformer

class MetaArchitectureMoE(nn.Module):
    def __init__(self, config):
        self.experts = [
            BDHTransformer(config),          # Hebbian expert
            SparseMoETransformer(config),    # Gradient expert
            # ... more architectures
        ]
    
    def forward(self, x):
        # Route to different experts
        # Each uses its own learning rule!
        pass
```

## References

- **Paper**: [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507)
- **Meta-Architecture Vision**: See `papers_and_notes/meta_architecture_vision.md`
- **Base Training Framework**: `sparse_moe_transformer/`

## Getting Started

1. **Install dependencies**: Already installed (PyTorch, transformers, etc.)
2. **Train baseline**: `python src/train_bdh.py --learning_mode backprop`
3. **Train BDH**: `python src/train_bdh.py --learning_mode hybrid`
4. **Compare results**: Check validation loss, sparsity, generation quality

## Status & Next Steps

**✅ Completed**:
- Hebbian learning primitives
- BDH transformer architecture
- Training script with hybrid learning
- Biological plasticity mechanisms
- Configuration integration

**🔄 In Progress**:
- First training run (need to start)

**⏳ TODO**:
- Validation on TinyStories
- Comparison with standard transformer
- Sparsity analysis
- Generation quality evaluation
- Integration with base train.py
- Documentation of results
