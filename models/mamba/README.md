# Mamba Experiments

Testing Mamba state space models for comparison with transformer architectures.

## Overview

Mamba is a state space model (SSM) that provides an efficient alternative to attention mechanisms for sequence modeling. This directory contains:

1. **Pure PyTorch Mamba** (`model_mamba.py`) - Educational implementation for understanding SSM math
2. **CUDA-Optimized Mamba** (`model_mamba_fast.py`) - 10-50x faster using `mamba-ssm` library (Linux/WSL only)
3. **Pretrained Mamba** - Load `state-spaces/mamba-130m` from HuggingFace (trained on The Pile)

## Goals

1. Evaluate Mamba performance on TinyStories dataset
2. Compare with transformer baselines (sparse MoE, standard)
3. Test pretrained Mamba on text generation
4. Assess for potential use as expert in meta-architecture

## Quick Start

### Option 1: Fast Training with mamba-ssm (Linux/WSL)

```bash
# Install optimized CUDA kernels
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.0.0

# Train with CUDA-optimized implementation (10-50x faster)
python train.py --model_type mamba --use_fast --max_steps 5000

# Benchmark PyTorch vs mamba-ssm
python model_mamba_fast.py --mode benchmark
```

### Option 2: Pure PyTorch (Educational, Any Platform)

```bash
# Train pure PyTorch Mamba (slower but educational)
python train.py --model_type mamba --max_steps 5000

# Generate from trained model
python generate.py --checkpoint output/mamba_30m_pytorch_5000steps/best_checkpoint.pt
```

### Option 3: Use Pretrained Model (Recommended for testing)

```bash
# Test pretrained mamba-130m from HuggingFace
python model_mamba.py --mode pretrained

# Generate text interactively
python generate.py --pretrained state-spaces/mamba-130m --interactive
```

Available pretrained models:
- `state-spaces/mamba-130m` (130M params) ← fits in 4GB VRAM
- `state-spaces/mamba-370m` (370M params)
- `state-spaces/mamba-790m` (790M params)
- `state-spaces/mamba-1.4b` (1.4B params)
- `state-spaces/mamba-2.8b` (2.8B params)

## Structure

```
models/mamba/
├── model_mamba.py       # Pure PyTorch Mamba (educational)
├── model_mamba_fast.py  # CUDA-optimized using mamba-ssm
├── train.py             # Training script (supports both implementations)
├── generate.py          # Text generation
├── requirements.txt     # Dependencies (including mamba-ssm)
├── tests/
└── README.md
```

## Implementation Details

### Pure PyTorch Mamba (`model_mamba.py`)

The `MambaLM` class implements the full Mamba architecture without external dependencies:

- **SelectiveSSM**: Core state space model with input-dependent parameters
- **Sequential scan**: O(L) complexity (vs attention's O(L²))
- **Causal convolution**: Local context before SSM
- **Gated output**: SiLU activation for gating

Key difference from transformers: **No attention mechanism**. Instead uses:
- State space recurrence: `h[k] = Ā·h[k-1] + B̄·x[k]`
- Input-dependent discretization (Δ, B, C computed from input)

### CUDA-Optimized Mamba (`model_mamba_fast.py`)

The `FastMambaLM` class wraps the official `mamba-ssm` library:

- **Parallel selective scan**: CUDA kernels instead of Python loop
- **Fused operations**: Minimize memory bandwidth
- **10-50x speedup**: Essential for serious training

```python
# Use factory function to auto-select best implementation
from models.mamba import create_mamba_model

model = create_mamba_model(config)  # Uses mamba-ssm if available
```

### Trade-offs

| Aspect | Pure PyTorch | mamba-ssm (CUDA) | Pretrained (HuggingFace) |
|--------|--------------|------------------|--------------------------|
| Speed | Slow (sequential) | **10-50x faster** | Fast |
| Platform | Any | Linux/WSL only | Any |
| Learning | Great for understanding | Production-ready | Black box |
| Training | From scratch | From scratch | Already trained |

## Expected Performance

From literature and our hardware (Quadro T1000 4GB):

- **Mamba-130M pretrained**: Good text generation quality
- **Mamba from scratch**: Val loss ~2-3 on TinyStories (5K steps)
- **Memory**: ~500MB for 130M model
- **Inference**: Linear time O(L) vs transformer's O(L²)

## Comparison Baseline

From existing models in this repo:
- Sparse MoE Transformer: Val loss 2.2, 158M params
- BDH Transformer (backprop): Val loss 1.28, 64M params

## References

- Mamba paper: https://arxiv.org/abs/2312.00752
- Mamba2 paper: https://arxiv.org/abs/2405.21060
- HuggingFace models: https://huggingface.co/state-spaces
