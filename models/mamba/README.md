# Mamba Experiments

Testing Mamba state space models for comparison with transformer architectures.

## Overview

Mamba is a state space model (SSM) that provides an efficient alternative to attention mechanisms for sequence modeling. This directory contains:

1. **Pure PyTorch Mamba** - Educational implementation for training from scratch
2. **Pretrained Mamba** - Load `state-spaces/mamba-130m` from HuggingFace (trained on The Pile)

## Goals

1. Evaluate Mamba performance on TinyStories dataset
2. Compare with transformer baselines (sparse MoE, standard)
3. Test pretrained Mamba on text generation
4. Assess for potential use as expert in meta-architecture

## Quick Start

### Option 1: Use Pretrained Model (Recommended for testing)

```bash
cd mamba_experiments

# Test pretrained mamba-130m from HuggingFace
python src/model_mamba.py --mode pretrained

# Generate text interactively
python src/generate.py --pretrained state-spaces/mamba-130m --interactive
```

Available pretrained models:
- `state-spaces/mamba-130m` (130M params) ← fits in 4GB VRAM
- `state-spaces/mamba-370m` (370M params)
- `state-spaces/mamba-790m` (790M params)
- `state-spaces/mamba-1.4b` (1.4B params)
- `state-spaces/mamba-2.8b` (2.8B params)

### Option 2: Train from Scratch

```bash
# Train pure PyTorch Mamba on TinyStories
python src/train.py --model_type mamba --max_steps 5000

# Generate from trained model
python src/generate.py --checkpoint output/mamba_130m_5000steps/best_checkpoint.pt
```

## Structure

```
mamba_experiments/
├── src/
│   ├── config.py          # Model configurations
│   ├── model_mamba.py     # Pure PyTorch Mamba + HuggingFace loader
│   ├── train.py           # Training script
│   └── generate.py        # Text generation
├── tests/
├── output/                # Checkpoints
├── notebooks/
└── README.md
```

## Implementation Details

### Pure PyTorch Mamba

The `MambaLM` class implements the full Mamba architecture without external dependencies:

- **SelectiveSSM**: Core state space model with input-dependent parameters
- **Sequential scan**: O(L) complexity (vs attention's O(L²))
- **Causal convolution**: Local context before SSM
- **Gated output**: SiLU activation for gating

Key difference from transformers: **No attention mechanism**. Instead uses:
- State space recurrence: `h[k] = Ā·h[k-1] + B̄·x[k]`
- Input-dependent discretization (Δ, B, C computed from input)

### Trade-offs

| Aspect | Pure PyTorch | Pretrained (HuggingFace) |
|--------|--------------|--------------------------|
| Speed | Slower (sequential scan) | Faster (optimized kernels) |
| Platform | Any (Windows/Mac/Linux) | Any |
| Learning | Great for understanding | Black box |
| Training | From scratch | Already trained on The Pile |

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
