# Phase 1: Foundation

Core neural network fundamentals - backpropagation and attention mechanisms.

## Learning Objectives
- Understand backpropagation through manual NumPy implementation
- Explore attention mechanisms using PyTorch
- Build intuition for gradient flow and self-attention

## Directory Structure

### `notebooks/`
Interactive exploration and learning:
- `backprop.ipynb` - Step-by-step backpropagation with visualizations
- `attention_mechanism.ipynb` - Attention mechanisms from basics to multi-head
- `ch3_llm_from_scratch.ipynb` - LLM concepts following book/tutorial

### `src/`
Clean, extracted implementations:
- `backprop_from_scratch.py` - Pure NumPy backprop (class-based + manual)
  - `TinyNN` class with parameter management
  - Manual matrix implementation for comparison
  - Example: XOR problem training
- `attention_mechanism.py` - Attention implementations (to be extracted from notebook)

### `tests/`
- `test_implementations.py` - Validation scripts for backprop and attention

## Running Examples

### Backpropagation
```bash
# Run the script
python phase1_foundation/src/backprop_from_scratch.py

# Or explore interactively
jupyter notebook phase1_foundation/notebooks/backprop.ipynb
```

### Attention Mechanism
```bash
# Explore concepts
jupyter notebook phase1_foundation/notebooks/attention_mechanism.ipynb
```

## Key Patterns

### Backpropagation (NumPy-only)
- Parameters stored in `self.params`: `W1`, `b1`, `W2`, `b2`
- Cache intermediate values: `Z1`, `A1`, `Z2`, `A2`
- Gradients in `self.grads`: `dW1`, `db1`, `dW2`, `db2`
- Training loop: forward → loss → backward → update

### Attention (PyTorch)
- Scaled dot-product: `softmax(Q @ K.T / sqrt(d_k)) @ V`
- Multi-head splits `d_model` across heads
- Use `torch.nn.MultiheadAttention` for efficiency
