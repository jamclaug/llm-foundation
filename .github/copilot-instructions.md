# LLM Foundation - Copilot Instructions

## Project Overview
Educational repository for building LLM understanding from first principles, plus a production Sparse Mixture-of-Experts Transformer implementation. This is a **learning-focused codebase** with educational implementations (phase1) and working production models (sparse_moe_transformer).

## Architecture & Organization

### Educational Path (`phase1_foundation/` through `phase5_production/`)
Five-phase learning progression, each building on the previous:
- `notebooks/` - Interactive exploration and learning
- `src/` - Clean, extracted implementations  
- `tests/` - Validation scripts
- `README.md` - Phase overview and learning objectives

**Phase 1: Foundation** (`phase1_foundation/`)
Core neural network fundamentals:
- **Backpropagation** (`src/backprop_from_scratch.py`): Pure NumPy implementation
  - Class-based `TinyNN` and manual matrix approaches
  - XOR problem training with tanh activation and MSE loss
  - Notebook: `notebooks/backprop.ipynb`
- **Attention mechanisms** (`notebooks/attention_mechanism.ipynb`): PyTorch-based
  - Multi-head attention using `torch.nn.MultiheadAttention`
  - Implementations extracted to `src/attention_mechanism.py`

**Phase 2-5**: Architecture, Components, Training, Production (planned expansion)

### Production Model (`sparse_moe_transformer/`)
Fully functional Sparse Mixture-of-Experts Transformer for language modeling:

**Architecture:**
- Decoder-only transformer (GPT-style) with 6 layers, 8 attention heads, 512 embedding dim
- **Sparse MoE FFN**: 16 expert networks, top-2 routing per token (87.5% sparsity)
- 158M total parameters, ~20M active per token
- Standard gradient-based learning (backpropagation + Adam)

**Key Files:**
- `src/train.py` - Main training script with `SparseMoETransformer`, `SparseMoEFFN`, `SparseMoETransformerLayer` classes
- `tests/test_sparse_activation.py` - Verification suite for expert routing and sparsity
- `output/` - Model checkpoints and visualizations
- `README.md` - Complete documentation

**Performance:**
- Val loss: 2.2 on TinyStories (2.1M samples, 5K steps)
- Inference: 37ms latency, 706MB GPU memory, 8.05 GFLOPs
- Hardware: Trained on Quadro T1000 (4GB VRAM)

## Development Patterns

### Educational Code Style (phase1_foundation)
- **Backpropagation**: Pure NumPy for educational transparency - no PyTorch/TensorFlow
  - Manual gradient computation: `dW = np.dot(A_prev.T, dA) / m`
  - Explicit caching of intermediate values (`Z`, `A`) for backprop
  - Weight initialization: `np.random.randn(n_in, n_out) * 0.1`
- **Attention mechanisms**: Use PyTorch for efficiency
  - Multi-head attention is complex to implement manually - leverage `torch.nn.MultiheadAttention`
  - Focus on understanding the mechanism rather than low-level matrix operations

### Production Code Style (sparse_moe_transformer)
- **Standard PyTorch patterns**: Use nn.Module, functional operations
- **Class structure**: 
  - `SparseMoETransformer` - Main model
  - `SparseMoETransformerLayer` - Single layer with attention + MoE FFN
  - `SparseMoEFFN` - Sparse expert routing module
- **Memory optimization**: Loop over unique experts to prevent memory explosion
- **Comprehensive comments**: Explain architecture decisions and mathematical operations

### Training Loop Pattern
```python
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y, y_pred)
    loss.backward()
    optimizer.step()
```

### Network Structure Convention
- Class-based networks store parameters in `self.params` dict or as nn.Parameter
- Caching in `self.cache` for educational implementations
- Gradients in `self.grads` for manual backprop

## Dependencies & Environment
- **Core**: `numpy`, `matplotlib`, `jupyter`, `ipykernel`, `graphviz` (see `requirements.txt`)
- **Python**: Use Python 3.x with pip for package management
- **Jupyter**: Required for interactive notebooks in phase1

### Setup Commands
```bash
pip install -r requirements.txt

# Run backprop example
python phase1_foundation/src/backprop_from_scratch.py

# Explore interactively
jupyter notebook phase1_foundation/notebooks/backprop.ipynb
jupyter notebook phase1_foundation/notebooks/attention_mechanism.ipynb
```

## Working with This Codebase

### Adding New Educational Implementations (phase1-5)
- Follow the phase structure - place foundational concepts in earlier phases
- Include both class-based and manual implementations for educational comparison
- Add training examples with simple datasets (XOR, OR gates) to verify correctness
- Document activation functions and their derivatives inline
- Maintain "from scratch" philosophy before introducing abstractions

### Working with Sparse MoE Transformer
- Model located in `sparse_moe_transformer/` directory
- Main script: `src/train.py` 
- Test suite: `tests/test_sparse_activation.py`
- Configuration via `Config` dataclass in train.py
- Checkpoints saved to `output/` directory

### Testing & Validation
- Educational: Run with toy problems (XOR, OR) to verify gradient descent
- Production: Use `test_sparse_activation.py` to verify expert routing and sparsity
- Expected output: "Epoch X, Loss: Y" format every N epochs
- Sparse MoE should show 87.5% sparsity (2/16 experts active)

## File Naming Conventions
- `*_from_scratch.py` - Pure NumPy implementations without ML frameworks
- `*.ipynb` - Interactive exploration and visualization
- Empty files represent planned future implementation areas

## Key Learning Objectives
This codebase prioritizes **understanding over production readiness**. When contributing:
- Favor clarity and explicit operations over optimization
- Include comments explaining mathematical concepts
- Show intermediate steps rather than hiding them in abstractions
- Compare manual implementations with class-based versions
