# llm-foundation

Educational repository for building LLM understanding from first principles, plus a working Sparse MoE Transformer implementation.

## Repository Structure

### `phase1_foundation/` - Learning Path
Educational implementations for understanding neural networks and transformers from scratch:
- **Backpropagation** (`backprop_from_scratch.py`) - Pure NumPy implementation with XOR training
- **Attention Mechanisms** (`attention_mechanism.py`) - Multi-head attention exploration
- Interactive Jupyter notebooks for hands-on learning

**Purpose**: Build foundational understanding before tackling advanced architectures.

### `sparse_moe_transformer/` - Production Model
A fully functional Sparse Mixture-of-Experts Transformer achieving state-of-the-art results:

**Performance Highlights:**
- 158M parameters, 2.2 validation loss on TinyStories
- 87.5% compute savings through sparse expert routing
- 37ms inference latency on 4GB GPU
- Trains on consumer hardware (Quadro T1000, RTX 3050+)

**Key Files:**
- `src/train.py` - Training script with full implementation
- `tests/test_sparse_activation.py` - Verification suite
- `README.md` - Complete documentation

### `phase2_architecture/` to `phase5_production/`
Planned expansion areas for transformer components, training techniques, and deployment strategies.

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Educational Examples
```bash
# Backpropagation from scratch
python phase1_foundation/src/backprop_from_scratch.py

# Explore in Jupyter
jupyter notebook phase1_foundation/notebooks/backprop.ipynb
```

### Train Sparse MoE Transformer
```bash
cd sparse_moe_transformer
python src/train.py --mode train --output_dir output/my_model

# Benchmark the model
python src/train.py --mode benchmark --model_path output/my_model/best_model.pt
```

## Learning Philosophy

This codebase prioritizes **understanding over production readiness** in the phase directories:
- Favor clarity and explicit operations over optimization
- Include comments explaining mathematical concepts
- Show intermediate steps rather than hiding them in abstractions
- Compare manual implementations with framework versions

The `sparse_moe_transformer/` directory demonstrates production-quality implementations.

## Key Concepts Covered

**Phase 1 - Foundation:**
- Manual gradient computation and backpropagation
- Weight initialization strategies
- Activation functions (tanh, GELU)
- Multi-head attention mechanisms

**Sparse MoE Transformer:**
- Sparse Mixture-of-Experts architecture
- Expert routing with learned gating
- Memory-efficient training on 4GB GPUs
- Load balancing and gradient flow

## Important Note

The Sparse MoE Transformer in this repo is **not** the biologically-inspired "Baby Dragon Hatchling" (BDH) model from arxiv.org/abs/2509.26507. That paper describes Hebbian learning and spiking neurons. Our implementation uses standard transformer architecture with sparse expert routing for computational efficiency.

## Contributing

This is an educational project. Contributions that enhance learning clarity are welcome:
- Improved explanations and comments
- Additional toy problems for validation
- Comparisons between manual and framework implementations
- Documentation of theoretical concepts

## Project Status

- ✅ Phase 1 Foundation: Backpropagation, Attention
- ✅ Sparse MoE Transformer: Fully functional, tested
- 🚧 Phase 2-5: Planned expansion areas

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Sparse MoE at scale
- Project Documentation: See individual directory READMEs

## License

MIT License - See LICENSE file for details
