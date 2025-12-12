# Experiments Index

This folder contains detailed experiment logs and results for the Meta-Architecture project.

## Completed Experiments

| Date | Experiment | Result | File |
|------|------------|--------|------|
| 2025-12-03 | Hebbian Learning in Transformers | ❌ Failed | [2025-12-03_hebbian_learning.md](2025-12-03_hebbian_learning.md) |
| 2025-12-05 | Architecture Comparison (Round 1) | ✅ Mamba≈Transformer | [2025-12-05_architecture_comparison.md](2025-12-05_architecture_comparison.md) |
| 2025-12-07 | Mamba WikiText-103 (1024 tokens) | ✅ Complete | See below |

## In Progress

| Experiment | Status | Notes |
|------------|--------|-------|
| Transformer WikiText-103 | 🔄 Training | O(L²) comparison at 1024 tokens |
| Hymba WikiText-103 | 📋 Queued | Test mixing weight differentiation |

## Key Results So Far

### Round 1: TinyStories (256 tokens)
| Model | Val Loss | Params | Speed | Notes |
|-------|----------|--------|-------|-------|
| Mamba | **1.25** | 27M | - | Pure SSM, 8 layers |
| Transformer | **1.26** | 27M | - | Pure Attention, 8 layers |
| Hymba v2 | 1.88 | ~30M | - | Static 62% attention all layers |

### Round 2: WikiText-103 (1024 tokens)
| Model | Val Loss | Params | Speed | Training Time | Notes |
|-------|----------|--------|-------|---------------|-------|
| Mamba | **0.508** | 27M | 0.05 steps/s | 27.86h | ✅ Complete |
| Transformer | ? | 159M | 0.02 steps/s | ~77h ETA | 🔄 In progress |
| Hymba | ? | ~30M | ? | ? | 📋 Queued |

**Key Observations:**
- Transformer is **2.5x slower** than Mamba at 1024 tokens (0.02 vs 0.05 steps/s)
- This is O(L²) vs O(L) showing up! At 256 tokens they were similar speed.
- Transformer has 159M params (Sparse MoE) vs Mamba's 27M

## Planned Experiments

| Experiment | Status | Notes |
|------------|--------|-------|
| Hymba WikiText (1024 tokens) | 📋 Next | Test if mixing weights differentiate on long sequences |
| Sparse MoE vs Dense | 📋 Planned | Same param count |

## Experiment Template

When adding new experiments, include:
1. **Goal**: What hypothesis are we testing?
2. **Methodology**: Model configs, dataset, training setup
3. **Results**: Tables, metrics, sample outputs
4. **Analysis**: Why did it work/fail?
5. **Next Steps**: What to try next based on findings
