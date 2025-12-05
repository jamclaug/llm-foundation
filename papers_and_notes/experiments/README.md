# Experiments Index

This folder contains detailed experiment logs and results for the Meta-Architecture project.

## Completed Experiments

| Date | Experiment | Result | File |
|------|------------|--------|------|
| 2025-12-03 | Hebbian Learning in Transformers | ❌ Failed | [2025-12-03_hebbian_learning.md](2025-12-03_hebbian_learning.md) |
| 2025-12-05 | Architecture Comparison (Round 1) | ✅ Mamba≈Transformer | [2025-12-05_architecture_comparison.md](2025-12-05_architecture_comparison.md) |

## In Progress

| Experiment | Status | Notes |
|------------|--------|-------|
| Hymba v2 (8 layers) | 🔄 Training | Fair comparison with matched layer count |

## Key Results So Far

| Model | Val Loss | Params | Notes |
|-------|----------|--------|-------|
| Mamba | **1.25** | 27M | Pure SSM, 8 layers |
| Transformer | **1.26** | 27M | Pure Attention, 8 layers |
| Hymba v1 | 1.90 | 30M | Unfair: only 6 layers |

## Planned Experiments

| Experiment | Status | Notes |
|------------|--------|-------|
| Hymba v2 (fair comparison) | 🔄 Next | 8 layers, matched FFN |
| Long sequence comparison | 📋 Planned | PG-19 or WikiText, 1K-4K tokens |
| Sparse MoE vs Dense | 📋 Planned | Same param count |

## Experiment Template

When adding new experiments, include:
1. **Goal**: What hypothesis are we testing?
2. **Methodology**: Model configs, dataset, training setup
3. **Results**: Tables, metrics, sample outputs
4. **Analysis**: Why did it work/fail?
5. **Next Steps**: What to try next based on findings
