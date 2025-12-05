# Architecture Comparison: Transformer vs Mamba vs Hymba

**Date**: 2025-12-05  
**Status**: ✅ Round 1 Complete, Round 2 In Progress

## Goal

Compare three architectures at similar parameter counts to understand:
1. Does Mamba (SSM) match Transformer (Attention) quality?
2. Does hybrid (Hymba) combine the best of both?
3. Where do architectural differences matter?

## Round 1 Results (5K steps, TinyStories, 256 tokens)

| Model | Val Loss | Parameters | Layers | Architecture | Speed |
|-------|----------|------------|--------|--------------|-------|
| **Mamba** | **1.25** | 27M | 8 | Pure SSM | 0.54 steps/s |
| **Transformer** | **1.26** | 27M | 8 | Pure Attention | 0.59 steps/s |
| **Hymba** | **1.90** | 30M | 6 | Hybrid Attn+SSM | 0.54 steps/s |

### Configuration Details

**Mamba (mamba-ssm CUDA kernels)**
```
d_model=384, n_layers=8, d_state=16, d_conv=4, expand=2
Checkpoint: models/mamba/output/mamba_30m_fast_5000steps/best_checkpoint.pt
```

**Transformer (Standard GPT-style)**
```
d_model=384, n_layers=8, n_heads=6, d_ff=1536
Checkpoint: models/transformer/output/transformer_30m_5000steps/best_checkpoint.pt
```

**Hymba v1 (Parallel Attention+SSM, UNFAIR - only 6 layers)**
```
d_model=384, n_layers=6, n_heads=6, d_ff=1024, d_state=16
Learned mixing: ~62% attention / 38% SSM
Checkpoint: output/hybrid/hymba_30m_fast_5k/best_checkpoint.pt
```

### Sample Outputs

**Mamba** (val loss 1.25):
> Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily found a shiny gem on the ground. She picked it up and showed it to her mom.

**Transformer** (val loss 1.26):
> Once upon a time, there was a little girl named Lily. She loved to play outside with her friends. One day, she saw a big tree with bright colors. She wanted to fly all the colors, so she went to the tree and found a pretty plant.

**Hymba v1** (val loss 1.90):
> Once upon a time there a little named who to his. lived a boy his and loved play. day day decided go a game game the.

## Analysis - Round 1

1. **Mamba ≈ Transformer** on short sequences
   - Both achieve ~1.25 val loss
   - Text quality is comparable
   - Validates that SSM can match attention for this task

2. **Hymba underperformed** - but unfair comparison!
   - Only 6 layers vs 8 layers (25% fewer)
   - d_ff=1024 vs d_ff=1536 (33% smaller FFN)
   - Need Round 2 with matched architecture

3. **Speed similar** at 256 tokens
   - O(L²) vs O(L) doesn't matter at short sequences
   - Need longer sequences to see Mamba's advantage

## Round 2: Fair Hymba Comparison (In Progress)

**Hymba v2 Config (matched to Mamba/Transformer)**
```
d_model=384, n_layers=8, n_heads=6, d_ff=1536, d_state=16
```

| Model | Val Loss | Status |
|-------|----------|--------|
| Mamba | 1.25 | ✅ Complete |
| Transformer | 1.26 | ✅ Complete |
| Hymba v2 | TBD | 🔄 Training |

## Future Experiments

1. **Longer sequences** (1K-4K tokens with PG-19 or WikiText)
   - Hypothesis: Mamba will outperform on throughput
   - Hypothesis: Hymba mixing will shift toward more SSM

2. **Scaling** (125M+ params)
   - Do architectural differences grow or shrink?

3. **Task-specific evaluation**
   - Long-range retrieval tasks
   - Multi-turn dialogue
