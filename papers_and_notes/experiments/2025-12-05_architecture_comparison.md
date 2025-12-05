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

### Round 3: Long-Sequence Comparison (Planned)

This is where Mamba should shine. At longer sequences:
- **Transformer**: O(L²) attention becomes bottleneck (memory & compute)
- **Mamba**: O(L) linear scaling, constant memory per token
- **Hymba**: Hybrid - will learned mixing adapt to favor SSM?

**Datasets Added:**
```bash
# Medium-length (1024 tokens) - tests attention scaling
python train.py --dataset wikitext --max_len 1024

# Long-range (2048 tokens) - where Mamba should dominate
python train.py --dataset pg19 --max_len 2048
```

**Expected Observations:**
1. GPU memory will grow ~16x for Transformer (256→1024) but ~4x for Mamba
2. Mamba throughput should stay constant, Transformer will slow dramatically
3. If Hymba is smart, its learned mixing should shift toward SSM at longer sequences

**Hardware Constraints (Quadro T1000 4GB):**
- 1024 tokens: May need batch_size=2
- 2048 tokens: May need batch_size=1 + gradient accumulation
- Consider using gradient checkpointing if OOM

### Round 4: Scaling (Future)

1. **125M+ parameters**
   - Do architectural differences grow or shrink?

2. **Task-specific evaluation**
   - Long-range retrieval tasks (copy, associative recall)
   - Multi-turn dialogue

### Future Improvement: RoPE

Current limitation: learned positional embeddings are fixed to max_len.
Training on 256 tokens means model can't handle longer sequences at inference.

**RoPE (Rotary Position Embeddings):**
- Encodes position via rotation matrix applied to Q and K
- No learned position embedding table
- Scales to arbitrary lengths without retraining
- Used by: LLaMA, Mistral, Qwen, most modern LLMs

**Implementation:** Apply rotation to each attention head:
```python
# RoPE pseudocode
cos_m, sin_m = get_rotary_embeddings(position_indices)
q = apply_rotary(q, cos_m, sin_m)
k = apply_rotary(k, cos_m, sin_m)
```

Will implement in Phase 2 if long-sequence experiments show promise.