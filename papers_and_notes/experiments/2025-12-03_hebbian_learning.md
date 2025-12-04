# Hebbian Learning Experiment in Transformers

**Date**: December 3, 2025  
**Status**: ❌ FAILED  
**Model**: BDH-SLIM Transformer (64.2M params)  
**Dataset**: TinyStories (2.1M train, 22K val)

## Goal

Determine if Hebbian learning (∆w = η·x·y) helps or hurts transformer language modeling when applied simultaneously with backpropagation.

## Hypothesis

Biological learning mechanisms might complement gradient-based learning:
- **Best case**: Hybrid converges faster and generalizes better
- **Realistic case**: Hybrid ≈ Backprop-only (Hebbian as regularization)
- **Worst case**: Hebbian hurts performance

## Methodology

### Model Architecture
```
d_model = 512
n_heads = 8
d_ff = 1024
n_layers = 6
max_len = 256
dropout = 0.1
Total params: 64.2M
```

### Training Modes
1. **Backprop-only** (`--learning_mode backprop`): Standard gradient descent
2. **Hybrid** (`--learning_mode hybrid`): Backprop + Hebbian updates
3. **Hebbian-only** (`--learning_mode hebbian`): Skipped after hybrid failure

### Training Config
```
batch_size = 4
grad_acc_steps = 8 (effective batch = 32)
lr = 3e-4 (backprop learning rate)
hebbian_lr = 0.001 (Hebbian learning rate)
max_steps = 5000
eval_every = 500
GPU: Quadro T1000 4GB
```

### Biological Mechanisms
```
prune_every = 1000 steps
prune_threshold = 0.01
homeostatic_scaling = True
weight_decay = 0.9999 (applied to Hebbian weights)
```

## Results

### Quantitative Comparison

| Metric | Backprop-only | Hybrid | Winner |
|--------|--------------|--------|--------|
| **Best Val Loss** | **1.2787** | 3.8957 | ✅ Backprop |
| **Steps to Best** | 4000 | 3500 | - |
| **Loss Reduction** | 87% (10.7→1.28) | 64% (10.7→3.9) | ✅ Backprop |
| **Training Stability** | Stable | Overfit after 3500 | ✅ Backprop |
| **Text Coherence** | ✅ High | ❌ Incoherent | ✅ Backprop |
| **Grammar** | ✅ Proper | ❌ Broken | ✅ Backprop |
| **Biological Sparsity** | 8.26% | 0.00% | ✅ Backprop |

**Verdict**: Backprop-only achieves **67% lower loss** (3× better perplexity).

### Training Curves

**Backprop-only** (output/bdh_backprop_5k/):
```
Step    Val Loss    Notes
500     2.1419      Strong initial learning
1000    1.7619      Steady improvement
1500    1.5748      Approaching convergence
2000    1.4642      
2500    1.3890      
3000    1.3351      
3500    1.3032      
4000    1.2787      Best checkpoint (stopped here)
```

**Hybrid** (output/bdh_hybrid_5k/):
```
Step    Val Loss    Notes
500     5.4532      Poor initial learning
1000    4.4821      Slow improvement
1500    3.9390      
2000    3.9110      
2500    3.9010      
3000    3.8990      
3500    3.8957      Best checkpoint
4000    3.9079      Starting to overfit
4500    4.0390      Clear overfitting
5000    4.2777      Final (degraded)
```

### Text Generation Comparison

**Prompt**: "Once upon a time"

**Backprop-only output** (coherent):
```
Once upon a time, there was a little girl named Lily. She loved to play 
outside in the garden with her friends. One day, she found a big box of 
toys and went inside to find them. But when she looked for them, she saw 
that they were very messy.

Lily's mom told her not to worry because they had to clean up their toys 
so they could wash them off together. They cleaned up their room, and even 
helped make sure it was safe again.

After a while, Lily and her friends went home happy and hugged each other 
tightly. They learned that sometimes things can be good and important to 
keep our favorite things tidy.
```
✅ Proper narrative structure, coherent grammar, logical progression

**Hybrid output** (broken):
```
Once upon a time.

 She's they to the her and.  They her in with the was, and had a he and 
on day very it for of to.One with so very the said is't and. Lily the mom. 
The little He said his he. One to up a,. It she friends, a there. It day 
the,. The day that, her and day a were of and a and are happy.

 " went and said could it to the little girl and.

 girl play them the it.

 girl go. He was!
" they was. The a. He with, was at and it. But. Suddenly amy, was. He's!" 
I, and he. Tim was to." When to her wanted. They. Tim.
```
❌ Completely incoherent word salad

## Analysis: Why Hebbian Learning Failed

### 1. Conflicting Objectives
- **Backprop**: Minimizes global loss via chain rule (∂L/∂W)
- **Hebbian**: Maximizes local correlation (Δw ∝ x·y)
- These objectives are **not aligned** for language modeling
- Weight updates fight each other

### 2. Weight Interference
```
W_new = W_old - η·∂L/∂W + η_hebb·(y·x^T)
         ↑                  ↑
   Global error signal   Local correlation
```
- Same weights updated by both mechanisms
- Hebbian "undoes" backprop's careful gradient steps

### 3. Information Flow Disruption
- Transformers rely on precise attention patterns
- Query-Key alignment must be learned globally
- Hebbian correlation doesn't capture task-specific attention needs

### 4. Output Layer Catastrophe
- Backprop learns: "boost probability of correct next token"
- Hebbian learns: "strengthen connections for active neurons"
- Active neurons ≠ correct tokens!
- Result: Output distribution becomes uniform noise

### 5. No Error Signal Propagation
- Hebbian is purely local: each synapse only "knows" pre/post activity
- No information about prediction correctness
- Can't learn task-specific representations

### 6. Biological Mechanisms Ineffective
- Activity-dependent pruning: 0% sparsity (vs 8% for backprop)
- Homeostatic scaling: Ineffective at stabilizing
- These mechanisms need different context to be useful

## Why Hebbian Works in Biology But Not Here

**Biology**:
1. Used during development, not all learning
2. Combined with dopamine (reward), inhibition, attention
3. Sparse activity: neurons fire ~1 Hz, not every forward pass
4. STDP considers spike timing, not just correlation
5. Structural plasticity over days/weeks, not every batch

**Transformers**:
1. Applied every batch (wrong timescale)
2. No complementary signals (reward, neuromodulation)
3. Dense activity: every neuron active every forward pass
4. Global attention, not local receptive fields

## Possible Future Approaches

If revisiting Hebbian learning:
1. **Hebbian only during pretraining**: Initial weight structure, then backprop finetune
2. **Hebbian only in attention**: Apply to Q,K,V but not FFN/output
3. **Contrastive Hebbian**: ∆w ∝ x_pos·y_pos - x_neg·y_neg
4. **BCM rule**: Sliding threshold for potentiation vs depression
5. **Predictive Hebbian**: ∆w ∝ x·(y - ŷ) where ŷ is prediction
6. **Different architecture**: Try on MLPs or RNNs instead of transformers

## Technical Challenges Encountered

### NaN Loss (Resolved)
- **Problem**: Loss immediately NaN with hebbian_lr = 0.01
- **Solution**: Reduced to 0.001, added `torch.clamp(delta_w, -1, 1)`

### Mixed Learning Gradient Flow
- **Solution**: Hebbian updates with `torch.no_grad()`, detach activations

### Checkpoint Saving Errors
- **Problem**: Disk space issues
- **Solution**: Try-catch to make non-fatal

## Conclusion

**Primary Finding**: Naive Hebbian learning is **incompatible** with transformer language modeling when applied simultaneously with backpropagation.

**Implications**:
- ✅ Framework works (can train different modes)
- ✅ Got empirical answer (Hebbian doesn't help - this was the goal!)
- ✅ Infrastructure solid (ready for Phase 2)
- ❌ BDH-SLIM with Hebbian not ready for expert ensemble

**Recommendation**: Skip Hebbian for now, proceed to Phase 2 with standard transformer variants (sparse MoE, dense, distilled). The meta-architecture framework is ready; we just need experts that actually work.

## Files

- Model: `models/bdh/model_bdh.py`
- Hebbian rules: `models/bdh/hebbian.py`
- Training: `models/bdh/train_bdh.py`
- Checkpoints: `output/bdh/bdh_backprop_5k/`, `output/bdh/bdh_hybrid_5k/`

## References

- Baby Dragon Hypothesis: https://arxiv.org/abs/2509.26507
- Hebbian Learning Theory: Hebb, D. O. (1949). "The Organization of Behavior"
- STDP: Bi & Poo (1998). "Synaptic modifications in cultured hippocampal neurons"
- BCM Theory: Bienenstock et al. (1982). "Theory for the development of neuron selectivity"
