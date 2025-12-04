# Meta-Architecture Vision: Mixture of Learning Paradigms

**Date**: December 4, 2025  
**Status**: Phase 1 In Progress - Hymba Hybrid Training  
**Repository**: `llm-foundation/models/` (refactored structure)

## Overview

A novel framework for composing different neural architectures as modular experts within a unified system. Each expert can use different architectures (transformers, state space models, etc.) while contributing to a shared task via learned routing.

## Current Status

### Validated Models (Phase 1 In Progress)
| Model | Params | Val Loss | Status | Location |
|-------|--------|----------|--------|---------|
| Sparse MoE Transformer | 158M (20M active) | 2.2 | ✅ Working | `models/sparse_moe/` |
| BDH Transformer | 35M | 1.28 | ✅ Working | `models/bdh/` |
| **Hymba Hybrid** | 33.5M | 3.06 (step 400) | 🔄 Training | `models/hybrid/` |
| **Jamba Hybrid** | 28.8M | - | ⏳ Queued | `models/hybrid/` |
| Mamba SSM (pure) | 30M | ⏸️ Skipped | Too slow (no CUDA) | `models/mamba/` |

### Key Findings
- ✅ **Sparse MoE**: Effective for scaling (87.5% sparsity, 16 experts, top-2 routing)
- ✅ **BDH Transformer**: Solid baseline with backprop-only training (val loss 1.28)
- ❌ **Hebbian Learning**: Failed experiment - see [experiments/2025-12-03_hebbian_learning.md](experiments/2025-12-03_hebbian_learning.md)
- ⏸️ **Pure Mamba**: Too slow without optimized CUDA kernels (119h ETA), skipped
- 🔄 **Hymba Hybrid**: Training in progress - learning 62% attention / 38% SSM mixing ratio
- ⏳ **Jamba Hybrid**: Queued after Hymba completes

## Core Concept

Traditional Mixture-of-Experts (MoE) uses multiple networks with the **same architecture** and **same learning rule**. Our vision extends this to a **Mixture of Architectures** where:

- Each expert has a **different architecture** (transformer, Mamba SSM, etc.)
- A meta-router learns **which architecture for which input**
- Experts can be added/removed modularly

## Architecture Diagram

```
User Input
    ↓
[Token Embedding]
    ↓
[Meta-Router] ← Learns input → architecture mapping
    ↓
┌─────────────────────────────────────────────────────────┐
│ Expert 1: Standard Transformer (Backprop)               │
│   - Dense or Sparse MoE FFN                             │
│   - Global gradient optimization                        │
│                                                         │
│ Expert 2: Hymba Hybrid (Parallel Attention + SSM)       │
│   - Learned mixing: α·Attn + (1-α)·Mamba                │
│   - Local precision + Global summarization              │
│                                                         │
│ Expert 3: Jamba Hybrid (Interleaved Layers)             │
│   - Attn → Mamba → Attn → Mamba (alternating)           │
│   - Simpler than Hymba, still effective                 │
└─────────────────────────────────────────────────────────┘
    ↓
[Weighted Combination] ← Soft or hard routing
    ↓
[Output Head]
```

### Hymba Architecture Detail (NEW)
```
Input x
    ↓
LayerNorm
    ↓
┌───────────────────┬───────────────────┐
│  Multi-Head Attn  │   Selective SSM   │  ← PARALLEL
│  (local, precise) │   (global, fast)  │
└─────────┬─────────┴─────────┬─────────┘
          │                   │
          └─────→ α·A + (1-α)·S ←─────┘  ← Learned mixing
                     ↓
                   + x (residual)
                     ↓
                LayerNorm → FFN → + x
                     ↓
                  Output
```

## Implementation Phases

### Phase 1: Validate Expert Architectures ✅ Complete

**Goal**: Implement and validate individual expert architectures on TinyStories.

**Completed**:
- ✅ Sparse MoE Transformer (158M params, val loss 2.2)
- ✅ BDH Transformer (35M params, val loss 1.28)
- ✅ Mamba SSM implementation (30M params, training in progress)
- ❌ Hebbian learning experiment - see [detailed results](experiments/2025-12-03_hebbian_learning.md)

**Files**:
- `models/sparse_moe/model.py` - Sparse MoE architecture
- `models/bdh/model_bdh.py` - BDH transformer
- `models/mamba/model_mamba.py` - Pure PyTorch Mamba
- `models/hybrid/model_hymba.py` - **NEW**: Hymba + Jamba hybrids

### Phase 2: Expert Interface Standardization (Adapter-Based)

**Goal**: Define common interface using **adapter modules** as lightweight, task-specific experts.

**Why Adapters?** (from literature)
- Only ~1% of parameters updated per task (>99% frozen base model)
- Almost entirely eliminates catastrophic forgetting
- Parameter-efficient: easy to add new experts without memory explosion
- Production-proven in RAG systems for continuous adaptation

**Design**:
```python
class AdapterExpert(nn.Module):
    """Lightweight adapter inserted into frozen base model"""
    
    def __init__(self, base_model, adapter_dim=64):
        self.base = base_model  # Frozen
        self.adapter = nn.Sequential(
            nn.Linear(base_model.hidden_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, base_model.hidden_dim)
        )  # Trainable
    
    def forward(self, x):
        base_out = self.base(x)  # Frozen forward
        return base_out + self.adapter(base_out)  # Adapter residual

class ExpertBase(nn.Module):
    """Base class for all expert architectures"""
    
    def forward(self, x, labels=None) -> dict:
        """Returns: {"loss": Tensor, "logits": Tensor, "hidden": Tensor}"""
        raise NotImplementedError
    
    def get_expert_type(self) -> str:
        """Returns: 'transformer', 'ssm', 'hybrid', 'adapter'"""
        raise NotImplementedError
    
    def freeze_base(self):
        """Freeze base model, keep adapters trainable"""
        pass
```

**Success Criteria**:
- Current SparseMoETransformer wrapped as expert
- BDH model implements interface
- Both trainable through common framework

### Phase 3: Meta-Router Implementation (ATLAS-Inspired)

**Goal**: Learn dynamic routing with **weighted knowledge fusion** across architectures.

**Key Insight from ATLAS**: Don't just pick one expert - learn to **blend multiple experts** with learned coefficients via cross-attention.

**Design**:
```python
class MetaArchitectureMoE(nn.Module):
    def __init__(self, config, expert_types):
        self.experts = nn.ModuleList([create_expert(t, config) for t in expert_types])
        self.router = LearnedRouter(config)
        # ATLAS-style knowledge coefficients
        self.knowledge_weights = nn.Parameter(torch.ones(len(expert_types)) / len(expert_types))
    
    def forward(self, x, labels=None):
        # Compute routing weights (can be soft or hard)
        routing_logits = self.router(x)  # [batch, seq, num_experts]
        routing_weights = F.softmax(routing_logits, dim=-1)
        
        # Each expert processes input
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(x, labels)
            # Weight by both router decision AND learned knowledge coefficient
            weighted_out = out['hidden'] * routing_weights[..., i:i+1] * self.knowledge_weights[i]
            expert_outputs.append(weighted_out)
        
        # Fuse outputs (soft combination)
        fused = sum(expert_outputs)
        return self.output_head(fused)
```

**Routing Strategies**:
1. **Soft routing**: Weighted combination of all experts (ATLAS-style)
2. **Hard routing**: Top-k expert selection (Switch Transformer)
3. **Task-based**: Route by detected task type
4. **Learned specialization**: Let experts self-organize
5. **Orthogonal regularization**: Reduce interference between experts (ATLAS)

### Phase 4: Continual Learning with EWC Regularization

**Goal**: Enable learning new tasks without catastrophic forgetting.

**The Problem**: Weight updates for new tasks overwrite parameters essential for old tasks.

**Solution 1: Elastic Weight Consolidation (EWC)**
- Identify critical parameters using Fisher Information matrix
- Add penalty term discouraging changes to important weights
- Validated for sequential learning of 10+ tasks

**Design**:
```python
def train_with_ewc(model, new_task_data, old_fisher_info, ewc_lambda=1000):
    """Train on new task while preserving old knowledge"""
    for batch in new_task_data:
        # Standard task loss
        loss = model(batch).loss
        
        # EWC penalty: discourage changing important parameters
        ewc_loss = 0
        for name, param in model.router.named_parameters():
            if name in old_fisher_info:
                fisher = old_fisher_info[name]
                old_param = old_params[name]
                ewc_loss += (fisher * (param - old_param)**2).sum()
        
        total_loss = loss + ewc_lambda * ewc_loss
        total_loss.backward()
        optimizer.step()

def compute_fisher_information(model, task_data):
    """Estimate parameter importance using Fisher Information diagonal"""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for batch in task_data:
        model.zero_grad()
        loss = model(batch).loss
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2
    return {n: f / len(task_data) for n, f in fisher.items()}
```

**Combined Strategy for Continual Learning**:
1. **Adapters** (Phase 2): Isolate task-specific knowledge in separate modules
2. **EWC**: Protect router parameters that govern expert activation
3. **ATLAS fusion** (Phase 3): Dynamically weight expert contributions
4. **Progressive expansion**: Add new expert modules for new domains

### Phase 5: Knowledge Distillation & Scaling

**Goal**: Efficiently create reasoning-capable experts via distillation from larger models.

**Why Distillation-First?** (from literature)
- Training from scratch is resource-intensive and unstable
- Distilled hybrids match teacher performance with half the attention layers
- Distilling "rationales" (reasoning chains) improves sample efficiency
- Distilled Mamba models achieve MT-Bench scores comparable to larger teachers

**Strategy**:
```python
def distill_expert(teacher_model, student_expert, data, temp=2.0):
    """Distill reasoning capability from large teacher to small expert"""
    for batch in data:
        with torch.no_grad():
            # Get teacher's output AND intermediate reasoning
            teacher_logits = teacher_model(batch).logits
            teacher_hidden = teacher_model.get_hidden_states(batch)
        
        # Student learns to match both output and reasoning process
        student_out = student_expert(batch)
        
        # KL divergence on output distribution
        kl_loss = F.kl_div(
            F.log_softmax(student_out.logits / temp, dim=-1),
            F.softmax(teacher_logits / temp, dim=-1),
            reduction='batchmean'
        )
        
        # Hidden state matching (reasoning transfer)
        hidden_loss = F.mse_loss(student_out.hidden, teacher_hidden)
        
        loss = kl_loss + 0.5 * hidden_loss
        loss.backward()
```

**Recommended Teacher Models**:
- Llama 3 8B (open-source, strong reasoning)
- Mistral 7B (efficient, good balance)
- GPT-2 XL 1.5B (budget-friendly baseline)

**Expanded Expert Library** (future):
- **Kolmogorov-Arnold Networks (KAN)**: Learnable activation functions
- **Liquid Neural Networks**: Time-continuous dynamics
- **Neural ODEs**: Continuous-depth models
- **Spiking Neural Networks**: Event-based computation

## Strategic Experimental Roadmap

### Core Hypothesis
> A composition of different architectures is **superior** to any single architecture for reasoning tasks.

### Key Experiments

#### 1. Task-Specific Expert Identification
Train MoE with specialized experts:
- **Transformer expert**: Retrieval-heavy tasks (exact copying, QA)
- **SSM expert**: Summarization, long-context compression
- **Hybrid expert**: Mixed reasoning tasks

**Metric**: MoE performance vs best single expert across task types.

#### 2. Catastrophic Forgetting Resistance
- Train system on Task A → measure performance
- Fine-tune on Task B → re-test Task A
- Compare: Adapters + EWC vs naive fine-tuning

**Metric**: Task A accuracy retention after learning Task B.

#### 3. Synergistic Benefit Analysis
Design tasks that require **multiple capabilities**:
- Long-context retrieval + creative synthesis
- Sequential pattern recognition + precise fact recall
- Example: "Summarize this 10K token document and answer specific questions"

**Hypothesis**: MoE learns to engage both Transformer (retrieval) and SSM (summary) experts.

#### 4. Addressing Known Weaknesses
Test if MoE mitigates Mamba's copying limitation:
- Perfect copying benchmark (100-1000+ tokens)
- Retrieval accuracy vs context length

**Hypothesis**: Router activates "precise retrieval" Transformer expert for copying tasks.

### Success Criteria
| Experiment | Success Threshold |
|------------|-------------------|
| Task specialization | MoE ≥ best single expert on 80%+ tasks |
| Forgetting resistance | <10% accuracy drop on old tasks |
| Synergy | MoE outperforms all singles on composite tasks |
| Weakness mitigation | Near-perfect copying via expert routing |

## Research Questions

### Specialization
- Do experts naturally specialize to task types?
- Can we predict which expert for which input?
- Does routing correlate with human intuitions?

### Learning Efficiency
- Do biological learning rules improve sample efficiency?
- Can Hebbian experts adapt faster to distribution shifts?
- Does mixing paradigms improve generalization?

### Emergent Behavior
- Do experts develop complementary representations?
- Can the system discover novel combinations?
- Does competition between experts improve overall performance?

### Computational Trade-offs
- Memory: Hebbian (no optimizer states) vs Backprop (needs momentum/variance)
- Compute: Local updates (Hebbian) vs Global gradients (Backprop)
- Convergence: Which learning rule converges faster?

## Potential Applications

### Scientific Computing
- **BDH Expert**: Fast adaptation to new physics
- **Standard Expert**: Precise numerical predictions
- **State Space Expert**: Long time-series modeling

### Biomedical
- **Hebbian Expert**: Biological plausibility for brain models
- **Standard Expert**: Diagnostic accuracy
- **Hybrid**: Interpretable + accurate

### Industrial (Borregaard Use Case)
- **Expert 1**: Chemical property prediction (Hebbian - fast adaptation)
- **Expert 2**: Process optimization (Backprop - precise control)
- **Expert 3**: Documentation generation (Standard transformer)
- **Router**: Learns task → expert mapping from data

### Continual Learning
- Hebbian experts handle new tasks without catastrophic forgetting
- Standard experts maintain stable base knowledge
- Router balances plasticity vs stability

## Technical Challenges

### 1. Training Stability
- Different learning rates per expert
- Balancing expert utilization
- Preventing collapse to single expert

### 2. Routing Learning
- How to train router with mixed learning rules?
- Should router use same rule as experts or always backprop?
- Exploration vs exploitation in routing

### 3. Computational Efficiency
- All experts forward pass vs selective activation
- Memory overhead of multiple architectures
- Parallelization across different expert types

### 4. Evaluation
- How to measure contribution of each expert?
- Metrics for specialization
- Comparing across learning paradigms

## Success Metrics

### Phase 1 (Individual Experts) ✅ Complete
- [x] Sparse MoE trains to convergence on TinyStories (val loss 2.2)
- [x] BDH transformer achieves strong baseline (val loss 1.28)
- [x] Mamba implementation working (training in progress)
- [x] Memory usage acceptable on 4GB GPU
- [x] Hebbian learning evaluated (failed - see experiment log)

### Phase 2-3 (Meta-Architecture)
- [ ] Multiple expert types trainable together
- [ ] Router learns meaningful specialization
- [ ] Combined performance ≥ best individual expert
- [ ] Expert utilization balanced (no collapse)

### Phase 4-5 (Full System)
- [ ] 3+ different architectures working together
- [ ] Interpretable routing decisions
- [ ] Modular addition of new expert types

## Repository Structure

```
llm-foundation/
├── models/
│   ├── sparse_moe/     # 158M Sparse MoE Transformer
│   ├── bdh/            # 35M BDH Transformer  
│   ├── mamba/          # 30M Mamba SSM (pure, slow)
│   └── hybrid/         # NEW: Hymba + Jamba hybrids
│       ├── model_hymba.py   # HymbaLM (parallel) + JambaLM (interleaved)
│       ├── train.py         # Training script
│       └── generate.py      # Text generation
├── shared/
│   ├── config.py       # All model configs
│   ├── dataset.py      # TinyStories loading
│   └── utils.py        # Common utilities
├── data/tinystories/   # Cached tokenized data
├── output/             # Checkpoints per model
└── papers_and_notes/
    ├── meta_architecture_vision.md  # This file
    └── experiments/    # Detailed experiment logs
```

## Experiment Log

All detailed experiment results are in `experiments/`:
- [2025-12-03: Hebbian Learning](experiments/2025-12-03_hebbian_learning.md) - ❌ Failed

## Literature Support

Recent research strongly validates our hybrid approach:

### Architecture Comparison (from literature)

| Feature | Transformer | Mamba (SSM) | RWKV (RNN) | Hybrid (Hymba/Jamba) |
|---------|-------------|-------------|------------|----------------------|
| **Complexity** | O(n²) | O(n) | O(n) | Sub-quadratic |
| **Inference Speed** | Baseline | 5-8x faster | Very fast | Up to 8x faster |
| **Context Length** | Limited by KV cache | Efficient, no KV cache | "Infinite", no KV cache | 256K+ efficient |
| **Perfect Copying** | ✅ Excellent | ❌ Fails on long strings | N/A | ✅ If attention present |
| **Retrieval Accuracy** | ✅ Stable | ❌ Degrades with length | N/A | ✅ Strong |
| **Reasoning** | General purpose | Weak retrieval, strong induction | Prompt-sensitive | Often superior |

### Key Research Findings

**Hymba** (parallel attention + SSM):
- Hymba-125M achieved 49.35% accuracy vs Mamba-130M at 42.43%
- Proves hybrid > pure architectures even at small scale
- Combines attention precision with SSM efficiency

**Jamba** (interleaved layers):
- Sub-quadratic complexity, 256K context windows
- 8B hybrid Mamba-2 exceeded Transformer by +2.65 points average

**Critical SSM Limitation**:
> "Any model with fixed-size memory cannot copy random strings longer than its memory capacity"
- Transformers learn to copy 1000+ token strings; Mamba fails
- Retrieval accuracy degrades significantly with context length
- This is why hybrids keep attention for precision tasks

**Knowledge Distillation**:
- Distilled hybrid Mamba models match/exceed Transformer teachers
- Even with only half the attention layers
- Reasoning patterns successfully transfer

**Key insight**: Transformer handles **local, fine-grained** interactions while Mamba handles **global summarization** efficiently.

## Related Work

- **Switch Transformer** (Google) - Same architecture, sparse routing
- **Mamba** - Selective state space models (arxiv:2312.00752)
- **DeepSeek MoE** - Same architecture, shared experts

### Our Innovation
- **Different**: Mix architectures, not just weights
- **Modular**: Add new paradigms without retraining

## References

### Core Architecture Papers
- **Mamba (SSM)**: https://arxiv.org/abs/2312.00752
- **Switch Transformers (MoE)**: https://arxiv.org/abs/2101.03961
- **Hymba (Hybrid-head SLM)**: https://www.jankautz.com/publications/Hymba_ICLR25.pdf
- **BlackMamba (MoE + SSM)**: https://arxiv.org/html/2402.01771v1
- **RWKV Survey**: https://arxiv.org/html/2412.14847v1

### Hybrid Architectures
- **Hybrid Neural Architecture Search**: https://utsysml.ece.utexas.edu/publications/prints/arXiv2025_sinha.pdf
- **Rise of Hybrid LLMs (AI21)**: https://www.ai21.com/blog/rise-of-hybrid-llms/
- **Mamba-in-Llama Distillation (Together AI)**: https://www.together.ai/blog/the-mamba-in-the-llama-distilling-and-accelerating-hybrid-models
- **Hybrid/Ensemble Deep Learning Review**: https://arxiv.org/html/2312.05589v1
- **Extremely Efficient Hybrid Models (AMD)**: https://rocm.blogs.amd.com/artificial-intelligence/hybrid-models,-mla,/README.html

### State Space Models & Alternatives
- **State Space Models Primer**: https://aman.ai/primers/ai/state-space-models/
- **Awesome SSM Resources**: https://github.com/AvivBick/awesome-ssm-ml
- **Official Mamba Implementation**: https://github.com/state-spaces/mamba
- **Transformers vs SSMs at Copying**: http://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/
- **Stuffed Mamba (State Size Limitations)**: https://openreview.net/forum?id=CdRauNXD1w

### Mixture of Experts
- **MoE Comprehensive Survey**: https://arxiv.org/html/2503.07137v1
- **MoE Evolution Survey**: https://www.preprints.org/manuscript/202408.0583
- **Conditional Computation (Cameron Wolfe)**: https://cameronrwolfe.substack.com/p/conditional-computation-the-birth

### Continual Learning & Catastrophic Forgetting
- **EWC (Elastic Weight Consolidation)**: https://www.pnas.org/doi/10.1073/pnas.1611835114
- **Catastrophic Forgetting Survey**: https://arxiv.org/html/2403.05175v1
- **ATLAS (Adapter-based Continual Learning)**: https://arxiv.org/html/2410.10923v1
- **Continual Lifelong Learning Review**: https://www.sciencedirect.com/science/article/pii/S0893608019300231
- **IBM: What is Catastrophic Forgetting**: https://www.ibm.com/think/topics/catastrophic-forgetting

### Small Language Models
- **SLM Comprehensive Survey (ACM)**: https://dl.acm.org/doi/full/10.1145/3768165
- **SLM Survey (arXiv)**: https://arxiv.org/html/2410.20011v1
- **SLM Survey (TechRxiv)**: https://www.techrxiv.org/doi/full/10.36227/techrxiv.175994221.11990581/v1
- **Small LMs for Architecture Experiments**: https://discuss.huggingface.co/t/small-lms-to-prototype-architecture-experiments-on/137438

### Few-Shot & Parameter-Efficient Fine-Tuning
- **Few-Shot PEFT is Better**: https://papers.nips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html
- **Power of Adapters in LLMs**: https://medium.com/@zbabar/the-power-of-adapters-in-fine-tuning-llms-722c87c5bca6
- **Making PLMs Better Few-Shot Learners**: https://aclanthology.org/2021.acl-long.295/

### RWKV
- **RWKV Official**: https://www.rwkv.com/
- **RWKV GitHub**: https://github.com/BlinkDL/RWKV-LM
- **RWKV-6 Analysis**: https://medium.com/@bnjmn_marie/rwkv-6-attention-free-and-state-of-the-art-7b-llm-320720df3c8c

### Reasoning & Chain-of-Thought
- **CoT Reasoning Survey (ACL 2024)**: https://github.com/zchuz/CoT-Reasoning-Survey
- **Baby Dragon Hypothesis**: https://arxiv.org/abs/2509.26507

### Benchmarks & Comparisons
- **GSM8k Leaderboard**: https://llm-stats.com/benchmarks/gsm8k
- **Mamba vs Transformer Comparison**: https://github.com/BiroAd/compare_llms
- **RankMamba Document Ranking**: https://arxiv.org/html/2403.18276v1
- **DBRX vs Jamba vs Qwen Comparison**: https://www.linkedin.com/pulse/efficiency-meets-performance-comparing-open-source-llms-vajariya-79zye

### Industry Analysis
- **2024 Post-Transformers (Latent Space)**: https://www.latent.space/p/2024-post-transformers
- **Do We Need Attention? (Interconnects)**: https://www.interconnects.ai/p/llms-beyond-attention
- **Transformer Alternatives (GreyB)**: https://xray.greyb.com/artificial-intelligence/alternatives-to-transformer-architecture
- **Who Will Replace Transformers**: https://docs.d.run/en/blogs/2024/0327-transformer.html

### Hardware & Implementation
- **In-Situ Backpropagation (Science)**: https://www.science.org/doi/10.1126/science.ade8450
- **Falcon3-Mamba-7B**: https://huggingface.co/tiiuae/Falcon3-Mamba-7B-Base
- **Mamba Training (LightOn)**: https://www.lighton.ai/lighton-blogs/passing-the-torch-training-a-mamba-model-for-smooth-handover
