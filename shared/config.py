#!/usr/bin/env python3
"""
Configuration for Sparse MoE Transformer model and training.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for Sparse MoE Transformer model and training.
    
    This dataclass holds all hyperparameters. Key design decisions:
    - Sparse MoE saves ~87.5% FFN compute (only 2/16 experts active)
    - Scaled for 4GB GPU (Quadro T1000, RTX 3050, etc.)
    - Achieves 158M params but runs efficiently due to sparsity
    - Standard backpropagation with Adam optimizer
    """
    
    # Model Architecture
    vocab_size: int = 50257    # GPT-2 tokenizer vocabulary size (must match!)
    d_model: int = 512         # Embedding dimension (larger = more capacity)
    n_heads: int = 8           # Multi-head attention heads (must divide d_model evenly)
    d_ff: int = 1024           # FFN hidden dimension (typically 4x d_model, reduced for memory)
    n_layers: int = 6          # Transformer layers (depth = better reasoning)
    n_experts: int = 16        # Total expert networks in MoE (more = more specialization)
    top_k: int = 2             # Active experts per token (2/16 = 87.5% sparsity)
    dropout: float = 0.1       # Regularization to prevent overfitting
    max_len: int = 256         # Maximum sequence length (longer = more context)

    # Training Hyperparameters (optimized for 4GB VRAM)
    batch_size: int = 4        # Physical batch size (small to fit in memory)
    grad_acc_steps: int = 8    # Gradient accumulation steps (effective batch = 4*8 = 32)
    lr: float = 3e-4           # Learning rate (standard for transformers)
    warmup_steps: int = 200    # Linear warmup to stabilize early training
    max_steps: int = 5000      # Total training steps (66K = 1 full epoch on TinyStories)
    weight_decay: float = 0.01 # L2 regularization (prevents large weights)
    clip_grad: float = 1.0     # Gradient clipping to prevent exploding gradients

    # Data
    dataset_name: str = "roneneldan/TinyStories"
    split: str = "train"
    val_split: str = "validation"

    # Benchmark
    benchmark_gsm8k: bool = False  # Disabled - requires generate() method implementation
    benchmark_latency_samples: int = 128
    profile_flops: bool = True

    # I/O
    output_dir: str = "output/sparse-moe-transformer"
    log_wandb: bool = False
    seed: int = 42
    
    # BDH (Biological) Parameters
    hebbian_lr: float = 0.001         # Hebbian learning rate (local updates) - conservative default
    learning_mode: str = 'backprop'   # 'backprop', 'hebbian', or 'hybrid'
    prune_every: int = 1000           # Apply biological pruning every N steps
    prune_threshold: float = 0.01     # Threshold for activity-dependent pruning
    homeostatic_scaling: bool = True  # Enable homeostatic plasticity


# =============================================================================
# MAMBA CONFIGURATIONS
# =============================================================================

@dataclass
class MambaConfig:
    """Configuration for Mamba/Mamba2 models."""
    from typing import Literal
    
    # Model architecture
    model_type: str = "mamba"  # "mamba" or "mamba2"
    d_model: int = 512  # Hidden dimension
    n_layers: int = 24  # Number of Mamba blocks
    vocab_size: int = 50257  # GPT-2 tokenizer vocab size
    
    # State space parameters
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4  # Convolution kernel size
    expand: int = 2  # Expansion factor for FFN
    
    # Training
    max_len: int = 256  # Maximum sequence length
    batch_size: int = 4
    grad_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 5000
    warmup_steps: int = 500
    
    # Evaluation
    eval_every: int = 500
    save_every: int = 1000
    generate_every: int = 1000
    
    # Dataset
    dataset_name: str = "roneneldan/TinyStories"
    cache_dir: str = "data/tinystories"
    
    # Output
    output_dir: str = "output/mamba"
    
    # Hardware
    device: str = "cuda"
    dtype: str = "float32"
    seed: int = 42
    
    def __post_init__(self):
        """Calculate effective batch size."""
        self.effective_batch_size = self.batch_size * self.grad_accumulation_steps


@dataclass
class Mamba130MConfig(MambaConfig):
    """Preset for ~130M parameter Mamba model."""
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    expand: int = 2


@dataclass
class Mamba30MConfig(MambaConfig):
    """
    Preset for ~30M parameter Mamba model.
    Comparable to BDH/Sparse MoE transformer (d_model=512, n_layers=6).
    """
    d_model: int = 384
    n_layers: int = 8
    d_state: int = 16
    expand: int = 2


@dataclass
class Mamba2_130MConfig(MambaConfig):
    """Preset for ~130M parameter Mamba2 model."""
    model_type: str = "mamba2"
    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    expand: int = 2
