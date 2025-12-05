"""
Mamba State Space Model implementations.

Provides two implementations:
1. MambaLM - Pure PyTorch (educational, works everywhere)
2. FastMambaLM - CUDA-optimized using mamba-ssm (10-50x faster, Linux/WSL only)

Usage:
    # Pure PyTorch (understanding SSM)
    from models.mamba import MambaLM
    
    # CUDA-optimized (fast training)
    from models.mamba import FastMambaLM, create_mamba_model
"""
from .model_mamba import MambaLM, load_pretrained_mamba, PretrainedMambaWrapper

# Try to import fast implementation (requires mamba-ssm)
try:
    from .model_mamba_fast import (
        FastMambaLM, 
        FastMambaBlock, 
        FastMamba2Block,
        create_mamba_model,
        benchmark_mamba_implementations,
        MAMBA_SSM_AVAILABLE,
    )
except ImportError:
    MAMBA_SSM_AVAILABLE = False
    FastMambaLM = None
    create_mamba_model = None
