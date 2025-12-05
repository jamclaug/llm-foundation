"""
Hybrid Transformer-Mamba models.

This module provides:
- HymbaLM: Parallel attention + SSM heads with learned mixing (Hymba-style)
- JambaLM: Interleaved attention and SSM layers (Jamba-style)

Supports two SSM implementations:
- mamba-ssm CUDA kernels (10-50x faster, Linux/WSL only)
- Pure PyTorch (educational, works everywhere)

Usage:
    from models.hybrid import HymbaLM, JambaLM, get_hymba_config, MAMBA_SSM_AVAILABLE
    
    config = get_hymba_config("30m")
    config.use_fast_ssm = True  # Use CUDA kernels if available
    model = HymbaLM(config)
"""

from .model_hymba import (
    HymbaLM,
    JambaLM,
    HymbaConfig,
    HymbaLayer,
    JambaLayer,
    SelectiveSSM,
    MultiHeadAttention,
    get_hymba_config,
    MAMBA_SSM_AVAILABLE,
)

__all__ = [
    "HymbaLM",
    "JambaLM",
    "HymbaConfig",
    "HymbaLayer",
    "JambaLayer",
    "SelectiveSSM",
    "MultiHeadAttention",
    "get_hymba_config",
    "MAMBA_SSM_AVAILABLE",
]
