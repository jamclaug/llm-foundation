"""
Hybrid Transformer-Mamba models.

This module provides:
- HymbaLM: Parallel attention + SSM heads with learned mixing (Hymba-style)
- JambaLM: Interleaved attention and SSM layers (Jamba-style)

Usage:
    from models.hybrid import HymbaLM, JambaLM, get_hymba_config
    
    config = get_hymba_config("30m")
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
]
