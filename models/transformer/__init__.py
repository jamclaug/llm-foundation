"""
Standard Transformer Language Model.

A baseline decoder-only transformer for comparison with:
- Mamba (SSM)
- Hymba (Hybrid Attention+SSM)
- Sparse MoE Transformer

Usage:
    from models.transformer import TransformerLM, TransformerConfig
    
    config = TransformerConfig()
    model = TransformerLM(config)
"""

from .model import TransformerLM, TransformerConfig

__all__ = ["TransformerLM", "TransformerConfig"]
