#!/usr/bin/env python3
"""
Standard Transformer Language Model.

A baseline decoder-only GPT-style transformer for comparison with other architectures.
No MoE, no SSM - just standard multi-head attention + FFN.

Architecture:
- Token + positional embeddings
- N transformer blocks (attention + FFN with pre-norm)
- Causal (autoregressive) attention mask
- Weight tying between embedding and output projection

This serves as the baseline for comparing:
- Mamba (pure SSM, no attention)
- Hymba (hybrid attention + SSM)
- Sparse MoE (attention + sparse experts)
"""

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for standard Transformer model."""
    # Model dimensions (matched to Mamba 30M for fair comparison)
    d_model: int = 384          # Hidden dimension
    n_layers: int = 8           # Number of transformer blocks
    n_heads: int = 6            # Attention heads
    d_ff: int = 1536            # FFN hidden dim (4x d_model)
    
    # Training
    vocab_size: int = 50257     # GPT-2 tokenizer
    max_len: int = 256          # Max sequence length
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 4
    grad_accumulation_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 5000
    warmup_steps: int = 500
    eval_every: int = 500
    
    # Output
    output_dir: str = "output/transformer"


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [seq_len, seq_len] causal mask (True = masked)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Attention with residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # FFN with residual
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TransformerLM(nn.Module):
    """
    Standard Transformer Language Model.
    
    Decoder-only GPT-style architecture:
    - Token embeddings + learned positional embeddings
    - N transformer blocks (pre-norm)
    - Output projection tied to embeddings
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        n_params = self.num_parameters()
        print(f"TransformerLM initialized")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ):
        """
        Args:
            input_ids: [batch, seq_len] token indices
            labels: [batch, seq_len] target tokens (optional)
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        for _ in range(max_new_tokens):
            # Crop to max_len
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_len else input_ids[:, -self.config.max_len:]
            
            # Forward
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# =============================================================================
# CONFIG PRESETS
# =============================================================================

def get_transformer_config(preset: str = "30m") -> TransformerConfig:
    """Get preset configurations."""
    configs = {
        # ~27M params - matches Mamba 30M for fair comparison
        "30m": TransformerConfig(
            d_model=384,
            n_layers=8,
            n_heads=6,
            d_ff=1536,
        ),
        # ~125M params - larger model
        "125m": TransformerConfig(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
        ),
    }
    return configs.get(preset, configs["30m"])


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRANSFORMER MODEL TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = get_transformer_config("30m")
    model = TransformerLM(config).to(device)
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"\nInput shape: {input_ids.shape}")
    outputs = model(input_ids, labels=input_ids)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    generated = model.generate(input_ids[:1, :10], max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    print("\n✓ TransformerLM test passed!")
    print("="*70)
