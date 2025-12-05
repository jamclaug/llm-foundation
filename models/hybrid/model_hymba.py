#!/usr/bin/env python3
"""
Hymba-Style Hybrid Transformer-Mamba Model

A hybrid architecture with parallel Transformer and Mamba heads operating
simultaneously within each layer. Inspired by:
- Hymba (parallel attention + SSM heads)
- Jamba (interleaved Transformer + Mamba layers)

Key insight from research:
- Transformer: handles fine-grained, LOCAL context (precise token interactions)
- Mamba SSM: handles efficient GLOBAL context (long-range summarization)
- Combined: superior reasoning even at small scale (Hymba-125M > Mamba-130M)

Architecture per layer:
    Input -> LayerNorm -> [Attention Head || Mamba Head] -> Weighted Combine -> FFN -> Output
                               ↑                ↑
                          Local detail    Global summary

Supports two SSM implementations:
1. Pure PyTorch (educational, works everywhere)
2. mamba-ssm CUDA kernels (10-50x faster, Linux/WSL only)

Reference: "Hymba: A Hybrid-head Architecture for Small Language Models"
"""

import math
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

# Try to import mamba-ssm for CUDA-optimized kernels
MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✓ mamba-ssm library loaded - Hymba will use CUDA-optimized SSM kernels")
except ImportError:
    print("⚠ mamba-ssm not installed - Hymba will use pure PyTorch SSM (slower)")
    print("  For 10-50x speedup, install: pip install causal-conv1d mamba-ssm")


@dataclass
class HymbaConfig:
    """Configuration for Hymba hybrid model."""
    # Model dimensions
    d_model: int = 384          # Hidden dimension
    n_layers: int = 6           # Number of hybrid layers
    n_heads: int = 6            # Attention heads (for transformer path)
    d_ff: int = 1024            # FFN intermediate dimension
    
    # SSM (Mamba) parameters
    d_state: int = 16           # SSM state dimension
    d_conv: int = 4             # Conv kernel size
    expand: int = 2             # SSM expansion factor
    
    # Hybrid mixing
    attn_ratio: float = 0.5     # Portion of heads for attention (vs SSM)
    learnable_mix: bool = True  # Learn mixing weights vs fixed 50/50
    
    # Implementation
    use_fast_ssm: bool = True   # Use mamba-ssm CUDA kernels if available
    
    # Training
    vocab_size: int = 50257
    max_len: int = 256
    dropout: float = 0.1
    
    @property
    def num_params_millions(self) -> float:
        """Rough parameter count estimate."""
        # Embedding + LM head
        embed = self.vocab_size * self.d_model * 2
        # Per layer: attn + ssm + ffn + norms
        attn = 4 * self.d_model * self.d_model  # Q,K,V,O
        ssm = self.d_model * self.d_model * self.expand * 4  # rough
        ffn = 2 * self.d_model * self.d_ff
        per_layer = attn + ssm + ffn
        total = embed + per_layer * self.n_layers
        return total / 1e6


# =============================================================================
# SELECTIVE SSM (Mamba core) - Supports both Pure PyTorch and mamba-ssm
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - the core of Mamba.
    
    Processes sequences with linear time complexity O(L) vs attention's O(L²).
    The "selective" part: B, C, Δ are input-dependent (not fixed).
    
    Supports two implementations:
    - use_fast=True: Use mamba-ssm CUDA kernels (10-50x faster)
    - use_fast=False: Pure PyTorch sequential scan (educational)
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, use_fast: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.use_fast = use_fast and MAMBA_SSM_AVAILABLE
        
        if self.use_fast:
            # Use mamba-ssm's optimized Mamba module
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Pure PyTorch implementation (slower but educational)
            self._init_pytorch_ssm(d_model, d_state, d_conv, expand)
    
    def _init_pytorch_ssm(self, d_model, d_state, d_conv, expand):
        """Initialize pure PyTorch SSM components."""
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Causal convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner
        )
        
        # SSM parameter projections
        self.dt_rank = max(d_model // 16, 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj()
        
        # A matrix (log-space for stability)
        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def _init_dt_proj(self):
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_proj.bias.data = dt + torch.log(-torch.expm1(-dt))
    
    def forward(self, x):
        """x: [batch, seq_len, d_model] -> [batch, seq_len, d_model]"""
        if self.use_fast:
            # Use mamba-ssm's optimized implementation
            return self.mamba(x)
        else:
            # Pure PyTorch implementation
            return self._forward_pytorch(x)
    
    def _forward_pytorch(self, x):
        """Pure PyTorch SSM forward pass."""
        batch, seq_len, _ = x.shape
        
        # Project and split into SSM path and gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Causal convolution for local context
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)
        
        # Compute selective SSM parameters (input-dependent!)
        x_dbc = self.x_proj(x_ssm)
        dt, B, C = torch.split(x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        # Sequential scan (could be parallelized with CUDA)
        y = self._sequential_scan(x_ssm, dt, A, B, C)
        
        # Gate and project
        y = (y + x_ssm * self.D) * F.silu(z)
        return self.out_proj(y)
    
    def _sequential_scan(self, x, dt, A, B, C):
        """O(L) sequential scan - main Mamba computation."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(dt_A)
        dt_B = dt.unsqueeze(-1) * B.unsqueeze(2)
        x_db = x.unsqueeze(-1) * dt_B
        
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for k in range(seq_len):
            h = A_bar[:, k] * h + x_db[:, k]
            y_k = torch.einsum('bn,bdn->bd', C[:, k], h)
            outputs.append(y_k)
        
        return torch.stack(outputs, dim=1)


# =============================================================================
# MULTI-HEAD ATTENTION (Transformer core)  
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """x: [batch, seq_len, d_model]"""
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


# =============================================================================
# HYMBA HYBRID LAYER
# =============================================================================

class HymbaLayer(nn.Module):
    """
    Hymba-style hybrid layer with PARALLEL Transformer + Mamba heads.
    
    Key insight: Run both attention AND SSM on the same input, then combine.
    - Attention: precise local token interactions
    - SSM: efficient global context summarization
    
    This is different from Jamba (which interleaves layers).
    """
    
    def __init__(self, config: HymbaConfig):
        super().__init__()
        self.config = config
        
        # Pre-norm
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Parallel heads
        self.attn = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        
        self.ssm = SelectiveSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            use_fast=config.use_fast_ssm,  # Use CUDA kernels if available
        )
        
        # Learnable mixing weights (or fixed 50/50)
        if config.learnable_mix:
            # Learned per-feature mixing: which features prefer attn vs ssm
            self.mix_weight = nn.Parameter(torch.ones(config.d_model) * 0.5)
        else:
            self.register_buffer('mix_weight', torch.ones(config.d_model) * config.attn_ratio)
        
        # FFN (shared, after mixing)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Parallel processing through attention and SSM, then combine.
        
        x: [batch, seq_len, d_model]
        """
        # Pre-norm for attention/SSM sublayer
        x_norm = self.ln1(x)
        
        # PARALLEL: both heads process same input
        attn_out = self.attn(x_norm, mask)  # Local, precise
        ssm_out = self.ssm(x_norm)          # Global, efficient
        
        # Learned mixing: sigmoid ensures [0, 1] range
        alpha = torch.sigmoid(self.mix_weight)  # [d_model]
        
        # Combine: α * attention + (1-α) * ssm
        mixed = alpha * attn_out + (1 - alpha) * ssm_out
        x = x + self.dropout(mixed)
        
        # FFN sublayer
        x = x + self.ffn(self.ln2(x))
        
        return x
    
    def get_mixing_stats(self):
        """Return attention vs SSM preference stats."""
        alpha = torch.sigmoid(self.mix_weight).detach()
        return {
            'attn_ratio_mean': alpha.mean().item(),
            'attn_ratio_std': alpha.std().item(),
            'attn_ratio_min': alpha.min().item(),
            'attn_ratio_max': alpha.max().item(),
        }


# =============================================================================
# FULL HYMBA MODEL
# =============================================================================

class HymbaLM(nn.Module):
    """
    Hymba Language Model - Hybrid Transformer-Mamba architecture.
    
    Each layer has parallel attention + SSM heads with learned mixing.
    Superior reasoning compared to pure Mamba at same param count.
    """
    
    def __init__(self, config: HymbaConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Hybrid layers
        self.layers = nn.ModuleList([HymbaLayer(config) for _ in range(config.n_layers)])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        # Print model info
        ssm_impl = "mamba-ssm (CUDA)" if (config.use_fast_ssm and MAMBA_SSM_AVAILABLE) else "PyTorch (slow)"
        print(f"HymbaLM: {self.num_params/1e6:.1f}M params | SSM: {ssm_impl}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] for computing loss
            
        Returns:
            loss (if labels provided), logits
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.drop(x)
        
        # Create causal mask once
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Process through hybrid layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
        
        return loss, logits
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_len
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_len else input_ids[:, -self.config.max_len:]
            
            # Forward
            _, logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_mixing_stats(self):
        """Get attention vs SSM mixing statistics per layer."""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_mixing_stats()
        return stats


# =============================================================================
# ALTERNATIVE: JAMBA-STYLE INTERLEAVED
# =============================================================================

class JambaLayer(nn.Module):
    """
    Jamba-style layer: either pure Attention OR pure Mamba (not both).
    Use with alternating pattern: Attn, Mamba, Attn, Mamba, ...
    
    Simpler than Hymba but still captures both local and global.
    """
    
    def __init__(self, config: HymbaConfig, use_attention: bool = True):
        super().__init__()
        self.config = config
        self.use_attention = use_attention
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        if use_attention:
            self.mixer = MultiHeadAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                dropout=config.dropout
            )
        else:
            self.mixer = SelectiveSSM(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                use_fast=config.use_fast_ssm,  # Use CUDA kernels if available
            )
        
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x_norm = self.ln1(x)
        if self.use_attention:
            x = x + self.dropout(self.mixer(x_norm, mask))
        else:
            x = x + self.dropout(self.mixer(x_norm))
        x = x + self.ffn(self.ln2(x))
        return x


class JambaLM(nn.Module):
    """
    Jamba-style LM with interleaved Attention and Mamba layers.
    Pattern: Attn, Mamba, Attn, Mamba, ... (alternating)
    """
    
    def __init__(self, config: HymbaConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Alternating layers
        self.layers = nn.ModuleList([
            JambaLayer(config, use_attention=(i % 2 == 0))
            for i in range(config.n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.apply(self._init_weights)
        
        # Print model info
        ssm_impl = "mamba-ssm (CUDA)" if (config.use_fast_ssm and MAMBA_SSM_AVAILABLE) else "PyTorch (slow)"
        print(f"JambaLM: {self.num_params/1e6:.1f}M params | SSM: {ssm_impl}")
        print(f"  Pattern: {['Attn' if i%2==0 else 'Mamba' for i in range(config.n_layers)]}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.drop(x)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100
            )
        
        return loss, logits


# =============================================================================
# PRESET CONFIGS
# =============================================================================

def get_hymba_config(preset: str = "30m") -> HymbaConfig:
    """Get preset configurations."""
    configs = {
        # ~30M params - comparable to BDH baseline
        "30m": HymbaConfig(
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_ff=1024,
            d_state=16,
            expand=2,
            learnable_mix=True,
        ),
        # ~125M params - comparable to Hymba paper
        "125m": HymbaConfig(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=2048,
            d_state=16,
            expand=2,
            learnable_mix=True,
        ),
        # Fast version - smaller SSM for quicker training
        "30m-fast": HymbaConfig(
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_ff=1024,
            d_state=8,      # Smaller state (16 -> 8)
            expand=1,       # No expansion (2 -> 1)
            d_conv=2,       # Smaller conv (4 -> 2)
            learnable_mix=True,
            max_len=128,    # Shorter sequences
        ),
    }
    return configs.get(preset, configs["30m"])


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HYMBA HYBRID MODEL TEST")
    print("=" * 60)
    
    # Test Hymba (parallel heads)
    config = get_hymba_config("30m")
    model = HymbaLM(config)
    
    # Dummy input
    x = torch.randint(0, config.vocab_size, (2, 64))
    
    # Forward pass
    loss, logits = model(x, labels=x)
    print(f"\nHymba forward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Check mixing stats
    print(f"\nMixing statistics (attention ratio per layer):")
    for layer_name, stats in model.get_mixing_stats().items():
        print(f"  {layer_name}: mean={stats['attn_ratio_mean']:.3f}, "
              f"range=[{stats['attn_ratio_min']:.3f}, {stats['attn_ratio_max']:.3f}]")
    
    # Test Jamba (interleaved)
    print("\n" + "=" * 60)
    jamba = JambaLM(config)
    loss2, logits2 = jamba(x, labels=x)
    print(f"\nJamba forward pass:")
    print(f"  Loss: {loss2.item():.4f}")
    
    print("\n✅ Both hybrid architectures working!")
