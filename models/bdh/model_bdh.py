#!/usr/bin/env python3
"""
BDH-SLIM (Baby Dragon Hatchling) Transformer

A biologically-inspired transformer architecture with Hebbian learning.
Based on "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"

Key features:
- Hebbian learning for synaptic plasticity
- Activity-dependent sparsity (biological pruning)
- Sparse, positive activations (monosemanticity)
- Can be trained with backprop, Hebbian, or hybrid approaches
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import Config
from hebbian import HebbianAttention, HebbianFFN, activity_dependent_pruning, homeostatic_scaling


class BDHTransformerLayer(nn.Module):
    """
    Single BDH transformer layer with Hebbian learning support.
    
    Architecture:
    1. LayerNorm -> Hebbian Multi-head Attention -> Residual
    2. LayerNorm -> Hebbian FFN -> Residual
    
    Can be trained with:
    - Pure backpropagation (standard gradient descent)
    - Pure Hebbian learning (local activity-based updates)
    - Hybrid (both mechanisms simultaneously)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Hebbian attention
        self.attn = HebbianAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            hebbian_lr=config.hebbian_lr
        )
        
        # Hebbian feed-forward network
        self.ffn = HebbianFFN(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            hebbian_lr=config.hebbian_lr
        )
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with caching for Hebbian updates.
        
        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        # Attention sublayer with residual
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm, mask=mask)
        x = x + self.dropout(attn_out)
        
        # FFN sublayer with residual
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x
    
    def hebbian_update(self, learning_rate: float = None):
        """Apply Hebbian updates to attention and FFN"""
        self.attn.hebbian_update(learning_rate)
        self.ffn.hebbian_update(learning_rate)
    
    def reset_cache(self):
        """Clear all cached activations"""
        self.attn.reset_cache()
        self.ffn.reset_cache()


class BDHTransformer(nn.Module):
    """
    BDH-SLIM Transformer for language modeling.
    
    A biologically-inspired alternative to standard transformers with:
    - Hebbian learning capability
    - Sparse, interpretable activations
    - Activity-dependent plasticity
    - Compatible with standard backpropagation
    
    Training modes:
    1. Backprop only: Standard gradient descent (like GPT)
    2. Hebbian only: Local learning rules (biological)
    3. Hybrid: Both mechanisms (best of both worlds)
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Stack of BDH transformer layers
        self.layers = nn.ModuleList([
            BDHTransformerLayer(config) for _ in range(config.n_layers)
        ])
        
        # Output layers
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Biological parameters
        self.pruning_steps = 0  # Track when to apply activity-dependent pruning
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights (Xavier for transformers)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """
        Forward pass with optional Hebbian learning.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            labels: Target tokens for loss computation [batch, seq_len]
            
        Returns:
            Dictionary with:
            - logits: Output logits [batch, seq_len, vocab_size]
            - loss: Cross-entropy loss (if labels provided)
            - hidden: Final hidden states (for analysis)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token and position embeddings
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        
        x = self.drop(tok_emb + pos_emb)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
        
        # Pass through transformer layers (caches activations for Hebbian)
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final layer norm and output head
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden": x.detach()  # For analysis
        }
    
    def hebbian_update(self, learning_rate: float = None):
        """
        Apply Hebbian learning to all layers.
        
        Call this after forward pass to update weights based on
        cached activations using local Hebbian rules.
        
        Args:
            learning_rate: Override default Hebbian learning rate
        """
        lr = learning_rate if learning_rate is not None else self.config.hebbian_lr
        
        for layer in self.layers:
            layer.hebbian_update(lr)
    
    def reset_cache(self):
        """Clear all cached activations"""
        for layer in self.layers:
            layer.reset_cache()
    
    def apply_biological_plasticity(self, prune_threshold: float = 0.01):
        """
        Apply biological plasticity mechanisms:
        1. Activity-dependent pruning (weak synapses die)
        2. Homeostatic scaling (maintain activity levels)
        
        Call periodically during training (e.g., every 1000 steps)
        """
        self.pruning_steps += 1
        
        # Apply to all layers
        for layer in self.layers:
            # Prune weak connections
            activity_dependent_pruning(layer, threshold=prune_threshold)
            
            # Homeostatic regulation every 10 pruning cycles
            if self.pruning_steps % 10 == 0:
                homeostatic_scaling(layer, target_activity=0.5)
    
    def get_sparsity_stats(self) -> dict:
        """
        Analyze sparsity of the model (biological property).
        
        Returns statistics about:
        - Weight sparsity (zero weights)
        - Activation sparsity (how many neurons fire)
        - Connection strength distribution
        """
        total_params = 0
        zero_params = 0
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param.abs() < 1e-6).sum().item()
        
        sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            "weight_sparsity": sparsity,
            "total_params": total_params,
            "zero_params": zero_params,
            "active_params": total_params - zero_params
        }
    
    def get_learning_rule(self) -> str:
        """Return learning rule type for meta-architecture compatibility"""
        if hasattr(self.config, 'learning_mode'):
            return self.config.learning_mode
        return 'hybrid'  # Default: both backprop and Hebbian


# Utility function for creating BDH config
def create_bdh_config(**kwargs) -> Config:
    """
    Create a Config object with BDH-specific defaults.
    
    Usage:
        config = create_bdh_config(
            d_model=512,
            n_layers=6,
            hebbian_lr=0.01,
            learning_mode='hybrid'
        )
    """
    defaults = {
        'vocab_size': 50257,
        'd_model': 512,
        'n_heads': 8,
        'd_ff': 1024,
        'n_layers': 6,
        'dropout': 0.1,
        'max_len': 256,
        'hebbian_lr': 0.01,  # Hebbian learning rate
        'learning_mode': 'hybrid',  # 'backprop', 'hebbian', or 'hybrid'
        'prune_every': 1000,  # Apply biological pruning every N steps
        'prune_threshold': 0.01,  # Threshold for weak connection pruning
    }
    defaults.update(kwargs)
    
    # Create Config object with these settings
    config = Config()
    for key, value in defaults.items():
        setattr(config, key, value)
    
    return config
