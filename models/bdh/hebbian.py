#!/usr/bin/env python3
"""
Hebbian Learning Module for BDH-SLIM Architecture

Implements biological learning rules as described in the Baby Dragon Hatchling paper.
Key concepts:
- Local Hebbian updates: Δw ∝ x * y (no global gradients needed)
- Activity-dependent plasticity
- Synaptic strengthening based on co-activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianLinear(nn.Module):
    """
    Linear layer with Hebbian learning support.
    
    Can be trained with either:
    1. Standard backpropagation (global gradients)
    2. Hebbian learning (local activity-based updates)
    3. Hybrid (both mechanisms)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 learning_rate: float = 0.01, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hebbian_lr = learning_rate
        
        # Standard weight matrix (can be trained with backprop)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Store activations for Hebbian update
        self.last_input = None
        self.last_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation caching for Hebbian learning.
        
        Args:
            x: Input tensor [batch, seq_len, in_features]
            
        Returns:
            Output tensor [batch, seq_len, out_features]
        """
        # Cache for Hebbian update (detach to avoid backprop interference)
        self.last_input = x.detach()
        
        # Standard linear transformation
        output = F.linear(x, self.weight, self.bias)
        
        # Cache output (after non-linearity will be applied externally)
        self.last_output = output.detach()
        
        return output
    
    def hebbian_update(self, learning_rate: float = None):
        """
        Apply Hebbian learning rule: Δw = η * x * y
        
        "Neurons that fire together, wire together"
        - Strengthens connections between co-active neurons
        - Local rule (no global error signal needed)
        - Biologically plausible
        
        Args:
            learning_rate: Override default Hebbian learning rate
        """
        if self.last_input is None or self.last_output is None:
            return  # No cached activations
        
        lr = learning_rate if learning_rate is not None else self.hebbian_lr
        
        # Hebbian update: Δw = η * y * x^T
        # Averaged over batch and sequence dimensions
        # [batch, seq, out] -> [out, batch*seq]
        # [batch, seq, in] -> [in, batch*seq]
        batch_size, seq_len, _ = self.last_input.shape
        
        # Flatten batch and sequence dims
        x_flat = self.last_input.view(-1, self.in_features)  # [batch*seq, in]
        y_flat = self.last_output.view(-1, self.out_features)  # [batch*seq, out]
        
        # Hebbian update: outer product of outputs and inputs
        # Δw = (1/N) * Σ(y_i * x_i^T)
        with torch.no_grad():
            # Weight update: w += lr * (y^T * x) / N
            N = batch_size * seq_len + 1e-8  # Add epsilon for stability
            delta_w = torch.matmul(y_flat.t(), x_flat) / N
            
            # Safety: Check for NaN/Inf
            if torch.isnan(delta_w).any() or torch.isinf(delta_w).any():
                return  # Skip update
            
            # Safety: Clip to prevent explosions
            delta_w = torch.clamp(delta_w, -1.0, 1.0)
            
            self.weight.data += lr * delta_w
            
            # Optional: Add weight decay to prevent unbounded growth
            self.weight.data *= 0.9999
    
    def reset_cache(self):
        """Clear cached activations (call after Hebbian update)"""
        self.last_input = None
        self.last_output = None


class HebbianAttention(nn.Module):
    """
    Attention mechanism with Hebbian learning support.
    
    Combines:
    - Standard scaled dot-product attention
    - Hebbian plasticity in attention weights
    - Activity-dependent strengthening of connections
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 hebbian_lr: float = 0.01):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.hebbian_lr = hebbian_lr
        
        # Q, K, V projections with Hebbian support
        self.w_q = HebbianLinear(d_model, d_model, learning_rate=hebbian_lr)
        self.w_k = HebbianLinear(d_model, d_model, learning_rate=hebbian_lr)
        self.w_v = HebbianLinear(d_model, d_model, learning_rate=hebbian_lr)
        self.w_o = HebbianLinear(d_model, d_model, learning_rate=hebbian_lr)
        
        self.dropout = nn.Dropout(dropout)
        
        # Cache attention patterns for Hebbian updates
        self.last_attn_weights = None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Multi-head attention with Hebbian learning support.
        
        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections (caches activations internally)
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn_weights = attn_weights.detach()  # Cache for analysis
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        
        return output
    
    def hebbian_update(self, learning_rate: float = None):
        """Apply Hebbian updates to all projection matrices"""
        lr = learning_rate if learning_rate is not None else self.hebbian_lr
        
        self.w_q.hebbian_update(lr)
        self.w_k.hebbian_update(lr)
        self.w_v.hebbian_update(lr)
        self.w_o.hebbian_update(lr)
    
    def reset_cache(self):
        """Clear all cached activations"""
        self.w_q.reset_cache()
        self.w_k.reset_cache()
        self.w_v.reset_cache()
        self.w_o.reset_cache()
        self.last_attn_weights = None


def activity_dependent_pruning(module: nn.Module, threshold: float = 0.01):
    """
    Prune weights based on activity (biological sparsity).
    
    Weak connections (low absolute weights) are pruned, simulating
    biological synaptic pruning where unused connections weaken.
    
    Args:
        module: Neural network module with weights to prune
        threshold: Absolute weight threshold below which to prune
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Create mask for weights above threshold
                mask = (param.abs() > threshold).float()
                # Apply mask (sets small weights to zero)
                param.data *= mask


def homeostatic_scaling(module: nn.Module, target_activity: float = 0.5):
    """
    Homeostatic plasticity: Scale weights to maintain target activity level.
    
    Biological neurons regulate their firing rates to prevent runaway
    excitation or silence. This implements a similar mechanism.
    
    Args:
        module: Module to apply homeostatic scaling to
        target_activity: Desired mean activity level
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Calculate current effective activity (approximated by weight magnitude)
                current_activity = param.abs().mean()
                
                if current_activity > 0:
                    # Scale weights to achieve target activity
                    scale_factor = target_activity / current_activity
                    param.data *= scale_factor


class HebbianFFN(nn.Module):
    """
    Feed-forward network with Hebbian learning.
    
    Standard transformer FFN but with Hebbian updates available.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 hebbian_lr: float = 0.01):
        super().__init__()
        self.fc1 = HebbianLinear(d_model, d_ff, learning_rate=hebbian_lr)
        self.fc2 = HebbianLinear(d_ff, d_model, learning_rate=hebbian_lr)
        self.dropout = nn.Dropout(dropout)
        self.hebbian_lr = hebbian_lr
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x -> fc1 -> gelu -> dropout -> fc2
        
        Args:
            x: Input [batch, seq_len, d_model]
            
        Returns:
            Output [batch, seq_len, d_model]
        """
        h = F.gelu(self.fc1(x))
        h = self.dropout(h)
        return self.fc2(h)
    
    def hebbian_update(self, learning_rate: float = None):
        """Apply Hebbian updates to both layers"""
        lr = learning_rate if learning_rate is not None else self.hebbian_lr
        self.fc1.hebbian_update(lr)
        self.fc2.hebbian_update(lr)
    
    def reset_cache(self):
        """Clear cached activations"""
        self.fc1.reset_cache()
        self.fc2.reset_cache()
