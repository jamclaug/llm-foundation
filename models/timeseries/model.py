"""
Time Series Transformer Model.

A decoder-only transformer for multivariate time series prediction,
using the TimeSeriesTokenizer for input encoding with percentage-based
delta patterns.

Architecture Overview
---------------------
1. TimeSeriesTokenizer converts raw values to embeddings:
   - Stream embeddings (which sensor)
   - Pattern embeddings (percentage changes, scale-invariant)
   - Time embeddings (window position)
   - Level projection (absolute scale information)

2. Transformer blocks process the sequence:
   - Multi-head self-attention with causal masking
   - Feed-forward network with GELU activation
   - Pre-norm (LayerNorm before attention/FFN)

3. Prediction head outputs next-window forecasts:
   - Linear projection from d_model to window_size
   - MSE loss between predicted and actual values

Why MSE Loss is a Good Evaluation Metric
----------------------------------------
For time series forecasting, MSE (Mean Squared Error) is well-suited because:

1. **Scale-Sensitivity**: MSE penalizes large errors more than small ones
   (quadratic penalty). This is desirable for forecasting where big misses
   (e.g., predicting +5% when actual is -10%) are worse than small misses.

2. **Differentiability**: MSE is smooth and differentiable everywhere,
   making gradient-based optimization stable and efficient.

3. **Direct Interpretation**: MSE in original units tells you the average
   squared deviation from true values. RMSE (√MSE) gives error in same
   units as the data.

4. **Proper Scoring Rule**: MSE is a "proper" scoring rule, meaning the
   model's best strategy is to predict the true expected value. There's
   no gaming the metric.

5. **Variance Decomposition**: MSE = Bias² + Variance + Irreducible Error
   This decomposition helps diagnose model issues.

Loss Interpretation for Time Series
-----------------------------------
When you see training loss decreasing:
- Loss 1.0 → RMSE ≈ 1.0 (predictions off by ~1 unit on average)
- Loss 0.01 → RMSE ≈ 0.1 (predictions off by ~0.1 units)

For percentage predictions (if predicting returns):
- Loss 0.0001 → RMSE ≈ 0.01 → ~1% average error in predictions

The model should achieve lower loss than naive baselines:
- "Predict previous value" baseline (random walk)
- "Predict historical mean" baseline
- "Predict moving average" baseline

If model loss > naive baseline loss, the model isn't learning useful patterns.

Usage Example
-------------
    from models.timeseries import (
        TimeSeriesTransformerConfig,
        TimeSeriesTransformer,
    )
    
    config = TimeSeriesTransformerConfig(
        num_streams=10,
        window_size=5,
        num_bins=100,
        d_model=256,
        n_layers=6,
        n_heads=8,
    )
    model = TimeSeriesTransformer(config)
    
    # Input: (batch, time_steps, num_streams)
    values = torch.randn(32, 100, 10)
    
    # Forward pass with targets for training
    outputs = model(values, targets=values)
    loss = outputs["loss"]
    predictions = outputs["predictions"]
    
    # Inference: predict future
    future = model.predict(values, num_future_windows=5)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .tokenizer import TimeSeriesTokenizer, TimeSeriesTokenizerConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TimeSeriesTransformerConfig:
    """
    Configuration for TimeSeriesTransformer.
    
    Tokenizer Settings (passed to TimeSeriesTokenizer):
        num_streams: Number of input streams/sensors.
        window_size: Time steps per window.
        num_bins: Bins for delta discretization.
        max_windows: Maximum windows for positional embeddings.
        delta_mode: How to compute deltas ("absolute", "percent", "log_return").
        delta_thresholds: Custom thresholds for delta binning (optional).
        default_threshold_scale: Scale for auto-generated thresholds.
        delta_combine: How to combine delta embeddings ("sum" or "mean").
    
    Model Architecture:
        d_model: Hidden dimension throughout the model.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        d_ff: FFN hidden dimension (default: 4 * d_model).
        dropout: Dropout probability.
    
    Prediction Settings:
        prediction_target: What to predict:
            - "values": Predict raw values of next window (default)
            - "deltas": Predict delta values (changes) of next window
    
    Parameter Counts (approximate for d_model=256, n_layers=6):
        - Tokenizer: ~100K params (embeddings)
        - Per layer: ~2M params (attention + FFN)
        - Total: ~13M params
    """
    # Tokenizer settings
    num_streams: int
    window_size: int = 5
    num_bins: int = 100
    max_windows: int = 1024
    delta_mode: Literal["absolute", "percent", "log_return"] = "percent"
    delta_thresholds: Optional[Tensor] = None
    default_threshold_scale: float = 0.1
    delta_combine: Literal["sum", "mean"] = "mean"
    
    # Model architecture
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: Optional[int] = None  # Defaults to 4 * d_model
    dropout: float = 0.1
    
    # Prediction settings
    prediction_target: Literal["values", "deltas"] = "values"
    
    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        
        # Validate
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
    
    def to_tokenizer_config(self) -> TimeSeriesTokenizerConfig:
        """Create a TimeSeriesTokenizerConfig from this config."""
        return TimeSeriesTokenizerConfig(
            num_streams=self.num_streams,
            window_size=self.window_size,
            num_bins=self.num_bins,
            d_model=self.d_model,
            max_windows=self.max_windows,
            delta_mode=self.delta_mode,
            delta_thresholds=self.delta_thresholds,
            default_threshold_scale=self.default_threshold_scale,
            delta_combine=self.delta_combine,
        )


# =============================================================================
# TRANSFORMER COMPONENTS
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional causal masking.
    
    Uses combined QKV projection for efficiency and scaled dot-product attention.
    
    Architecture:
        Input: (batch, seq_len, d_model)
        1. Project to Q, K, V via single linear layer
        2. Split into n_heads
        3. Compute attention: softmax(QK^T / sqrt(d_k)) * V
        4. Concatenate heads and project output
        Output: (batch, seq_len, d_model)
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Combined QKV projection (3x d_model for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional attention mask (True = masked positions).
            is_causal: If True, apply causal (autoregressive) masking.
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq, 3*d_model)
        qkv = qkv.view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        # attn = softmax(Q @ K^T / sqrt(d_k)) @ V
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # (batch, heads, seq, seq)
        
        # Apply causal mask if requested (for autoregressive modeling)
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        out = (attn_probs @ v)  # (batch, heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    
    Architecture:
        Input: (batch, seq_len, d_model)
        1. Linear: d_model -> d_ff
        2. GELU activation
        3. Dropout
        4. Linear: d_ff -> d_model
        Output: (batch, seq_len, d_model)
    
    The expansion ratio d_ff/d_model is typically 4x.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        # GELU is smoother than ReLU, works better for transformers
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.
    
    Pre-norm (used here) vs Post-norm:
        Pre-norm:  x = x + Attn(LN(x))  <- more stable training
        Post-norm: x = LN(x + Attn(x))  <- original Transformer
    
    Architecture:
        1. LayerNorm -> Multi-Head Attention -> Residual
        2. LayerNorm -> Feed-Forward -> Residual
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None, 
        is_causal: bool = False,
    ) -> Tensor:
        # Attention with residual (pre-norm)
        x = x + self.dropout(self.attn(self.ln1(x), mask, is_causal))
        # FFN with residual (pre-norm)
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


# =============================================================================
# MAIN MODEL
# =============================================================================

class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for multivariate time series prediction.
    
    Uses TimeSeriesTokenizer for input encoding with percentage-based
    pattern embeddings, making the model scale-invariant. Predicts
    future values using causal attention.
    
    Data Flow
    ---------
    1. Input: raw values (batch, time_steps, num_streams)
    
    2. Tokenization: TimeSeriesTokenizer
       → embeddings (batch, num_windows, num_streams, d_model)
       - Stream embedding: which sensor
       - Pattern embedding: percentage changes (scale-invariant!)
       - Time embedding: window position
       - Level projection: absolute scale information
    
    3. Flatten for Transformer:
       → (batch, num_windows * num_streams, d_model)
       Order: [w0s0, w0s1, ..., w0sN, w1s0, w1s1, ..., wMsN]
    
    4. Transformer Blocks (causal attention):
       → (batch, num_windows * num_streams, d_model)
       Each position can only attend to previous positions
    
    5. Unflatten:
       → (batch, num_windows, num_streams, d_model)
    
    6. Prediction Head:
       → (batch, num_windows, num_streams, window_size)
       Each window predicts the NEXT window's values
    
    Training Objective
    ------------------
    Given windows [0, 1, 2, ..., T], predict windows [1, 2, 3, ..., T+1].
    Loss = MSE(predicted[:-1], actual[1:])
    
    This teaches the model to extrapolate patterns, not just interpolate.
    """
    
    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.config = config
        
        # Create tokenizer
        tokenizer_config = config.to_tokenizer_config()
        self.tokenizer = TimeSeriesTokenizer(tokenizer_config)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm (pre-norm architecture)
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Prediction head: predict next window's values for each stream
        # Output: window_size values per (window, stream) position
        self.pred_head = nn.Linear(config.d_model, config.window_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model info
        n_params = self.num_parameters()
        print(f"TimeSeriesTransformer initialized:")
        print(f"  Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"  Tokenizer: num_streams={config.num_streams}, window_size={config.window_size}")
        print(f"  Pattern encoding: delta_mode={config.delta_mode!r}, num_bins={config.num_bins}")
        print(f"  Architecture: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following GPT-2 style."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        values: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for time series prediction.
        
        Args:
            values: Input time series of shape (batch, time_steps, num_streams).
            targets: Optional target values for computing loss.
                     If shape is (batch, time_steps, num_streams), it will be
                     windowed to match predictions.
        
        Returns:
            dict containing:
                - "predictions": (batch, num_windows, num_streams, window_size)
                  Predicted values for each window. predictions[:, t] predicts
                  what window t+1's values should be.
                - "embeddings": (batch, num_windows, num_streams, d_model)
                  Intermediate embeddings after tokenization.
                - "hidden_states": (batch, num_windows * num_streams, d_model)
                  Final hidden states from transformer.
                - "loss": scalar MSE loss if targets provided, else None.
                - "loss_details": dict with additional loss info if targets provided.
        
        Example:
            >>> model = TimeSeriesTransformer(config)
            >>> values = torch.randn(32, 100, 10)
            >>> outputs = model(values, targets=values)
            >>> loss = outputs["loss"]  # Use for training
            >>> predictions = outputs["predictions"]
        """
        batch_size = values.shape[0]
        
        # =================================================================
        # Step 1: Tokenize
        # =================================================================
        # (batch, time_steps, num_streams) → (batch, num_windows, num_streams, d_model)
        embeddings = self.tokenizer(values)
        num_windows = embeddings.shape[1]
        num_streams = embeddings.shape[2]
        
        # =================================================================
        # Step 2: Flatten for Transformer
        # =================================================================
        # (batch, num_windows, num_streams, d_model) → (batch, seq_len, d_model)
        # where seq_len = num_windows * num_streams
        x = embeddings.view(batch_size, num_windows * num_streams, self.config.d_model)
        
        # =================================================================
        # Step 3: Apply Transformer Blocks
        # =================================================================
        # Causal attention: each position only attends to previous positions
        for layer in self.layers:
            x = layer(x, is_causal=True)
        
        # Final layer norm
        x = self.ln_f(x)
        hidden_states = x  # Save for output
        
        # =================================================================
        # Step 4: Unflatten and Predict
        # =================================================================
        # (batch, seq_len, d_model) → (batch, num_windows, num_streams, d_model)
        x = x.view(batch_size, num_windows, num_streams, self.config.d_model)
        
        # Predict next window's values for each (window, stream)
        # (batch, num_windows, num_streams, d_model) → (batch, num_windows, num_streams, window_size)
        predictions = self.pred_head(x)
        
        # =================================================================
        # Step 5: Compute Loss (if targets provided)
        # =================================================================
        loss = None
        loss_details = None
        
        if targets is not None:
            loss, loss_details = self._compute_loss(predictions, targets, values)
        
        return {
            "predictions": predictions,
            "embeddings": embeddings,
            "hidden_states": hidden_states,
            "loss": loss,
            "loss_details": loss_details,
        }
    
    def _compute_loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        input_values: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute MSE loss between predictions and targets.
        
        The key insight: predictions[:, t] predicts targets[:, t+1].
        So we align: pred[:-1] vs targets[1:]
        
        Args:
            predictions: (batch, num_windows, num_streams, window_size)
            targets: (batch, time_steps, num_streams) or already windowed
            input_values: Original input for baseline comparison
        
        Returns:
            loss: Scalar MSE loss
            loss_details: Dict with MSE, RMSE, MAE, and baseline comparisons
        """
        # Window targets if needed
        if targets.dim() == 3:
            targets = self._window_values(targets)
        # targets: (batch, num_windows, num_streams, window_size)
        
        # Align predictions and targets
        # predictions[:, t] predicts what comes AFTER window t, which is window t+1
        # So pred[:-1] should match target[1:]
        pred_aligned = predictions[:, :-1]   # (batch, num_windows-1, num_streams, window_size)
        target_aligned = targets[:, 1:]       # (batch, num_windows-1, num_streams, window_size)
        
        # Compute MSE loss
        loss = F.mse_loss(pred_aligned, target_aligned)
        
        # Compute detailed metrics for monitoring
        with torch.no_grad():
            mse = loss.item()
            rmse = math.sqrt(mse)
            mae = F.l1_loss(pred_aligned, target_aligned).item()
            
            # Naive baseline: predict previous window's values (random walk)
            # baseline_pred = targets[:, :-1] predicts targets[:, 1:]
            baseline_pred = targets[:, :-1]
            baseline_mse = F.mse_loss(baseline_pred, target_aligned).item()
            baseline_rmse = math.sqrt(baseline_mse)
            
            # Skill score: how much better than baseline (>0 is better)
            # skill = 1 - (model_mse / baseline_mse)
            skill_score = 1.0 - (mse / (baseline_mse + 1e-8))
            
            loss_details = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "baseline_mse": baseline_mse,
                "baseline_rmse": baseline_rmse,
                "skill_score": skill_score,  # >0 means better than baseline
            }
        
        return loss, loss_details
    
    def _window_values(self, values: Tensor) -> Tensor:
        """
        Reshape values into non-overlapping windows.
        
        Args:
            values: (batch, time_steps, num_streams)
        
        Returns:
            windowed: (batch, num_windows, num_streams, window_size)
        """
        batch_size, time_steps, num_streams = values.shape
        window_size = self.config.window_size
        
        num_windows = time_steps // window_size
        usable_steps = num_windows * window_size
        
        # Truncate and reshape
        values = values[:, :usable_steps, :]
        # (batch, time_steps) → (batch, num_windows, window_size, num_streams)
        windowed = values.view(batch_size, num_windows, window_size, num_streams)
        # Transpose to (batch, num_windows, num_streams, window_size)
        windowed = windowed.permute(0, 1, 3, 2).contiguous()
        
        return windowed
    
    @torch.no_grad()
    def predict_next(self, values: Tensor) -> Tensor:
        """
        Predict the next window given input values.
        
        This is the simplest inference mode: given past data,
        predict one window into the future.
        
        Args:
            values: Input time series of shape (batch, time_steps, num_streams).
        
        Returns:
            next_window: (batch, window_size, num_streams)
            The predicted values for the next window after the input.
        """
        self.eval()
        
        outputs = self.forward(values)
        predictions = outputs["predictions"]
        
        # Take the last window's prediction
        # predictions[:, -1] gives the prediction for what comes after the last window
        next_window = predictions[:, -1, :, :]  # (batch, num_streams, window_size)
        next_window = next_window.permute(0, 2, 1)  # (batch, window_size, num_streams)
        
        return next_window
    
    @torch.no_grad()
    def predict_future(
        self,
        values: Tensor,
        num_windows: int = 1,
    ) -> Tensor:
        """
        Autoregressively predict multiple future windows.
        
        Repeatedly predicts the next window and appends it to the input
        for the next prediction. This allows forecasting arbitrarily
        far into the future (though accuracy degrades with distance).
        
        Args:
            values: Input time series of shape (batch, time_steps, num_streams).
            num_windows: Number of windows to predict into the future.
        
        Returns:
            future_values: (batch, num_windows * window_size, num_streams)
            Concatenated predictions for all future windows.
        
        Example:
            >>> # Given 100 time steps, predict next 25 (5 windows of size 5)
            >>> future = model.predict_future(values, num_windows=5)
            >>> future.shape  # (batch, 25, num_streams)
        """
        self.eval()
        
        window_size = self.config.window_size
        current_values = values.clone()
        predictions_list = []
        
        for _ in range(num_windows):
            # Predict next window
            next_window = self.predict_next(current_values)
            predictions_list.append(next_window)
            
            # Append prediction to input for next iteration
            current_values = torch.cat([current_values, next_window], dim=1)
        
        # Concatenate all predictions
        future_values = torch.cat(predictions_list, dim=1)
        return future_values


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_model(
    num_streams: int,
    window_size: int = 5,
    num_bins: int = 100,
    d_model: int = 256,
    n_layers: int = 6,
    n_heads: int = 8,
    delta_mode: str = "percent",
    **kwargs,
) -> TimeSeriesTransformer:
    """
    Convenience function to create a TimeSeriesTransformer.
    
    Args:
        num_streams: Number of input streams/sensors.
        window_size: Time steps per window.
        num_bins: Bins for delta discretization.
        d_model: Hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads.
        delta_mode: How to compute deltas ("absolute", "percent", "log_return").
        **kwargs: Additional config options.
    
    Returns:
        Configured TimeSeriesTransformer instance.
    
    Example:
        >>> model = create_model(num_streams=10, d_model=128, n_layers=4)
    """
    config = TimeSeriesTransformerConfig(
        num_streams=num_streams,
        window_size=window_size,
        num_bins=num_bins,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        delta_mode=delta_mode,
        **kwargs,
    )
    return TimeSeriesTransformer(config)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a model by component.
    
    Returns dict with total, trainable, and per-module counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    by_module = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        by_module[name] = params
    
    return {
        "total": total,
        "trainable": trainable,
        "by_module": by_module,
    }
