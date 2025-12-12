"""
Time Series Mamba Model.

A Mamba-based model for multivariate time series prediction,
using the same TimeSeriesTokenizer as the Transformer variant.

Mamba is ideal for long-range time series because:
1. O(n) complexity vs O(n²) for attention
2. Can handle 2000+ time steps on 4GB GPU
3. Natural fit for sequential processes (industrial, financial)

The tokenizer converts raw values to embeddings (stream, pattern, time, level),
then Mamba blocks process the sequence with selective state spaces.

Usage Example
-------------
    from models.timeseries import TimeSeriesMamba, TimeSeriesMambaConfig
    
    config = TimeSeriesMambaConfig(
        num_streams=10,
        window_size=50,  # Can use much larger windows!
        d_model=192,
        n_layers=4,
    )
    model = TimeSeriesMamba(config)
    
    # Same interface as Transformer
    values = torch.randn(32, 2000, 10)  # 2000 time steps!
    outputs = model(values, targets=values)
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
# TRY TO IMPORT MAMBA-SSM FOR CUDA-OPTIMIZED KERNELS
# =============================================================================

MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba
    MAMBA_SSM_AVAILABLE = True
    print("✓ mamba-ssm library loaded - using CUDA-optimized SSM kernels")
except ImportError:
    print("⚠ mamba-ssm not installed - using pure PyTorch SSM (slower)")
    print("  For 10-50x speedup, install: pip install causal-conv1d mamba-ssm")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TimeSeriesMambaConfig:
    """
    Configuration for TimeSeriesMamba.
    
    Tokenizer Settings (same as Transformer):
        num_streams: Number of input streams/sensors.
        window_size: Time steps per window (can be larger than Transformer!).
        num_bins: Bins for delta discretization.
        max_windows: Maximum windows for positional embeddings.
        delta_mode: How to compute deltas ("absolute", "percent", "log_return").
    
    Mamba Architecture:
        d_model: Hidden dimension throughout the model.
        n_layers: Number of Mamba blocks.
        d_state: SSM state dimension (default 16).
        d_conv: Local convolution width (default 4).
        expand: Inner dimension expansion factor (default 2).
        dropout: Dropout probability.
    
    Comparison with Transformer:
        - Mamba has no n_heads (no attention)
        - d_state controls "memory capacity" (similar role to context length)
        - expand controls FFN-like expansion within SSM
    """
    # Tokenizer settings
    num_streams: int
    window_size: int = 20  # Default larger for Mamba
    num_bins: int = 100
    max_windows: int = 4096  # Mamba can handle more!
    delta_mode: Literal["absolute", "percent", "log_return"] = "percent"
    delta_thresholds: Optional[Tensor] = None
    default_threshold_scale: float = 0.1
    delta_combine: Literal["sum", "mean"] = "mean"
    
    # Mamba architecture
    d_model: int = 192
    n_layers: int = 4
    d_state: int = 16  # SSM state dimension
    d_conv: int = 4    # Local convolution width
    expand: int = 2    # Inner dimension = d_model * expand
    dropout: float = 0.1
    
    # Prediction settings
    prediction_target: Literal["values", "deltas"] = "values"
    
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
# MAMBA COMPONENTS (Selective State Space)
# =============================================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - the core of Mamba.
    
    Unlike traditional SSMs with fixed parameters, Mamba makes the
    state transition matrices input-dependent (selective), allowing
    the model to filter information based on content.
    
    State space equation:
        h'(t) = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)
    
    The "selective" part: B, C, and Δ are computed from input x.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        
        # Input projection: x -> (z, x_ssm)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters projection: x -> (Δ, B, C)
        self.dt_rank = max(d_model // 16, 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        
        # Δ projection with special initialization
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self._init_dt_proj()
        
        # A matrix (diagonal, stored as log for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D matrix (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def _init_dt_proj(self):
        """Initialize dt projection for good discretization range."""
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Bias: inverse softplus of uniform [0.001, 0.1]
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.data = inv_dt
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            y: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Project and split
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Causal convolution
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :seq_len]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)
        
        # Compute selective parameters
        x_dbc = self.x_proj(x_ssm)
        dt, B, C = torch.split(x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        # Run selective scan
        y = self.selective_scan(x_ssm, dt, A, B, C)
        
        # Skip connection + gating
        y = y + x_ssm * self.D
        y = y * F.silu(z)
        
        return self.out_proj(y)
    
    def selective_scan(self, x, dt, A, B, C):
        """
        Sequential selective scan (SSM recurrence).
        
        This is O(L) per step - the key advantage over attention's O(L²).
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize: Ā = exp(Δ * A), B̄ = Δ * B
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(dt_A)
        dt_B = dt.unsqueeze(-1) * B.unsqueeze(2)
        x_db = x.unsqueeze(-1) * dt_B
        
        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        outputs = []
        
        for k in range(seq_len):
            h = A_bar[:, k] * h + x_db[:, k]
            y_k = torch.einsum('bn,bdn->bd', C[:, k], h)
            outputs.append(y_k)
        
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """
    Single Mamba block: LayerNorm -> SSM -> Residual + Dropout
    
    Uses CUDA-optimized mamba-ssm when available, falls back to pure PyTorch.
    """
    
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        if MAMBA_SSM_AVAILABLE:
            # Use CUDA-optimized Mamba from mamba-ssm library
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fall back to pure PyTorch implementation
            self.ssm = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
    
    def forward(self, x):
        return x + self.dropout(self.ssm(self.norm(x)))


# =============================================================================
# TIME SERIES MAMBA MODEL
# =============================================================================

class TimeSeriesMamba(nn.Module):
    """
    Mamba-based Time Series Model.
    
    Architecture:
    1. TimeSeriesTokenizer: values → embeddings (stream + pattern + time + level)
    2. Mamba blocks: process sequence with selective state spaces
    3. Prediction head: embeddings → next-window predictions
    
    Key difference from Transformer:
    - O(n) complexity instead of O(n²)
    - Can handle 2000+ time steps on 4GB GPU
    - Better for long-range dependencies (slow industrial processes)
    """
    
    def __init__(self, config: TimeSeriesMambaConfig):
        super().__init__()
        self.config = config
        
        # Create tokenizer
        tokenizer_config = config.to_tokenizer_config()
        self.tokenizer = TimeSeriesTokenizer(tokenizer_config)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Prediction head: predict next window's values (per stream position)
        # Each position in the flattened sequence is one (window, stream) pair
        self.pred_head = nn.Linear(config.d_model, config.window_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model info
        n_params = sum(p.numel() for p in self.parameters())
        print(f"TimeSeriesMamba initialized:")
        print(f"  Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"  Tokenizer: num_streams={config.num_streams}, window_size={config.window_size}")
        print(f"  Pattern encoding: delta_mode='{config.delta_mode}', num_bins={config.num_bins}")
        print(f"  Architecture: d_model={config.d_model}, n_layers={config.n_layers}")
        print(f"  SSM: d_state={config.d_state}, d_conv={config.d_conv}, expand={config.expand}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        values: Tensor,
        targets: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with optional loss computation.
        
        Args:
            values: Input time series (batch, time_steps, num_streams)
            targets: Target values for loss computation (same shape as values)
        
        Returns:
            Dict with:
                - "embeddings": Final hidden states (batch, num_windows * num_streams, d_model)
                - "predictions": Predicted values (batch, num_windows-1, window_size, num_streams)
                - "loss": MSE loss if targets provided
                - "loss_details": Additional metrics if targets provided
        """
        batch_size = values.shape[0]
        
        # =================================================================
        # Step 1: Tokenize
        # =================================================================
        # (batch, time_steps, num_streams) -> (batch, num_windows, num_streams, d_model)
        embeddings = self.tokenizer(values)
        num_windows = embeddings.shape[1]
        num_streams = embeddings.shape[2]
        
        # =================================================================
        # Step 2: Flatten for Mamba
        # =================================================================
        # (batch, num_windows, num_streams, d_model) -> (batch, seq_len, d_model)
        # where seq_len = num_windows * num_streams
        x = embeddings.view(batch_size, num_windows * num_streams, self.config.d_model)
        
        # =================================================================
        # Step 3: Apply Mamba blocks
        # =================================================================
        for layer in self.layers:
            x = layer(x)
        
        # Final norm
        x = self.ln_f(x)
        hidden_states = x  # Save for output
        
        # =================================================================
        # Step 4: Unflatten and Predict
        # =================================================================
        # (batch, seq_len, d_model) -> (batch, num_windows, num_streams, d_model)
        x = x.view(batch_size, num_windows, num_streams, self.config.d_model)
        
        # Predict next window's values for each (window, stream)
        # (batch, num_windows, num_streams, d_model) -> (batch, num_windows, num_streams, window_size)
        predictions = self.pred_head(x)
        
        # Transpose to match expected output format
        # (batch, num_windows, num_streams, window_size) -> (batch, num_windows, window_size, num_streams)
        predictions = predictions.transpose(2, 3)
        
        result = {
            "embeddings": embeddings,  # Original tokenizer output
            "hidden_states": hidden_states,  # Final Mamba output (flattened)
            "predictions": predictions[:, :-1],  # All but last predict forward
            "loss": None,
            "loss_details": None,
        }
        
        # Compute loss if targets provided
        if targets is not None:
            result["loss"], result["loss_details"] = self._compute_loss(
                predictions, values, targets
            )
        
        return result
    
    def _compute_loss(
        self,
        predictions: Tensor,
        values: Tensor,
        targets: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute MSE loss in DELTA SPACE (absolute differences from last value).
        
        The model predicts changes relative to the last value of each window.
        We use absolute deltas (not percent) because the data is normalized
        and percent changes are unstable when values cross near zero.
        
        predictions: (batch, num_windows, window_size, num_streams) - predicted deltas
        values/targets: (batch, time_steps, num_streams) - raw values
        
        Loss is computed on deltas: pred_delta vs actual_delta where
        actual_delta[t] = actual_value[t] - last_known
        """
        batch_size, num_windows, window_size, num_streams = predictions.shape
        
        # Extract actual next windows and compute target deltas
        # Window i predicts values at time steps [(i+1)*W : (i+2)*W]
        # relative to the last value of window i (at time step (i+1)*W - 1)
        actual_deltas = []
        last_values = []
        
        for i in range(num_windows - 1):
            # Last value of current window i
            last_idx = (i + 1) * window_size - 1
            if last_idx >= targets.shape[1]:
                break
            last_val = targets[:, last_idx:last_idx+1, :]  # (batch, 1, num_streams)
            
            # Actual values in next window (i+1)
            start = (i + 1) * window_size
            end = (i + 2) * window_size
            if end > targets.shape[1]:
                break
            actual_window = targets[:, start:end, :]  # (batch, window_size, num_streams)
            
            # Compute absolute delta from last_val
            # delta = actual - last_val (simple difference, stable for normalized data)
            delta = actual_window - last_val  # (batch, window_size, num_streams)
            
            actual_deltas.append(delta)
            last_values.append(last_val)
        
        if len(actual_deltas) == 0:
            return torch.tensor(0.0, device=predictions.device), {}
        
        # Stack: (batch, num_valid_windows, window_size, num_streams)
        actual_deltas = torch.stack(actual_deltas, dim=1)
        last_values = torch.stack(last_values, dim=1).squeeze(2)  # (batch, num_valid_windows, num_streams)
        
        # Align predictions to same number of windows
        pred_aligned = predictions[:, :actual_deltas.shape[1]]
        
        # MSE loss on deltas (absolute differences)
        mse_loss = F.mse_loss(pred_aligned, actual_deltas)
        
        # Reconstruct predicted values for metric computation
        # pred_values = last_val + pred_delta
        last_expanded = last_values.unsqueeze(2)  # (batch, n_windows, 1, num_streams)
        pred_values = last_expanded + pred_aligned
        
        # Actual values for comparison
        actual_windows = []
        for i in range(actual_deltas.shape[1]):
            start = (i + 1) * window_size
            end = (i + 2) * window_size
            actual_windows.append(targets[:, start:end, :])
        actual_values = torch.stack(actual_windows, dim=1)
        
        # Compute baseline (predict last value = 0 delta)
        baseline_mse = F.mse_loss(last_expanded.expand_as(actual_values), actual_values).item()
        
        # Value-space metrics for interpretability
        value_mse = F.mse_loss(pred_values, actual_values).item()
        
        loss_details = {
            "mse": mse_loss.item(),  # Delta-space MSE (what we optimize)
            "rmse": mse_loss.item() ** 0.5,
            "mae": F.l1_loss(pred_aligned, actual_deltas).item(),
            "value_mse": value_mse,  # Value-space MSE (for comparison)
            "value_rmse": value_mse ** 0.5,
            "baseline_mse": baseline_mse,
            "baseline_rmse": baseline_mse ** 0.5,
            "skill_score": 1.0 - (value_mse / (baseline_mse + 1e-8)),  # Skill in value space
        }
        
        return mse_loss, loss_details
    
    def reconstruct_values(
        self,
        predictions: Tensor,
        values: Tensor,
    ) -> Tensor:
        """
        Reconstruct absolute values from delta predictions.
        
        Args:
            predictions: (batch, num_windows, window_size, num_streams) - predicted deltas
            values: (batch, time_steps, num_streams) - input values (to get last known values)
        
        Returns:
            reconstructed: (batch, num_windows, window_size, num_streams) - predicted values
        """
        batch_size, num_windows, window_size, num_streams = predictions.shape
        
        reconstructed = []
        for i in range(num_windows):
            # Last value of window i (the reference point)
            last_idx = (i + 1) * window_size - 1
            if last_idx >= values.shape[1]:
                break
            last_val = values[:, last_idx:last_idx+1, :]  # (batch, 1, num_streams)
            
            # Reconstruct: value = last_val + delta (absolute)
            pred_delta = predictions[:, i]  # (batch, window_size, num_streams)
            pred_values = last_val + pred_delta
            reconstructed.append(pred_values)
        
        if len(reconstructed) == 0:
            return predictions  # Fallback
        
        return torch.stack(reconstructed, dim=1)
    
    @torch.no_grad()
    def predict_next(self, values: Tensor, return_deltas: bool = False) -> Tensor:
        """
        Predict the next window of values.
        
        Args:
            values: Input time series (batch, time_steps, num_streams)
            return_deltas: If True, return raw delta predictions. If False, return reconstructed values.
        
        Returns:
            next_window: (batch, window_size, num_streams) - predicted values or deltas
        """
        outputs = self.forward(values)
        delta_preds = outputs["predictions"][:, -1]  # (batch, window_size, num_streams)
        
        if return_deltas:
            return delta_preds
        
        # Reconstruct values from deltas (absolute)
        window_size = self.config.window_size
        num_windows = values.shape[1] // window_size
        last_idx = num_windows * window_size - 1
        last_val = values[:, last_idx:last_idx+1, :]  # (batch, 1, num_streams)
        
        return last_val + delta_preds
    
    @torch.no_grad()
    def predict_future(
        self,
        values: Tensor,
        num_windows: int = 1,
    ) -> Tensor:
        """
        Autoregressively predict multiple future windows.
        
        Args:
            values: Input time series (batch, time_steps, num_streams)
            num_windows: Number of future windows to predict
        
        Returns:
            future: (batch, num_windows * window_size, num_streams)
        """
        current = values
        predictions = []
        
        for _ in range(num_windows):
            next_window = self.predict_next(current)
            predictions.append(next_window)
            # Append prediction to input for next iteration
            current = torch.cat([current, next_window], dim=1)
        
        return torch.cat(predictions, dim=1)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_mamba_model(
    num_streams: int,
    window_size: int = 20,
    num_bins: int = 64,
    d_model: int = 192,
    n_layers: int = 4,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.1,
    delta_mode: str = "percent",
) -> TimeSeriesMamba:
    """
    Factory function to create a TimeSeriesMamba model.
    
    Convenience wrapper around TimeSeriesMambaConfig + TimeSeriesMamba.
    """
    config = TimeSeriesMambaConfig(
        num_streams=num_streams,
        window_size=window_size,
        num_bins=num_bins,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        delta_mode=delta_mode,
    )
    return TimeSeriesMamba(config)
