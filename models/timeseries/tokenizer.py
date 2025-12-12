"""
Time Series Tokenizer for Transformer/Mamba Models.

This module converts multivariate time series into discrete "pattern tokens" 
plus continuous level features, suitable for sequence modeling architectures.

Concept Overview
----------------
- A time series is divided into non-overlapping windows (e.g., 5 minutes each).
- Each window × stream becomes a "token" that encodes:
    1. Stream identity (which sensor/variable)
    2. Derivative pattern (discrete binned deltas within the window)
    3. Level information (continuous statistics: last, mean, std)
    4. Temporal position (window index)

Pattern Encoding (Factorized)
-----------------------------
For a window of length W, we have W-1 deltas. Each delta is binned into
one of K bins using thresholds. We use factorized embeddings:

1. Embed each delta bin separately using a shared delta_emb table (K embeddings)
2. Add position information to distinguish delta[0] from delta[1], etc. (W-1 embeddings)
3. Combine the delta embeddings by summing or averaging

This scales to any number of bins with only O(K + W) parameters.

Factorized Embeddings
---------------------
The final token embedding is:

    token_emb = stream_emb + pattern_emb + time_emb + level_proj(level_features)

Where pattern_emb is the combined (summed/averaged) delta embeddings.

Usage Example
-------------
    config = TimeSeriesTokenizerConfig(
        num_streams=10,
        window_size=5,
        num_bins=100,  # Can now use many bins!
        d_model=128,
    )
    tokenizer = TimeSeriesTokenizer(config)
    
    # values: (batch, time_steps, num_streams)
    values = torch.randn(32, 100, 10)
    
    # Get embeddings: (batch, num_windows, num_streams, d_model)
    embeddings = tokenizer(values)
    
    # Or get raw delta bins and levels
    delta_bins, level_features = tokenizer.tokenize(values)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TimeSeriesTokenizerConfig:
    """
    Configuration for TimeSeriesTokenizer.
    
    Attributes:
        num_streams: Number of input streams/sensors/variables.
        window_size: Number of time steps per window (e.g., 5 for 5-minute windows).
        num_bins: Number of bins for discretizing delta values. Can be large (e.g., 100).
        d_model: Embedding dimension for the output tokens.
        max_windows: Maximum number of windows to support (for time embeddings).
        delta_mode: How to compute deltas between consecutive time steps:
            - "absolute": Raw difference x[i] - x[i-1] (original behavior)
            - "percent": Percentage change (x[i] - x[i-1]) / |x[i-1]| (default)
              This makes the pattern encoding scale-invariant: a +5% change
              gets the same bin whether at level=10 or level=1000.
            - "log_return": Log return log(x[i] / x[i-1]) (common in finance)
              Symmetric for gains/losses and additive across time.
        delta_thresholds: Optional tensor of shape (num_bins - 1,) specifying bin edges.
            If None, symmetric thresholds are created using default_threshold_scale.
        default_threshold_scale: Scale for auto-generated symmetric thresholds.
            Thresholds will span [-scale, scale] evenly.
            Recommended values by delta_mode:
            - "absolute": 1.0 (depends on your data scale)
            - "percent": 0.1 (±10% changes, common for many applications)
            - "log_return": 0.1 (similar to percent for small changes)
        eps: Epsilon for numerical stability in division and std computation.
        drop_last: If True, drop the last incomplete window. If False, raise error
            when time_steps is not divisible by window_size.
        delta_combine: How to combine delta embeddings into pattern embedding.
            - "sum": Sum all delta embeddings
            - "mean": Average all delta embeddings (default, more stable)
    
    Properties:
        num_deltas: Number of deltas per window = window_size - 1.
        num_level_features: Number of continuous level features (currently 3).
    
    Why Percentage Mode?
    --------------------
    With absolute deltas, a +5 change is treated the same whether the value
    went from 10→15 (50% increase) or 1000→1005 (0.5% increase). This loses
    important relative magnitude information.
    
    With percentage mode:
    - 10→15 gives delta = +0.5 (50% increase)
    - 1000→1005 gives delta = +0.005 (0.5% increase)
    
    The pattern embedding now captures the *relative* movement, while the
    level features (last, mean, std) capture the absolute scale. This is
    a much more informative factorization for most time series tasks.
    """
    num_streams: int
    window_size: int = 5
    num_bins: int = 5
    d_model: int = 128
    max_windows: int = 1024
    delta_mode: Literal["absolute", "percent", "log_return"] = "percent"
    delta_thresholds: Optional[Tensor] = None
    default_threshold_scale: float = 0.1  # Default for percent mode
    eps: float = 1e-8
    drop_last: bool = True
    delta_combine: Literal["sum", "mean"] = "mean"
    
    @property
    def num_deltas(self) -> int:
        """Number of deltas per window = window_size - 1."""
        return self.window_size - 1
    
    @property
    def num_level_features(self) -> int:
        """Number of continuous level features per token."""
        # Currently: last_value, mean_value, std_value
        return 3
    
    def get_delta_thresholds(self) -> Tensor:
        """
        Get or create delta thresholds for binning.
        
        The thresholds determine the bin edges for discretizing delta values.
        If custom thresholds are provided, they are used directly.
        Otherwise, symmetric thresholds are auto-generated using default_threshold_scale.
        
        Recommended default_threshold_scale by delta_mode:
        
        - "absolute": Depends on your data scale. If values typically change by
          ±10 units, use scale=10. Examine your data's delta distribution.
          
        - "percent" (default): Use 0.1 for ±10% changes. This means:
          - Bin 0: < -10% (strong decrease)
          - Middle bins: -10% to +10% (small changes)
          - Last bin: > +10% (strong increase)
          For volatile data (stocks), use 0.2-0.5. For stable data, use 0.05.
          
        - "log_return": Similar to percent for small changes (log(1.1) ≈ 0.095).
          Use 0.1 for most applications. Log returns are symmetric, so ±10% 
          log return corresponds to roughly ±10.5% actual change.
        
        Returns:
            Tensor of shape (num_bins - 1,) with sorted threshold values.
        
        Example:
            # For 5 bins with scale=0.1, thresholds are:
            # [-0.1, -0.033, 0.033, 0.1]
            # Bins represent: <-10%, -10% to -3.3%, -3.3% to +3.3%, +3.3% to +10%, >+10%
        """
        if self.delta_thresholds is not None:
            return self.delta_thresholds
        
        # Create symmetric thresholds around 0
        num_thresholds = self.num_bins - 1
        scale = self.default_threshold_scale
        
        if num_thresholds == 1:
            return torch.tensor([0.0])
        
        # Evenly spaced from -scale to +scale
        thresholds = torch.linspace(-scale, scale, num_thresholds)
        return thresholds


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_delta_bins_and_levels(
    values: Tensor,
    config: TimeSeriesTokenizerConfig,
) -> Tuple[Tensor, Tensor]:
    """
    Compute binned deltas and level features from raw time series values.
    
    This function:
    1. Reshapes the input into non-overlapping windows
    2. Computes level features (last, mean, std) for each window
    3. Computes deltas within each window
    4. Bins deltas into discrete values
    
    Args:
        values: Input tensor of shape (batch_size, time_steps, num_streams).
        config: TimeSeriesTokenizerConfig instance.
    
    Returns:
        delta_bins: LongTensor of shape (batch_size, num_windows, num_streams, num_deltas).
            Each value is in [0, num_bins-1].
        level_features: FloatTensor of shape (batch_size, num_windows, num_streams, num_level_features).
            Contains [last_value, mean_value, std_value] for each window.
    
    Raises:
        ValueError: If time_steps is not divisible by window_size and drop_last=False.
    
    Example:
        >>> values = torch.randn(2, 20, 3)  # batch=2, time=20, streams=3
        >>> config = TimeSeriesTokenizerConfig(num_streams=3, window_size=5)
        >>> delta_bins, levels = compute_delta_bins_and_levels(values, config)
        >>> delta_bins.shape  # (2, 4, 3, 4)  -- 4 deltas per window
        >>> levels.shape      # (2, 4, 3, 3)
    """
    batch_size, time_steps, num_streams = values.shape
    window_size = config.window_size
    device = values.device
    dtype = values.dtype
    
    # Validate and compute number of windows
    if time_steps < window_size:
        raise ValueError(
            f"time_steps ({time_steps}) must be >= window_size ({window_size})"
        )
    
    num_windows = time_steps // window_size
    remainder = time_steps % window_size
    
    if remainder != 0:
        if config.drop_last:
            # Truncate to fit exact windows
            usable_steps = num_windows * window_size
            values = values[:, :usable_steps, :]
        else:
            raise ValueError(
                f"time_steps ({time_steps}) is not divisible by window_size ({window_size}). "
                f"Set drop_last=True to truncate the remainder."
            )
    
    # Reshape into windows: (batch, num_windows, window_size, num_streams)
    windowed = values.view(batch_size, num_windows, window_size, num_streams)
    
    # =========================================================================
    # Compute Level Features
    # =========================================================================
    # last_value: last time step in each window
    last_value = windowed[:, :, -1, :]  # (batch, num_windows, num_streams)
    
    # mean_value: mean across time steps in window
    mean_value = windowed.mean(dim=2)  # (batch, num_windows, num_streams)
    
    # std_value: std across time steps in window (with epsilon for stability)
    std_value = windowed.std(dim=2, unbiased=False) + config.eps  # (batch, num_windows, num_streams)
    
    # Stack level features: (batch, num_windows, num_streams, 3)
    level_features = torch.stack([last_value, mean_value, std_value], dim=-1)
    
    # =========================================================================
    # Compute Derivative Pattern (Binned Deltas)
    # =========================================================================
    # Compute raw deltas within each window: x[i] - x[i-1] for i in 1..window_size-1
    # windowed shape: (batch, num_windows, window_size, num_streams)
    raw_deltas = windowed[:, :, 1:, :] - windowed[:, :, :-1, :]
    # raw_deltas shape: (batch, num_windows, window_size-1, num_streams)
    
    # Apply delta_mode transformation
    # This determines whether we use absolute differences, percentages, or log returns
    delta_mode = getattr(config, 'delta_mode', 'percent')  # Backward compat default
    
    if delta_mode == "absolute":
        # Original behavior: raw difference
        # Use case: When absolute changes matter (e.g., temperature in Celsius)
        deltas = raw_deltas
        
    elif delta_mode == "percent":
        # Percentage change: (x[i] - x[i-1]) / |x[i-1]|
        # This makes the encoding scale-invariant:
        #   - 10 -> 15 gives +0.5 (50% increase)
        #   - 1000 -> 1005 gives +0.005 (0.5% increase)
        # The level features capture absolute scale, pattern captures relative movement
        prev_values = windowed[:, :, :-1, :]  # x[i-1]
        # Use abs() + eps to avoid division by zero and handle negative values
        denominator = prev_values.abs().clamp(min=config.eps)
        deltas = raw_deltas / denominator
        
    elif delta_mode == "log_return":
        # Log return: log(x[i] / x[i-1])
        # Properties:
        #   - Symmetric for gains/losses: +10% and -10% have same magnitude
        #   - Additive across time: sum of log returns = total log return
        #   - Common in finance for this reason
        # Note: Requires positive values; we use abs() + eps for safety
        curr_values = windowed[:, :, 1:, :]   # x[i]
        prev_values = windowed[:, :, :-1, :]  # x[i-1]
        # Clamp both to positive values
        curr_safe = curr_values.abs().clamp(min=config.eps)
        prev_safe = prev_values.abs().clamp(min=config.eps)
        deltas = torch.log(curr_safe / prev_safe)
        
    else:
        raise ValueError(
            f"Unknown delta_mode: {delta_mode!r}. "
            f"Expected 'absolute', 'percent', or 'log_return'."
        )
    
    # deltas shape: (batch, num_windows, window_size-1, num_streams)
    
    # Get thresholds for binning
    thresholds = config.get_delta_thresholds().to(device=device, dtype=dtype)
    # thresholds shape: (num_bins - 1,)
    
    # Bin each delta using thresholds
    # torch.bucketize maps values to bin indices based on thresholds
    # For thresholds [t0, t1, t2, t3], bucketize returns:
    #   0 if x < t0
    #   1 if t0 <= x < t1
    #   2 if t1 <= x < t2
    #   3 if t2 <= x < t3
    #   4 if x >= t3
    binned_deltas = torch.bucketize(deltas, thresholds)
    # binned_deltas shape: (batch, num_windows, num_deltas, num_streams)
    # values in [0, num_bins-1]
    
    # Transpose to (batch, num_windows, num_streams, num_deltas) to match level_features layout
    delta_bins = binned_deltas.permute(0, 1, 3, 2).contiguous()
    
    return delta_bins, level_features


def create_default_thresholds(
    num_bins: int,
    scale: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Create default symmetric thresholds for delta binning.
    
    Creates evenly-spaced thresholds from -scale to +scale.
    
    Args:
        num_bins: Number of bins (requires num_bins >= 2).
        scale: Range of thresholds (thresholds span [-scale, scale]).
        device: Device for the returned tensor.
    
    Returns:
        Tensor of shape (num_bins - 1,) with threshold values.
    
    Example:
        >>> create_default_thresholds(5, scale=1.0)
        tensor([-1.0000, -0.3333,  0.3333,  1.0000])
    """
    if num_bins < 2:
        raise ValueError(f"num_bins must be >= 2, got {num_bins}")
    
    num_thresholds = num_bins - 1
    thresholds = torch.linspace(-scale, scale, num_thresholds, device=device)
    return thresholds


def estimate_thresholds_from_data(
    values: Tensor,
    num_bins: int,
    window_size: int,
) -> Tensor:
    """
    Estimate delta thresholds from data using percentiles.
    
    Computes deltas from the input data and finds percentile-based
    thresholds that create roughly equal-sized bins.
    
    Args:
        values: Sample data tensor of shape (batch_size, time_steps, num_streams).
        num_bins: Number of bins to create.
        window_size: Window size for delta computation.
    
    Returns:
        Tensor of shape (num_bins - 1,) with threshold values.
    
    Example:
        >>> values = torch.randn(100, 50, 5)  # sample data
        >>> thresholds = estimate_thresholds_from_data(values, num_bins=5, window_size=5)
    """
    # Compute all deltas from the data
    batch_size, time_steps, num_streams = values.shape
    num_windows = time_steps // window_size
    
    if num_windows == 0:
        raise ValueError("Not enough time steps to form windows")
    
    # Reshape into windows
    usable_steps = num_windows * window_size
    windowed = values[:, :usable_steps, :].view(batch_size, num_windows, window_size, num_streams)
    
    # Compute deltas
    deltas = windowed[:, :, 1:, :] - windowed[:, :, :-1, :]
    deltas_flat = deltas.flatten()
    
    # Compute percentiles for threshold positions
    # For num_bins bins, we need num_bins-1 thresholds
    # Place them at percentiles: 100/num_bins, 200/num_bins, ..., (num_bins-1)*100/num_bins
    percentiles = torch.linspace(0, 100, num_bins + 1)[1:-1]  # exclude 0% and 100%
    
    thresholds = torch.quantile(deltas_flat, percentiles / 100.0)
    return thresholds


# =============================================================================
# MAIN MODULE
# =============================================================================

class TimeSeriesTokenizer(nn.Module):
    """
    PyTorch module for tokenizing multivariate time series.
    
    Converts raw time series values into embedded tokens using factorized embeddings:
    - Stream embeddings (which sensor/variable)
    - Delta embeddings (binned change values) + Delta position embeddings
    - Time embeddings (window position)
    - Level projection (continuous statistics)
    
    The pattern embedding is built by embedding each delta separately and combining
    them (sum or mean), which scales to any number of bins.
    
    Embedding tables:
    - stream_emb: (num_streams, d_model) - identifies which sensor
    - delta_emb: (num_bins, d_model) - encodes delta magnitude/direction
    - delta_pos_emb: (num_deltas, d_model) - encodes position within window
    - time_emb: (max_windows, d_model) - encodes window position in sequence
    - level_proj: (3 -> d_model) - projects continuous features
    
    Example:
        >>> config = TimeSeriesTokenizerConfig(num_streams=10, window_size=5, num_bins=100, d_model=128)
        >>> tokenizer = TimeSeriesTokenizer(config)
        >>> values = torch.randn(32, 100, 10)  # (batch, time, streams)
        >>> embeddings = tokenizer(values)     # (batch, 20, 10, 128)
        >>> flat_embeddings = tokenizer.forward_flat(values)  # (batch, 200, 128)
    """
    
    def __init__(self, config: TimeSeriesTokenizerConfig):
        """
        Initialize the TimeSeriesTokenizer.
        
        Args:
            config: Configuration object with tokenizer parameters.
        """
        super().__init__()
        self.config = config
        
        # Stream embedding: which sensor/variable
        self.stream_emb = nn.Embedding(config.num_streams, config.d_model)
        
        # Delta embeddings (factorized):
        # - delta_emb: what kind of change (shared across all positions)
        # - delta_pos_emb: which position in the window (1st delta, 2nd delta, etc.)
        self.delta_emb = nn.Embedding(config.num_bins, config.d_model)
        self.delta_pos_emb = nn.Embedding(config.num_deltas, config.d_model)
        
        # Time embedding: which window in the sequence
        self.time_emb = nn.Embedding(config.max_windows, config.d_model)
        
        # Projection for continuous level features
        self.level_proj = nn.Linear(config.num_level_features, config.d_model)
        
        # Store thresholds as a buffer (not a parameter, but saved with model)
        thresholds = config.get_delta_thresholds()
        self.register_buffer("delta_thresholds", thresholds)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding and projection weights."""
        # Standard initialization for embeddings
        nn.init.normal_(self.stream_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.delta_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.delta_pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_emb.weight, mean=0.0, std=0.02)
        
        # Xavier initialization for linear layer
        nn.init.xavier_uniform_(self.level_proj.weight)
        nn.init.zeros_(self.level_proj.bias)
    
    def tokenize(self, values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Tokenize raw time series values into delta bins and level features.
        
        This method computes the discrete binned deltas and continuous
        level statistics without applying the embedding layers.
        
        Args:
            values: Input tensor of shape (batch_size, time_steps, num_streams).
        
        Returns:
            delta_bins: LongTensor of shape (batch_size, num_windows, num_streams, num_deltas).
            level_features: FloatTensor of shape (batch_size, num_windows, num_streams, num_level_features).
        """
        # Create config with registered buffer thresholds
        config_with_thresholds = TimeSeriesTokenizerConfig(
            num_streams=self.config.num_streams,
            window_size=self.config.window_size,
            num_bins=self.config.num_bins,
            d_model=self.config.d_model,
            max_windows=self.config.max_windows,
            delta_thresholds=self.delta_thresholds,
            default_threshold_scale=self.config.default_threshold_scale,
            eps=self.config.eps,
            drop_last=self.config.drop_last,
            delta_combine=self.config.delta_combine,
        )
        
        return compute_delta_bins_and_levels(values, config_with_thresholds)
    
    def forward(self, values: Tensor) -> Tensor:
        """
        Convert raw time series to embedded tokens.
        
        Computes factorized embeddings by summing:
        - Stream embedding (which sensor)
        - Pattern embedding (combined delta embeddings)
        - Time embedding (window position)
        - Level projection (continuous statistics)
        
        The pattern embedding is built by:
        1. Looking up each delta bin in delta_emb
        2. Adding delta position embedding (delta_pos_emb)
        3. Combining across deltas (sum or mean based on config.delta_combine)
        
        Args:
            values: Input tensor of shape (batch_size, time_steps, num_streams).
        
        Returns:
            token_embeddings: FloatTensor of shape (batch_size, num_windows, num_streams, d_model).
        
        Example:
            >>> tokenizer = TimeSeriesTokenizer(config)
            >>> values = torch.randn(32, 100, 10)
            >>> embeddings = tokenizer(values)  # (32, 20, 10, 128)
        """
        batch_size = values.shape[0]
        device = values.device
        
        # Get delta bins and level features
        delta_bins, level_features = self.tokenize(values)
        # delta_bins: (batch, num_windows, num_streams, num_deltas)
        # level_features: (batch, num_windows, num_streams, num_level_features)
        
        num_windows = delta_bins.shape[1]
        num_streams = delta_bins.shape[2]
        num_deltas = delta_bins.shape[3]
        
        # =====================================================================
        # Stream Embeddings
        # =====================================================================
        # stream_indices: [0, 1, 2, ..., num_streams-1]
        stream_indices = torch.arange(num_streams, device=device)
        stream_embeddings = self.stream_emb(stream_indices)  # (num_streams, d_model)
        # Broadcast to (batch, num_windows, num_streams, d_model)
        stream_embeddings = stream_embeddings.view(1, 1, num_streams, -1)
        stream_embeddings = stream_embeddings.expand(batch_size, num_windows, -1, -1)
        
        # =====================================================================
        # Pattern Embeddings (Factorized Delta Embeddings)
        # =====================================================================
        # Step 1: Look up delta bin embeddings
        # delta_bins: (batch, num_windows, num_streams, num_deltas) -> lookup
        delta_embeddings = self.delta_emb(delta_bins)
        # delta_embeddings: (batch, num_windows, num_streams, num_deltas, d_model)
        
        # Step 2: Add delta position embeddings
        # delta_pos_indices: [0, 1, 2, ..., num_deltas-1]
        delta_pos_indices = torch.arange(num_deltas, device=device)
        delta_pos_embeddings = self.delta_pos_emb(delta_pos_indices)  # (num_deltas, d_model)
        # Broadcast to match delta_embeddings shape
        delta_pos_embeddings = delta_pos_embeddings.view(1, 1, 1, num_deltas, -1)
        
        # Combine: each delta gets its bin embedding + its position embedding
        delta_embeddings = delta_embeddings + delta_pos_embeddings
        # delta_embeddings: (batch, num_windows, num_streams, num_deltas, d_model)
        
        # Step 3: Combine across deltas (sum or mean)
        if self.config.delta_combine == "sum":
            pattern_embeddings = delta_embeddings.sum(dim=3)
        else:  # "mean"
            pattern_embeddings = delta_embeddings.mean(dim=3)
        # pattern_embeddings: (batch, num_windows, num_streams, d_model)
        
        # =====================================================================
        # Time Embeddings
        # =====================================================================
        # time_indices: [0, 1, 2, ..., num_windows-1]
        time_indices = torch.arange(num_windows, device=device)
        time_embeddings = self.time_emb(time_indices)  # (num_windows, d_model)
        # Broadcast to (batch, num_windows, num_streams, d_model)
        time_embeddings = time_embeddings.view(1, num_windows, 1, -1)
        time_embeddings = time_embeddings.expand(batch_size, -1, num_streams, -1)
        
        # =====================================================================
        # Level Feature Projection
        # =====================================================================
        # level_features: (batch, num_windows, num_streams, num_level_features)
        level_embeddings = self.level_proj(level_features)
        # level_embeddings: (batch, num_windows, num_streams, d_model)
        
        # =====================================================================
        # Combine All Embeddings
        # =====================================================================
        token_embeddings = (
            stream_embeddings
            + pattern_embeddings
            + time_embeddings
            + level_embeddings
        )
        
        return token_embeddings
    
    def forward_flat(self, values: Tensor) -> Tensor:
        """
        Convert raw time series to embedded tokens with flattened sequence dimension.
        
        Same as forward(), but flattens (num_windows, num_streams) into a single
        sequence dimension, suitable for feeding into Transformer/Mamba.
        
        The flattening order is: for each window, iterate over all streams.
        So token 0 is (window 0, stream 0), token 1 is (window 0, stream 1), etc.
        
        Args:
            values: Input tensor of shape (batch_size, time_steps, num_streams).
        
        Returns:
            token_embeddings: FloatTensor of shape (batch_size, num_windows * num_streams, d_model).
        
        Example:
            >>> tokenizer = TimeSeriesTokenizer(config)
            >>> values = torch.randn(32, 100, 10)  # 20 windows, 10 streams
            >>> embeddings = tokenizer.forward_flat(values)  # (32, 200, 128)
        """
        # Get 4D embeddings
        token_embeddings = self.forward(values)
        # token_embeddings: (batch, num_windows, num_streams, d_model)
        
        batch_size, num_windows, num_streams, d_model = token_embeddings.shape
        
        # Flatten (num_windows, num_streams) -> sequence
        token_embeddings = token_embeddings.view(batch_size, num_windows * num_streams, d_model)
        
        return token_embeddings
    
    def get_sequence_length(self, time_steps: int) -> int:
        """
        Calculate the output sequence length for a given input time_steps.
        
        Args:
            time_steps: Number of input time steps.
        
        Returns:
            Sequence length after tokenization (num_windows * num_streams).
        """
        num_windows = time_steps // self.config.window_size
        return num_windows * self.config.num_streams
    
    def extra_repr(self) -> str:
        """Extra representation for print(module)."""
        return (
            f"num_streams={self.config.num_streams}, "
            f"window_size={self.config.window_size}, "
            f"num_bins={self.config.num_bins}, "
            f"num_deltas={self.config.num_deltas}, "
            f"d_model={self.config.d_model}, "
            f"delta_combine={self.config.delta_combine!r}"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_tokenizer(
    num_streams: int,
    window_size: int = 5,
    num_bins: int = 5,
    d_model: int = 128,
    **kwargs,
) -> TimeSeriesTokenizer:
    """
    Convenience function to create a TimeSeriesTokenizer.
    
    Args:
        num_streams: Number of input streams/sensors.
        window_size: Time steps per window.
        num_bins: Number of bins for delta discretization.
        d_model: Embedding dimension.
        **kwargs: Additional arguments passed to TimeSeriesTokenizerConfig.
    
    Returns:
        Configured TimeSeriesTokenizer instance.
    
    Example:
        >>> tokenizer = create_tokenizer(num_streams=10, window_size=5, num_bins=100, d_model=256)
    """
    config = TimeSeriesTokenizerConfig(
        num_streams=num_streams,
        window_size=window_size,
        num_bins=num_bins,
        d_model=d_model,
        **kwargs,
    )
    return TimeSeriesTokenizer(config)


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

def compute_patterns_and_levels(
    values: Tensor,
    config: TimeSeriesTokenizerConfig,
) -> Tuple[Tensor, Tensor]:
    """
    Deprecated: Use compute_delta_bins_and_levels instead.
    
    This function is kept for backward compatibility but now returns delta_bins
    instead of pattern_ids.
    """
    return compute_delta_bins_and_levels(values, config)
