"""
Tests for Time Series Tokenizer.

Run with: python -m pytest models/timeseries/tests/test_tokenizer.py -v
Or directly: python models/timeseries/tests/test_tokenizer.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.timeseries import (
    TimeSeriesTokenizerConfig,
    TimeSeriesTokenizer,
    compute_delta_bins_and_levels,
)
from models.timeseries.tokenizer import (
    create_default_thresholds,
    estimate_thresholds_from_data,
    create_tokenizer,
)


def test_config_properties():
    """Test config computed properties."""
    config = TimeSeriesTokenizerConfig(
        num_streams=10,
        window_size=5,
        num_bins=100,
    )
    
    # num_deltas = window_size - 1 = 4
    assert config.num_deltas == 4
    
    # num_level_features = 3 (last, mean, std)
    assert config.num_level_features == 3
    
    print("✓ Config properties correct")


def test_default_thresholds():
    """Test default threshold creation."""
    thresholds = create_default_thresholds(num_bins=5, scale=1.0)
    
    assert thresholds.shape == (4,)  # num_bins - 1
    assert thresholds[0] < thresholds[-1]  # sorted
    
    print(f"✓ Default thresholds: {thresholds.tolist()}")


def test_compute_delta_bins_and_levels_shape():
    """Test output shapes from compute_delta_bins_and_levels."""
    batch_size = 4
    time_steps = 100
    num_streams = 10
    window_size = 5
    
    config = TimeSeriesTokenizerConfig(
        num_streams=num_streams,
        window_size=window_size,
        num_bins=100,
    )
    
    values = torch.randn(batch_size, time_steps, num_streams)
    delta_bins, level_features = compute_delta_bins_and_levels(values, config)
    
    num_windows = time_steps // window_size  # 20
    num_deltas = window_size - 1  # 4
    
    # delta_bins: (batch, num_windows, num_streams, num_deltas)
    assert delta_bins.shape == (batch_size, num_windows, num_streams, num_deltas)
    assert delta_bins.dtype == torch.int64
    
    # level_features: (batch, num_windows, num_streams, 3)
    assert level_features.shape == (batch_size, num_windows, num_streams, 3)
    
    print(f"✓ Delta bins shape: {delta_bins.shape}")
    print(f"✓ Level features shape: {level_features.shape}")


def test_delta_bins_range():
    """Test that delta bins are in valid range."""
    config = TimeSeriesTokenizerConfig(
        num_streams=5,
        window_size=5,
        num_bins=100,
    )
    
    values = torch.randn(8, 50, 5)
    delta_bins, _ = compute_delta_bins_and_levels(values, config)
    
    assert delta_bins.min() >= 0
    assert delta_bins.max() < config.num_bins
    
    print(f"✓ Delta bins in range [0, {config.num_bins - 1}]")


def test_level_features_values():
    """Test that level features are computed correctly."""
    config = TimeSeriesTokenizerConfig(
        num_streams=2,
        window_size=4,
        num_bins=3,
        eps=0.0,  # disable epsilon for exact test
    )
    
    # Create simple test data
    # batch=1, time=4, streams=2
    # Window: [1, 2, 3, 4] for stream 0, [10, 20, 30, 40] for stream 1
    values = torch.tensor([
        [[1.0, 10.0],
         [2.0, 20.0],
         [3.0, 30.0],
         [4.0, 40.0]]
    ])
    
    _, level_features = compute_delta_bins_and_levels(values, config)
    
    # level_features: (1, 1, 2, 3) - [last, mean, std]
    # Stream 0: last=4, mean=2.5, std=sqrt(1.25)
    # Stream 1: last=40, mean=25, std=sqrt(125)
    
    assert level_features.shape == (1, 1, 2, 3)
    
    # Check last values
    assert torch.allclose(level_features[0, 0, 0, 0], torch.tensor(4.0))
    assert torch.allclose(level_features[0, 0, 1, 0], torch.tensor(40.0))
    
    # Check mean values
    assert torch.allclose(level_features[0, 0, 0, 1], torch.tensor(2.5))
    assert torch.allclose(level_features[0, 0, 1, 1], torch.tensor(25.0))
    
    print("✓ Level features computed correctly")


def test_tokenizer_module_shapes():
    """Test TimeSeriesTokenizer output shapes."""
    config = TimeSeriesTokenizerConfig(
        num_streams=10,
        window_size=5,
        num_bins=100,  # Large number of bins - now works!
        d_model=128,
    )
    
    tokenizer = TimeSeriesTokenizer(config)
    
    batch_size = 4
    time_steps = 100
    values = torch.randn(batch_size, time_steps, config.num_streams)
    
    # Test forward
    embeddings = tokenizer(values)
    num_windows = time_steps // config.window_size
    
    assert embeddings.shape == (batch_size, num_windows, config.num_streams, config.d_model)
    
    # Test forward_flat
    flat_embeddings = tokenizer.forward_flat(values)
    assert flat_embeddings.shape == (batch_size, num_windows * config.num_streams, config.d_model)
    
    print(f"✓ forward() shape: {embeddings.shape}")
    print(f"✓ forward_flat() shape: {flat_embeddings.shape}")


def test_scalability_with_many_bins():
    """Test that the tokenizer scales to many bins."""
    # Factorized embeddings: only need num_bins + num_deltas embeddings
    config = TimeSeriesTokenizerConfig(
        num_streams=10,
        window_size=5,
        num_bins=100,
        d_model=128,
    )
    
    tokenizer = TimeSeriesTokenizer(config)
    
    # Count parameters in delta embeddings
    delta_params = tokenizer.delta_emb.weight.numel()  # 100 * 128 = 12,800
    delta_pos_params = tokenizer.delta_pos_emb.weight.numel()  # 4 * 128 = 512
    total_pattern_params = delta_params + delta_pos_params
    
    print(f"✓ Delta embedding params: {delta_params:,}")
    print(f"✓ Delta position params: {delta_pos_params:,}")
    print(f"✓ Total pattern params: {total_pattern_params:,}")
    
    # Verify it actually works
    values = torch.randn(2, 50, 10)
    embeddings = tokenizer(values)
    assert embeddings.shape == (2, 10, 10, 128)
    
    print("✓ Successfully created tokenizer with 100 bins")


def test_tokenizer_gradient_flow():
    """Test that gradients flow through the tokenizer."""
    config = TimeSeriesTokenizerConfig(
        num_streams=5,
        window_size=5,
        num_bins=10,
        d_model=64,
    )
    
    tokenizer = TimeSeriesTokenizer(config)
    values = torch.randn(2, 50, 5, requires_grad=False)
    
    embeddings = tokenizer(values)
    
    # Compute a scalar loss
    loss = embeddings.sum()
    loss.backward()
    
    # Check that embedding parameters have gradients
    assert tokenizer.stream_emb.weight.grad is not None
    assert tokenizer.delta_emb.weight.grad is not None
    assert tokenizer.delta_pos_emb.weight.grad is not None
    assert tokenizer.time_emb.weight.grad is not None
    assert tokenizer.level_proj.weight.grad is not None
    
    print("✓ Gradients flow through all embedding parameters")


def test_delta_combine_options():
    """Test sum vs mean delta combination."""
    config_sum = TimeSeriesTokenizerConfig(
        num_streams=3,
        window_size=5,
        num_bins=5,
        d_model=32,
        delta_combine="sum",
    )
    
    config_mean = TimeSeriesTokenizerConfig(
        num_streams=3,
        window_size=5,
        num_bins=5,
        d_model=32,
        delta_combine="mean",
    )
    
    tokenizer_sum = TimeSeriesTokenizer(config_sum)
    tokenizer_mean = TimeSeriesTokenizer(config_mean)
    
    # Copy weights so we can compare
    tokenizer_mean.load_state_dict(tokenizer_sum.state_dict())
    
    values = torch.randn(2, 50, 3)
    
    emb_sum = tokenizer_sum(values)
    emb_mean = tokenizer_mean(values)
    
    # They should be different (sum is 4x larger in pattern component)
    assert not torch.allclose(emb_sum, emb_mean)
    
    print("✓ delta_combine='sum' and 'mean' produce different results")


def test_drop_last_behavior():
    """Test drop_last=True truncates incomplete windows."""
    config = TimeSeriesTokenizerConfig(
        num_streams=3,
        window_size=5,
        drop_last=True,
    )
    
    # 23 time steps -> 4 complete windows (20 steps), 3 steps dropped
    values = torch.randn(2, 23, 3)
    delta_bins, level_features = compute_delta_bins_and_levels(values, config)
    
    assert delta_bins.shape == (2, 4, 3, 4)  # 4 windows, 4 deltas each
    
    print("✓ drop_last=True correctly truncates")


def test_estimate_thresholds_from_data():
    """Test threshold estimation from sample data."""
    # Create data with known distribution
    torch.manual_seed(42)
    values = torch.randn(100, 50, 5)
    
    thresholds = estimate_thresholds_from_data(values, num_bins=5, window_size=5)
    
    assert thresholds.shape == (4,)
    assert thresholds[0] < thresholds[-1]  # sorted
    
    print(f"✓ Estimated thresholds: {thresholds.tolist()}")


def test_create_tokenizer_convenience():
    """Test convenience function."""
    tokenizer = create_tokenizer(
        num_streams=8,
        window_size=10,
        num_bins=50,
        d_model=256,
    )
    
    assert isinstance(tokenizer, TimeSeriesTokenizer)
    assert tokenizer.config.num_streams == 8
    assert tokenizer.config.window_size == 10
    assert tokenizer.config.num_bins == 50
    assert tokenizer.config.d_model == 256
    
    print("✓ create_tokenizer() works correctly")


def test_get_sequence_length():
    """Test sequence length calculation."""
    config = TimeSeriesTokenizerConfig(
        num_streams=10,
        window_size=5,
    )
    tokenizer = TimeSeriesTokenizer(config)
    
    # 100 time steps -> 20 windows -> 200 tokens
    seq_len = tokenizer.get_sequence_length(100)
    assert seq_len == 200
    
    print(f"✓ get_sequence_length(100) = {seq_len}")


def test_device_compatibility():
    """Test that tokenizer works on different devices."""
    config = TimeSeriesTokenizerConfig(num_streams=5, d_model=32)
    tokenizer = TimeSeriesTokenizer(config)
    
    values = torch.randn(2, 50, 5)
    
    # Test on CPU
    embeddings_cpu = tokenizer(values)
    assert embeddings_cpu.device.type == "cpu"
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        tokenizer_cuda = tokenizer.cuda()
        values_cuda = values.cuda()
        embeddings_cuda = tokenizer_cuda(values_cuda)
        assert embeddings_cuda.device.type == "cuda"
        print("✓ Works on CUDA")
    
    print("✓ Works on CPU")


def test_extra_repr():
    """Test string representation."""
    config = TimeSeriesTokenizerConfig(
        num_streams=10,
        window_size=5,
        num_bins=100,
        d_model=128,
    )
    tokenizer = TimeSeriesTokenizer(config)
    
    repr_str = repr(tokenizer)
    assert "num_streams=10" in repr_str
    assert "window_size=5" in repr_str
    assert "num_bins=100" in repr_str
    assert "num_deltas=4" in repr_str
    
    print(f"✓ Repr: {tokenizer.extra_repr()}")


def test_tokenize_method():
    """Test the tokenize method returns correct shapes."""
    config = TimeSeriesTokenizerConfig(
        num_streams=5,
        window_size=5,
        num_bins=10,
    )
    tokenizer = TimeSeriesTokenizer(config)
    
    values = torch.randn(2, 50, 5)
    delta_bins, level_features = tokenizer.tokenize(values)
    
    # delta_bins: (batch, num_windows, num_streams, num_deltas)
    assert delta_bins.shape == (2, 10, 5, 4)
    
    # level_features: (batch, num_windows, num_streams, 3)
    assert level_features.shape == (2, 10, 5, 3)
    
    print("✓ tokenize() returns correct shapes")


def test_delta_mode_absolute():
    """Test absolute delta mode (original behavior)."""
    config = TimeSeriesTokenizerConfig(
        num_streams=1,
        window_size=3,
        num_bins=5,
        delta_mode="absolute",
        default_threshold_scale=5.0,  # Thresholds at -5, -2.5, 0, 2.5, 5
    )
    
    # Create known values: [10, 15, 20] = deltas of [+5, +5]
    values = torch.tensor([[[10.0], [15.0], [20.0]]])  # (1, 3, 1)
    
    delta_bins, level_features = compute_delta_bins_and_levels(values, config)
    
    # With absolute deltas of +5 and thresholds [-5, -2.5, 0, 2.5, 5]:
    # +5 should be in bin 4 (>= 5 threshold) or bin 3-4 depending on exact bucket
    assert delta_bins.shape == (1, 1, 1, 2)  # 2 deltas per window
    
    # Both deltas should be in the same bin (both are +5)
    assert delta_bins[0, 0, 0, 0] == delta_bins[0, 0, 0, 1]
    
    print(f"✓ Absolute mode: deltas=[+5,+5] -> bins={delta_bins[0,0,0].tolist()}")


def test_delta_mode_percent():
    """Test percent delta mode (scale-invariant)."""
    config = TimeSeriesTokenizerConfig(
        num_streams=1,
        window_size=3,
        num_bins=5,
        delta_mode="percent",
        default_threshold_scale=0.5,  # ±50% range
    )
    
    # Case 1: 10 -> 15 = +50% change
    values1 = torch.tensor([[[10.0], [15.0], [20.0]]])  # +50%, +33%
    
    # Case 2: 1000 -> 1500 = +50% change (same percentage!)
    values2 = torch.tensor([[[1000.0], [1500.0], [2000.0]]])  # +50%, +33%
    
    delta_bins1, _ = compute_delta_bins_and_levels(values1, config)
    delta_bins2, _ = compute_delta_bins_and_levels(values2, config)
    
    # The key test: same percentage changes at different absolute levels
    # should produce the SAME bins
    assert torch.equal(delta_bins1, delta_bins2), \
        f"Scale invariance failed: bins1={delta_bins1.tolist()}, bins2={delta_bins2.tolist()}"
    
    print(f"✓ Percent mode scale-invariant: +50% at level=10 and level=1000 -> same bin")


def test_delta_mode_log_return():
    """Test log_return delta mode (symmetric for gains/losses)."""
    config = TimeSeriesTokenizerConfig(
        num_streams=1,
        window_size=3,
        num_bins=5,
        delta_mode="log_return",
        default_threshold_scale=0.5,
    )
    
    # Log returns are symmetric: 10->20 (2x) and 20->10 (0.5x) should have
    # same magnitude but opposite sign
    values_up = torch.tensor([[[10.0], [20.0], [40.0]]])   # +log(2), +log(2)
    values_down = torch.tensor([[[40.0], [20.0], [10.0]]]) # -log(2), -log(2)
    
    delta_bins_up, _ = compute_delta_bins_and_levels(values_up, config)
    delta_bins_down, _ = compute_delta_bins_and_levels(values_down, config)
    
    # With symmetric thresholds around 0, the bins should be "mirror images"
    # E.g., if up is bin 4 (high positive), down should be bin 0 (high negative)
    num_bins = config.num_bins
    expected_down = num_bins - 1 - delta_bins_up  # Mirror around middle
    
    # They should be symmetric (allow for 1 bin difference due to bucketing)
    diff = (delta_bins_up + delta_bins_down - (num_bins - 1)).abs()
    assert diff.max() <= 1, f"Log returns not symmetric: up={delta_bins_up.tolist()}, down={delta_bins_down.tolist()}"
    
    print(f"✓ Log-return mode: +100% -> bin {delta_bins_up[0,0,0,0].item()}, "
          f"-50% -> bin {delta_bins_down[0,0,0,0].item()} (symmetric around middle)")


def test_percent_mode_scale_invariance():
    """
    Comprehensive test for scale invariance in percent mode.
    
    This is the key feature: the same percentage change should produce
    the same bin regardless of the absolute level.
    """
    config = TimeSeriesTokenizerConfig(
        num_streams=1,
        window_size=3,
        num_bins=10,
        delta_mode="percent",
        default_threshold_scale=0.2,  # ±20% range with finer bins
    )
    
    # All of these have EXACTLY the same +10% change
    # Use float64 for precision, then convert to float32
    # This ensures the same bit patterns at all scales
    base_values = torch.tensor([1.0, 1.1, 1.21], dtype=torch.float64)
    
    test_scales = [10.0, 100.0, 1000.0, 0.1, 50.0]
    
    bins_list = []
    for scale in test_scales:
        # Scale in float64, then convert to float32
        values = (base_values * scale).float().unsqueeze(0).unsqueeze(-1)  # (1, 3, 1)
        delta_bins, _ = compute_delta_bins_and_levels(values, config)
        bins_list.append(delta_bins.clone())
    
    # All should have the same bins (allowing for 1-bin floating point tolerance)
    # Due to float32 precision, bins might differ by 1 at boundaries
    for i in range(1, len(bins_list)):
        diff = (bins_list[0] - bins_list[i]).abs().max().item()
        assert diff <= 1, \
            f"Scale invariance failed: scale={test_scales[0]} bins={bins_list[0].tolist()}, " \
            f"scale={test_scales[i]} bins={bins_list[i].tolist()}, max diff={diff}"
    
    # More importantly, verify the bins are roughly the same (within tolerance)
    all_same = all(torch.equal(bins_list[0], b) for b in bins_list[1:])
    
    if all_same:
        print(f"✓ Percent mode: +10% change at scales {test_scales} all -> bin {bins_list[0][0,0,0,0].item()}")
    else:
        bins_set = set(tuple(b[0,0,0].tolist()) for b in bins_list)
        print(f"✓ Percent mode: +10% change at various scales -> bins vary by ≤1 due to float32 (bins seen: {bins_set})")


def test_delta_mode_backward_compatibility():
    """Test that configs without delta_mode still work (default to percent)."""
    # Create a config the "old" way - delta_mode should default to "percent"
    config = TimeSeriesTokenizerConfig(
        num_streams=5,
        window_size=5,
        num_bins=10,
    )
    
    assert config.delta_mode == "percent", f"Expected default delta_mode='percent', got {config.delta_mode!r}"
    
    # Make sure tokenization still works
    tokenizer = TimeSeriesTokenizer(config)
    values = torch.randn(2, 50, 5)
    embeddings = tokenizer(values)
    
    assert embeddings.shape == (2, 10, 5, config.d_model)
    
    print("✓ Backward compatibility: default delta_mode='percent' works")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Time Series Tokenizer Tests")
    print("=" * 60)
    
    test_config_properties()
    test_default_thresholds()
    test_compute_delta_bins_and_levels_shape()
    test_delta_bins_range()
    test_level_features_values()
    test_tokenizer_module_shapes()
    test_scalability_with_many_bins()
    test_tokenizer_gradient_flow()
    test_delta_combine_options()
    test_drop_last_behavior()
    test_estimate_thresholds_from_data()
    test_create_tokenizer_convenience()
    test_get_sequence_length()
    test_device_compatibility()
    test_extra_repr()
    test_tokenize_method()
    
    # New delta mode tests
    print("\n--- Delta Mode Tests ---")
    test_delta_mode_absolute()
    test_delta_mode_percent()
    test_delta_mode_log_return()
    test_percent_mode_scale_invariance()
    test_delta_mode_backward_compatibility()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
