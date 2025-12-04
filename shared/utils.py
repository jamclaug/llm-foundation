#!/usr/bin/env python3
"""
Utility functions for Sparse MoE Transformer.
"""

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_flops(model: nn.Module, seq_len: int = 256) -> float:
    """Very rough FLOP estimate for forward pass (ignores attention constants).
    
    Args:
        model: SparseMoETransformer model
        seq_len: Sequence length for estimation
        
    Returns:
        Estimated FLOPs for forward pass
    """
    cfg = model.config
    L, H, D, F, E, K = cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.top_k

    # Attention per layer: ~ 4 * L * H * D * D * seq_len (QK^T + softmax + PV)
    attn_flops = 4 * L * H * D * D * seq_len

    # Sparse FFN: only K experts active → 2 * K * D * F * seq_len per layer
    ffn_flops = 2 * L * K * D * F * seq_len

    return attn_flops + ffn_flops


@torch.no_grad()
def benchmark_latency(model, tokenizer, device, n_samples=128):
    """Measure average inference latency per sample.
    
    Process:
    1. Warmup runs to stabilize GPU state and cache
    2. Timed runs on single samples (batch=1 for realistic latency)
    3. GPU synchronization to ensure accurate timing
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for input preparation
        device: Device to run on
        n_samples: Number of samples for averaging
        
    Returns:
        Average latency in milliseconds per sample
    """
    model.eval()
    # Prepare test input: simple prompt repeated n_samples times
    inputs = tokenizer(
        ["Once upon a time"] * n_samples,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    # Warmup: run several times to warm up GPU and fill caches
    for _ in range(10):
        _ = model(inputs.input_ids)

    # Actual timing: measure single-sample inference
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(n_samples):
        _ = model(inputs.input_ids[:1])  # Process one sample at a time
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end = time.time()

    # Convert to milliseconds per sample
    latency_ms = (end - start) / n_samples * 1000
    return latency_ms
