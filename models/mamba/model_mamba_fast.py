"""
Optimized Mamba model using the official mamba-ssm library.

This module provides CUDA-optimized Mamba implementations that are 10-50x faster
than the pure PyTorch version. Requires Linux/WSL with CUDA.

Key optimizations from mamba-ssm:
1. Parallel selective scan using CUDA kernels (vs sequential Python loop)
2. Fused operations to minimize memory bandwidth
3. Efficient causal conv1d implementation

For understanding the SSM math, see model_mamba.py (pure PyTorch version).
For fast training, use this file.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           https://arxiv.org/abs/2312.00752
"""
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from config import MambaConfig

# Try to import mamba-ssm, fall back to pure PyTorch if unavailable
MAMBA_SSM_AVAILABLE = False
try:
    from mamba_ssm import Mamba, Mamba2
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    MAMBA_SSM_AVAILABLE = True
    print("✓ mamba-ssm library loaded - using CUDA-optimized kernels")
except ImportError:
    print("⚠ mamba-ssm not installed - falling back to pure PyTorch")
    print("  For 10-50x speedup, install: pip install causal-conv1d mamba-ssm")


def check_mamba_ssm():
    """Check if mamba-ssm is available and print diagnostic info."""
    if MAMBA_SSM_AVAILABLE:
        import mamba_ssm
        print(f"mamba-ssm version: {mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'unknown'}")
        return True
    return False


# =============================================================================
# FAST MAMBA BLOCK (Using mamba-ssm)
# =============================================================================

class FastMambaBlock(nn.Module):
    """
    Mamba block using the official mamba-ssm CUDA kernels.
    
    This wraps the optimized Mamba module which includes:
    - CUDA parallel selective scan
    - Fused in/out projections
    - Efficient causal convolution
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for FastMambaBlock. "
                "Install with: pip install causal-conv1d mamba-ssm"
            )
        
        self.norm = nn.LayerNorm(config.d_model)
        
        # Use the optimized Mamba module from mamba-ssm
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
    
    def forward(self, x):
        """Forward pass with pre-norm and residual connection."""
        return x + self.mamba(self.norm(x))


class FastMamba2Block(nn.Module):
    """
    Mamba2 block using the official mamba-ssm CUDA kernels.
    
    Mamba2 includes State Space Duality (SSD) optimization that allows
    even more efficient parallel computation.
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for FastMamba2Block. "
                "Install with: pip install causal-conv1d mamba-ssm"
            )
        
        self.norm = nn.LayerNorm(config.d_model)
        
        # Mamba2 requires headdim parameter
        # d_model must be divisible by headdim
        headdim = 64 if config.d_model >= 64 else config.d_model
        
        self.mamba = Mamba2(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=headdim,
        )
    
    def forward(self, x):
        """Forward pass with pre-norm and residual connection."""
        return x + self.mamba(self.norm(x))


# =============================================================================
# FAST MAMBA LANGUAGE MODEL
# =============================================================================

class FastMambaLM(nn.Module):
    """
    Optimized Mamba Language Model using mamba-ssm CUDA kernels.
    
    Architecture:
    - Token embedding
    - N x FastMambaBlock (or FastMamba2Block)
    - Layer norm
    - LM head (tied to embedding weights)
    
    This achieves 10-50x speedup over pure PyTorch by using:
    - Parallel selective scan instead of sequential
    - Fused CUDA kernels for all operations
    - Memory-efficient implementation
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        if not MAMBA_SSM_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for FastMambaLM. "
                "Install with: pip install causal-conv1d mamba-ssm\n"
                "For pure PyTorch version, use MambaLM from model_mamba.py"
            )
        
        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Mamba blocks
        if config.model_type == "mamba2":
            self.layers = nn.ModuleList([
                FastMamba2Block(config) for _ in range(config.n_layers)
            ])
            block_type = "Mamba2"
        else:
            self.layers = nn.ModuleList([
                FastMambaBlock(config) for _ in range(config.n_layers)
            ])
            block_type = "Mamba"
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        n_params = self.num_parameters()
        print(f"FastMambaLM ({block_type}, CUDA-optimized) initialized")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}")
        print(f"  d_state={config.d_state}, d_conv={config.d_conv}, expand={config.expand}")
    
    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len] token indices
            labels: [batch, seq_len] target tokens (optional, for loss)
        
        Returns:
            dict with 'logits' and optionally 'loss'
        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}
    
    def num_parameters(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: [batch, seq_len] initial tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k tokens
            top_p: if set, nucleus sampling threshold
        
        Returns:
            [batch, seq_len + max_new_tokens] generated tokens
        """
        for _ in range(max_new_tokens):
            # Get predictions for the last position
            outputs = self(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_mamba_model(config: MambaConfig, force_fast: bool = False):
    """
    Create a Mamba model, using fast implementation if available.
    
    Args:
        config: MambaConfig with model parameters
        force_fast: If True, raise error if mamba-ssm not available
    
    Returns:
        MambaLM (pure PyTorch) or FastMambaLM (CUDA-optimized)
    """
    if MAMBA_SSM_AVAILABLE:
        print("Using CUDA-optimized FastMambaLM")
        return FastMambaLM(config)
    elif force_fast:
        raise ImportError(
            "mamba-ssm is required but not installed. "
            "Install with: pip install causal-conv1d mamba-ssm"
        )
    else:
        print("Falling back to pure PyTorch MambaLM")
        from model_mamba import MambaLM
        return MambaLM(config)


# =============================================================================
# BENCHMARK: Compare Pure PyTorch vs mamba-ssm
# =============================================================================

def benchmark_mamba_implementations(
    batch_size: int = 4,
    seq_len: int = 256,
    d_model: int = 512,
    n_layers: int = 8,
    num_iterations: int = 100,
):
    """
    Benchmark pure PyTorch vs mamba-ssm implementations.
    
    This demonstrates the speedup from using CUDA-optimized kernels.
    """
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print("MAMBA IMPLEMENTATION BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"d_model: {d_model}, n_layers: {n_layers}")
    print(f"Iterations: {num_iterations}")
    print(f"{'='*70}\n")
    
    # Config
    config = MambaConfig(
        d_model=d_model,
        n_layers=n_layers,
        d_state=16,
        d_conv=4,
        expand=2,
        vocab_size=50257,
    )
    
    # Test input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    
    # Benchmark pure PyTorch
    print("Testing Pure PyTorch implementation...")
    try:
        from model_mamba import MambaLM
        model_pytorch = MambaLM(config).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_pytorch(input_ids)
        
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model_pytorch(input_ids)
        
        torch.cuda.synchronize() if device == "cuda" else None
        pytorch_time = time.time() - start
        
        results["PyTorch"] = {
            "time": pytorch_time,
            "tokens_per_sec": (batch_size * seq_len * num_iterations) / pytorch_time,
        }
        print(f"  Time: {pytorch_time:.2f}s")
        print(f"  Speed: {results['PyTorch']['tokens_per_sec']:.0f} tokens/sec\n")
        
        del model_pytorch
        torch.cuda.empty_cache() if device == "cuda" else None
        
    except Exception as e:
        print(f"  Error: {e}\n")
    
    # Benchmark mamba-ssm (if available)
    if MAMBA_SSM_AVAILABLE:
        print("Testing mamba-ssm (CUDA-optimized) implementation...")
        try:
            model_fast = FastMambaLM(config).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model_fast(input_ids)
            
            torch.cuda.synchronize() if device == "cuda" else None
            start = time.time()
            
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model_fast(input_ids)
            
            torch.cuda.synchronize() if device == "cuda" else None
            fast_time = time.time() - start
            
            results["mamba-ssm"] = {
                "time": fast_time,
                "tokens_per_sec": (batch_size * seq_len * num_iterations) / fast_time,
            }
            print(f"  Time: {fast_time:.2f}s")
            print(f"  Speed: {results['mamba-ssm']['tokens_per_sec']:.0f} tokens/sec\n")
            
            del model_fast
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            print(f"  Error: {e}\n")
    else:
        print("mamba-ssm not available - skipping optimized benchmark\n")
    
    # Summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if "PyTorch" in results and "mamba-ssm" in results:
        speedup = results["mamba-ssm"]["tokens_per_sec"] / results["PyTorch"]["tokens_per_sec"]
        print(f"Pure PyTorch: {results['PyTorch']['tokens_per_sec']:.0f} tokens/sec")
        print(f"mamba-ssm:    {results['mamba-ssm']['tokens_per_sec']:.0f} tokens/sec")
        print(f"Speedup:      {speedup:.1f}x")
    elif "PyTorch" in results:
        print(f"Pure PyTorch: {results['PyTorch']['tokens_per_sec']:.0f} tokens/sec")
        print("mamba-ssm: Not available")
    
    print(f"{'='*70}\n")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "benchmark"], default="test")
    parser.add_argument("--model_type", choices=["mamba", "mamba2"], default="mamba")
    args = parser.parse_args()
    
    if args.mode == "benchmark":
        benchmark_mamba_implementations()
    else:
        # Quick test
        print("\n" + "="*70)
        print("FAST MAMBA MODEL TEST")
        print("="*70 + "\n")
        
        if not MAMBA_SSM_AVAILABLE:
            print("❌ mamba-ssm not installed")
            print("Install with: pip install causal-conv1d mamba-ssm")
            sys.exit(1)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        config = MambaConfig(
            model_type=args.model_type,
            d_model=256,
            n_layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            vocab_size=1000,
        )
        
        model = FastMambaLM(config).to(device)
        
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
        generated = model.generate(
            input_ids[:1, :10],
            max_new_tokens=20,
            temperature=0.8,
            top_k=50
        )
        print(f"Generated shape: {generated.shape}")
        
        print("\n✓ FastMambaLM test passed!")
        print("="*70 + "\n")
