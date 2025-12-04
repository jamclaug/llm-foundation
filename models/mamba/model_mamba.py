"""
Mamba model implementations for language modeling.

This module provides:
1. Pure PyTorch Mamba implementation (works on Windows, no special deps)
2. Hugging Face pretrained model loader (state-spaces/mamba-130m, trained on The Pile)

The pure PyTorch version is slower than optimized CUDA but fully readable
and works anywhere. Use pretrained models for evaluation/comparison.

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           https://arxiv.org/abs/2312.00752
"""
import math
import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from config import MambaConfig


# =============================================================================
# PRETRAINED MODEL LOADING (Hugging Face)
# =============================================================================

def load_pretrained_mamba(
    model_name: str = "state-spaces/mamba-130m",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32
):
    """
    Load a pretrained Mamba model from Hugging Face.
    
    Available models:
        - state-spaces/mamba-130m  (130M params, trained on The Pile)
        - state-spaces/mamba-370m  (370M params, trained on The Pile)
        - state-spaces/mamba-790m  (790M params, trained on The Pile)
        - state-spaces/mamba-1.4b  (1.4B params, trained on The Pile)
        - state-spaces/mamba-2.8b  (2.8B params, trained on The Pile)
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
        dtype: Model dtype (float32 or float16)
    
    Returns:
        model: The loaded model
        tokenizer: The tokenizer
    """
    try:
        from transformers import AutoTokenizer, AutoConfig
        from transformers import MambaForCausalLM
    except ImportError:
        raise ImportError("transformers not installed. Run: pip install transformers>=4.40.0")
    
    print(f"Loading pretrained model: {model_name}")
    print("(First time downloads ~500MB from HuggingFace)")
    
    # Load tokenizer - Mamba uses GPT-NeoX tokenizer
    print("Loading tokenizer (EleutherAI/gpt-neox-20b)...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load config and fix known mismatches for state-spaces models
    print(f"Loading config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # The state-spaces/mamba-130m checkpoint has:
    # - 24 layers (not 32 as config sometimes says)
    # - vocab_size 50280 (not 50277)
    # Check and fix based on model name
    if "130m" in model_name.lower():
        config.num_hidden_layers = 24
        config.vocab_size = 50280
    elif "370m" in model_name.lower():
        config.num_hidden_layers = 48
        config.vocab_size = 50280
    
    print(f"Config: {config.num_hidden_layers} layers, vocab {config.vocab_size}")
    
    # Load model with corrected config
    print(f"Loading model weights...")
    
    # Try loading - transformer's Mamba implementation can create meta tensors
    # so we use a workaround: load state dict directly
    try:
        from huggingface_hub import hf_hub_download
        
        # Download the pytorch model file
        print("Downloading model weights...")
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="pytorch_model.bin"
        )
        
        # Create model on target device
        print(f"Creating model on {device}...")
        with torch.device(device):
            model = MambaForCausalLM(config)
        
        # Load state dict
        print("Loading state dict...")
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        
    except Exception as e:
        print(f"Direct loading failed ({e}), trying standard from_pretrained...")
        # Fallback to standard loading
        model = MambaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
    
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Loaded {model_name}")
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    return model, tokenizer


class PretrainedMambaWrapper:
    """
    Wrapper for pretrained HuggingFace Mamba models.
    
    Provides a unified interface matching our custom MambaLM.
    """
    
    def __init__(self, model_name: str = "state-spaces/mamba-130m", device: str = "cuda"):
        self.model, self.tokenizer = load_pretrained_mamba(model_name, device)
        self.device = device
        self.model_name = model_name
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_loss(self, text: str) -> float:
        """Calculate loss (perplexity proxy) for given text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        return outputs.loss.item()


# =============================================================================
# PURE PYTORCH IMPLEMENTATION (No mamba-ssm required)
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
    
    In discrete form:
        h[k] = Ā h[k-1] + B̄ x[k]
        y[k] = C h[k] + D x[k]
    
    Where Ā, B̄ are discretized using timestep Δ.
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
        Note: This sequential version is slower than parallel scan in mamba-ssm.
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
    """Single Mamba block: LayerNorm -> SelectiveSSM -> Residual"""
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.ssm = SelectiveSSM(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
    
    def forward(self, x):
        return x + self.ssm(self.norm(x))


class MambaLM(nn.Module):
    """
    Pure PyTorch Mamba Language Model.
    
    Use this for training from scratch or understanding the architecture.
    For pretrained models, use load_pretrained_mamba() instead.
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying
        
        self.apply(self._init_weights)
        
        print(f"MambaLM (Pure PyTorch) initialized")
        print(f"  Parameters: {self.num_parameters():,} ({self.num_parameters()/1e6:.1f}M)")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}")
        print(f"  d_state={config.d_state}, d_conv={config.d_conv}, expand={config.expand}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
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
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# Mamba2 - same implementation for educational purposes
# Real Mamba2 has SSD (Structured State Space Duality) optimizations
class Mamba2LM(MambaLM):
    """Mamba2 LM - uses same pure PyTorch implementation."""
    pass


# =============================================================================
# MAIN: Test both implementations
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["custom", "pretrained", "both"], default="both")
    parser.add_argument("--model", default="state-spaces/mamba-130m")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MAMBA MODEL TESTS")
    print("=" * 70)
    
    # Test custom pure PyTorch implementation
    if args.mode in ["custom", "both"]:
        print("\n--- Testing Pure PyTorch MambaLM ---\n")
        from config import MambaConfig
        
        # Small config for quick testing
        config = MambaConfig(
            d_model=256,
            n_layers=4,
            d_state=16,
            d_conv=4,
            expand=2,
            vocab_size=1000,
            max_len=128,
        )
        
        model = MambaLM(config)
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        print(f"Input shape: {input_ids.shape}")
        outputs = model(input_ids, labels=input_ids)
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Loss: {outputs['loss'].item():.4f}")
        
        print("\nTesting generation...")
        generated = model.generate(input_ids[:1, :10], max_new_tokens=20, temperature=0.8, top_k=50)
        print(f"Generated shape: {generated.shape}")
        print("✓ Pure PyTorch implementation works!\n")
    
    # Test pretrained model from HuggingFace
    if args.mode in ["pretrained", "both"]:
        print("\n--- Testing Pretrained Mamba from HuggingFace ---\n")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        try:
            wrapper = PretrainedMambaWrapper(args.model, device=device)
            
            prompts = [
                "Once upon a time",
                "The quick brown fox",
                "In a galaxy far away",
            ]
            
            print("\nGenerating text samples:\n")
            for prompt in prompts:
                print(f"Prompt: '{prompt}'")
                output = wrapper.generate_text(prompt, max_new_tokens=50, temperature=0.8)
                print(f"Output: {output}\n")
                print("-" * 50)
            
            # Test perplexity calculation
            test_text = "The cat sat on the mat."
            loss = wrapper.get_loss(test_text)
            print(f"\nTest loss on '{test_text}': {loss:.4f}")
            print(f"Perplexity: {math.exp(loss):.2f}")
            
            print("\n✓ Pretrained model works!")
            
        except Exception as e:
            print(f"❌ Error loading pretrained model: {e}")
            print("Make sure transformers is installed: pip install transformers")
    
    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
