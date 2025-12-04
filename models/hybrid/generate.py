#!/usr/bin/env python3
"""
Text generation script for Hymba hybrid models.

Usage:
    python generate.py --checkpoint path/to/checkpoint.pt --prompt "Once upon a time"
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from model_hymba import HymbaLM, JambaLM, HymbaConfig


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    config_dict = checkpoint['config']
    config = HymbaConfig(**config_dict)
    
    # Determine model type
    model_type = checkpoint.get('model_type', 'hymba')
    print(f"Model type: {model_type}")
    
    # Create model
    if model_type == "hymba":
        model = HymbaLM(config)
    else:
        model = JambaLM(config)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def generate(
    checkpoint_path: str,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = None,
):
    """Generate text from a prompt."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model, config = load_model(checkpoint_path, device)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Temperature: {temperature}, Top-k: {top_k}")
    print(f"{'='*60}\n")
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    # Decode
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)
    
    # Print mixing stats for Hymba
    if hasattr(model, 'get_mixing_stats'):
        print(f"\n{'='*60}")
        print("Attention vs SSM mixing (learned weights):")
        for layer_name, stats in model.get_mixing_stats().items():
            print(f"  {layer_name}: attn_ratio={stats['attn_ratio_mean']:.3f} "
                  f"(range: {stats['attn_ratio_min']:.2f}-{stats['attn_ratio_max']:.2f})")
    
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text with Hymba model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    generate(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()
