#!/usr/bin/env python3
"""
Text generation script for Mamba models.

Supports:
1. Pretrained models from HuggingFace (state-spaces/mamba-130m, etc.)
2. Custom trained checkpoints

Usage:
    # Pretrained model (recommended for quick testing)
    python generate.py --pretrained state-spaces/mamba-130m
    python generate.py --pretrained state-spaces/mamba-130m --interactive
    
    # Custom trained checkpoint
    python generate.py --checkpoint output/mamba_130m_5000steps/best_checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import MambaConfig
from model_mamba import MambaLM, PretrainedMambaWrapper, load_pretrained_mamba


def load_custom_model(checkpoint_path: str, device: torch.device):
    """Load custom trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config from checkpoint
    saved_config = checkpoint.get('config', {})
    config = MambaConfig(
        model_type=saved_config.get('model_type', 'mamba'),
        d_model=saved_config.get('d_model', 768),
        n_layers=saved_config.get('n_layers', 24),
        d_state=saved_config.get('d_state', 16),
        d_conv=saved_config.get('d_conv', 4),
        expand=saved_config.get('expand', 2),
        vocab_size=saved_config.get('vocab_size', 50257),
        max_len=saved_config.get('max_len', 256),
    )
    
    model = MambaLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {config.model_type}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Trained for: {checkpoint.get('step', 'unknown')} steps")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")
    
    return model, config


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None
):
    """Generate text from prompt (for custom MambaLM models)."""
    if device is None:
        device = next(model.parameters()).device
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def interactive_mode_pretrained(wrapper, args):
    """Interactive generation mode for pretrained models."""
    print("\n" + "="*70)
    print("INTERACTIVE GENERATION MODE (Pretrained Mamba)")
    print("Type your prompt and press Enter to generate")
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue
            
            print("\nGenerating...\n")
            output = wrapper.generate_text(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(f"Output:\n{output}\n")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def interactive_mode_custom(model, tokenizer, device, args):
    """Interactive generation mode for custom trained models."""
    print("\n" + "="*70)
    print("INTERACTIVE GENERATION MODE (Custom Trained)")
    print("Type your prompt and press Enter to generate")
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            if not prompt:
                continue
            
            print("\nGenerating...\n")
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            print(f"Output:\n{output}\n")
            print("-" * 70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mamba model")
    
    # Model source (one required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pretrained", type=str,
                       help="HuggingFace model name (e.g., state-spaces/mamba-130m)")
    group.add_argument("--checkpoint", type=str,
                       help="Path to custom trained checkpoint")
    
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation")
    parser.add_argument("--interactive", action="store_true",
                       help="Interactive generation mode")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pretrained model
    if args.pretrained:
        print(f"\nLoading pretrained model: {args.pretrained}")
        try:
            wrapper = PretrainedMambaWrapper(args.pretrained, device=device)
        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("\nMake sure transformers is installed: pip install transformers")
            return
        
        if args.interactive:
            interactive_mode_pretrained(wrapper, args)
        elif args.prompt:
            for i in range(args.num_samples):
                if args.num_samples > 1:
                    print(f"\n--- Sample {i+1}/{args.num_samples} ---")
                output = wrapper.generate_text(
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                print(output)
        else:
            # Default demo
            prompts = [
                "Once upon a time",
                "The quick brown fox",
                "In a galaxy far away",
                "The meaning of life is",
            ]
            print("\n" + "="*70)
            print("SAMPLE GENERATIONS")
            print("="*70)
            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                print("-" * 40)
                output = wrapper.generate_text(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(output)
        return
    
    # Load custom trained model
    if args.checkpoint:
        try:
            model, config = load_custom_model(args.checkpoint, torch.device(device))
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        if args.interactive:
            interactive_mode_custom(model, tokenizer, torch.device(device), args)
        elif args.prompt:
            for i in range(args.num_samples):
                if args.num_samples > 1:
                    print(f"\n--- Sample {i+1}/{args.num_samples} ---")
                output = generate(
                    model, tokenizer, args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=torch.device(device)
                )
                print(output)
        else:
            # Default prompts
            prompts = [
                "Once upon a time",
                "The little girl",
                "One day, a cat",
                "There was a boy named",
            ]
            print("\n" + "="*70)
            print("SAMPLE GENERATIONS")
            print("="*70)
            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                print("-" * 40)
                output = generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=torch.device(device)
                )
                print(output)


if __name__ == "__main__":
    main()
