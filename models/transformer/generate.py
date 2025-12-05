#!/usr/bin/env python3
"""
Text generation script for Standard Transformer model.

Usage:
    python generate.py --checkpoint output/transformer_30m_5000steps/best_checkpoint.pt
    python generate.py --checkpoint output/transformer_30m_5000steps/best_checkpoint.pt --interactive
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import TransformerLM, TransformerConfig


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    saved_config = checkpoint.get('config', {})
    config = TransformerConfig(
        d_model=saved_config.get('d_model', 384),
        n_layers=saved_config.get('n_layers', 8),
        n_heads=saved_config.get('n_heads', 6),
        d_ff=saved_config.get('d_ff', 1536),
        vocab_size=saved_config.get('vocab_size', 50257),
        max_len=saved_config.get('max_len', 256),
    )
    
    model = TransformerLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded:")
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
    device: torch.device = None
):
    """Generate text from prompt."""
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


def interactive_mode(model, tokenizer, device, args):
    """Interactive generation mode."""
    print("\n" + "="*70)
    print("INTERACTIVE GENERATION MODE")
    print("Type your prompt and press Enter to generate")
    print("Type 'quit' or 'exit' to stop")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            
            print(f"\nGenerated:\n{output}\n")
            print("-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text with Transformer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    try:
        model, config = load_model(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    if args.interactive:
        interactive_mode(model, tokenizer, device, args)
    elif args.prompt:
        output = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print(f"\n{output}")
    else:
        # Default sample generations
        print("\n" + "="*70)
        print("SAMPLE GENERATIONS")
        print("="*70)
        
        prompts = [
            "Once upon a time",
            "The little girl",
            "One day, a cat",
            "There was a boy named"
        ]
        
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            print("-"*40)
            output = generate(
                model, tokenizer, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device
            )
            print(output)


if __name__ == "__main__":
    main()
