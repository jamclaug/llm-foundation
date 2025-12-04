#!/usr/bin/env python3
"""
Text generation script for BDH Transformer.

Implements greedy decoding and sampling strategies for story generation.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import Config
from model_bdh import BDHTransformer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: str = "cuda"
):
    """
    Generate text continuation from a prompt.
    
    Args:
        model: Trained BDHTransformer
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top-k tokens (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage)
        device: Device to run on
        
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass - get logits for next token
            # Use only the last max_len tokens if sequence gets too long
            seq_len = generated.shape[1]
            if seq_len > model.config.max_len:
                input_slice = generated[:, -model.config.max_len:]
            else:
                input_slice = generated
            
            # Get logits
            output = model(input_slice)
            logits = output["logits"][:, -1, :]  # Last token predictions
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least 1 token
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode and return
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text with BDH Transformer")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Starting prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Nucleus sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                       help="Penalty for repeating tokens")
    parser.add_argument("--num_stories", type=int, default=1,
                       help="Number of stories to generate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Extract config and model state
    if 'model_state_dict' in checkpoint:
        # Full checkpoint (from training)
        model_state = checkpoint['model_state_dict']
        # Load config from checkpoint if available
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            # Filter out training-specific parameters
            model_config_keys = ['vocab_size', 'max_len', 'n_layers', 'n_heads', 
                               'embed_dim', 'ff_dim', 'dropout', 'learning_mode']
            config_dict_filtered = {k: v for k, v in config_dict.items() if k in model_config_keys}
            config = Config(**config_dict_filtered)
        else:
            config = Config()
    else:
        # Direct model weights
        model_state = checkpoint
        config = Config()
    
    model = BDHTransformer(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    
    # Generate stories
    print("\n" + "="*70)
    print(f"GENERATING {args.num_stories} STORY/STORIES")
    print("="*70)
    print(f"Prompt: '{args.prompt}'")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print("="*70 + "\n")
    
    for i in range(args.num_stories):
        if args.num_stories > 1:
            print(f"\n--- Story {i+1} ---\n")
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        
        print(generated_text)
        print()


if __name__ == "__main__":
    main()
