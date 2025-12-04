#!/usr/bin/env python3
"""
Text generation script for Sparse MoE Transformer.

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
from model import SparseMoETransformer


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
        model: Trained SparseMoETransformer
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
                input_chunk = generated[:, -model.config.max_len:]
            else:
                input_chunk = generated
            
            outputs = model(input_chunk)
            logits = outputs["logits"]
            
            # Get logits for last position (next token prediction)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least one token
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode and return
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate stories with Sparse MoE Transformer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting prompt")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling (0-1, 1=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Penalty for repetition (1.0=none, 1.2=moderate)")
    parser.add_argument("--num_stories", type=int, default=3, help="Number of stories to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle both checkpoint files and direct model weights
    if 'model_state_dict' in checkpoint:
        # Full checkpoint (from training)
        model_state = checkpoint['model_state_dict']
        # Load config from checkpoint if available
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = Config(**config_dict)
        else:
            config = Config()
    else:
        # Direct model weights
        model_state = checkpoint
        config = Config()
    
    model = SparseMoETransformer(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    
    # Generate stories
    print("\n" + "="*70)
    print(f"Generating {args.num_stories} stories with prompt: '{args.prompt}'")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("="*70 + "\n")
    
    for i in range(args.num_stories):
        print(f"Story {i+1}:")
        print("-" * 70)
        
        story = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=str(device)
        )
        
        print(story)
        print("\n")
    
    print("="*70)
    print("✅ Generation complete!")


if __name__ == "__main__":
    main()
