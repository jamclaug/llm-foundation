#!/usr/bin/env python3
"""
Training script for Sparse MoE Transformer.

Uses the shared Trainer module for:
- Mixed precision training (fp16/bf16) - ~2x speedup
- torch.compile support - ~30% speedup  
- Gradient checkpointing - enables longer sequences
- Early stopping, W&B logging, auto-checkpointing

Usage:
    # Basic training on TinyStories
    python train.py --max_steps 5000

    # Long-sequence training on WikiText
    python train.py --dataset wikitext --max_len 1024 --batch_size 2 --grad_acc_steps 16
    
    # With all optimizations
    python train.py --dataset wikitext --mixed_precision fp16 --compile
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from trainer import create_trainer
from dataset import get_dataset
from config import Config
from model import SparseMoETransformer


def main():
    parser = argparse.ArgumentParser(description="Train Sparse MoE Transformer")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512,
                        help="Embedding dimension")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="FFN hidden dimension")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--n_experts", type=int, default=16,
                        help="Number of expert networks")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Active experts per token")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "wikitext", "pg19"],
                        help="Dataset to train on")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Max sequence length (default: auto based on dataset)")
    
    # Training
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_acc_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=500)
    
    # Performance optimizations
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "none"],
                        help="Mixed precision mode (fp16 gives ~2x speedup)")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (PyTorch 2.0+, ~30%% speedup)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Trade compute for memory (enables longer sequences)")
    
    # Logging & output
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging")
    parser.add_argument("--early_stopping", action="store_true",
                        help="Stop if validation loss stops improving")
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Determine max_len based on dataset
    if args.max_len is None:
        max_len_defaults = {"tinystories": 256, "wikitext": 1024, "pg19": 2048}
        args.max_len = max_len_defaults[args.dataset]
    
    print(f"\n{'='*70}")
    print("SPARSE MOE TRANSFORMER TRAINING")
    print(f"{'='*70}")
    print(f"Model: {args.n_layers} layers, {args.d_model} dim, {args.n_experts} experts (top-{args.top_k})")
    print(f"Dataset: {args.dataset} (max_len={args.max_len})")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"{'='*70}\n")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Config
    config = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        top_k=args.top_k,
        max_len=args.max_len,
    )
    
    # Model
    print("Loading model...")
    model = SparseMoETransformer(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_active = n_params - (config.n_experts - config.top_k) * config.d_model * config.d_ff * 2 * config.n_layers // config.n_experts
    print(f"Model parameters: {n_params/1e6:.1f}M total, ~{n_active/1e6:.1f}M active per token")
    
    # Datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = get_dataset(args.dataset, tokenizer, split="train", max_len=args.max_len)
    val_dataset = get_dataset(args.dataset, tokenizer, split="validation", max_len=args.max_len)
    
    # Output directory
    if args.output_dir is None:
        ds = f"_{args.dataset}" if args.dataset != "tinystories" else ""
        args.output_dir = f"output/sparse_moe_transformer{ds}_{args.max_steps}steps"
    
    # Create trainer using shared module
    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_acc_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile,
        gradient_checkpointing=args.gradient_checkpointing,
        early_stopping=args.early_stopping,
        log_wandb=args.wandb,
    )
    
    # Train!
    best_val_loss = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
