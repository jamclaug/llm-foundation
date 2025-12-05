#!/usr/bin/env python3
"""
Training script for Standard Transformer language model.

Uses the shared Trainer module for:
- Mixed precision training (fp16/bf16) - ~2x speedup
- torch.compile support - ~30% speedup  
- Gradient checkpointing - enables longer sequences
- Early stopping, W&B logging, auto-checkpointing

A baseline for comparing with:
- Mamba (pure SSM)
- Hymba (hybrid Attention+SSM)
- Sparse MoE Transformer

Usage:
    # Basic training on TinyStories
    python train.py --preset 30m --max_steps 5000

    # Long-sequence training on PG-19 (tests O(L²) attention scaling)
    python train.py --dataset pg19 --max_steps 5000
    
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
from model import TransformerLM, get_transformer_config


def main():
    parser = argparse.ArgumentParser(description="Train standard Transformer")
    
    # Model
    parser.add_argument("--preset", type=str, default="30m",
                        choices=["30m", "125m"],
                        help="Model size preset")
    
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
    print("STANDARD TRANSFORMER TRAINING")
    print(f"{'='*70}")
    print(f"Model: Transformer ({args.preset})")
    print(f"Dataset: {args.dataset} (max_len={args.max_len})")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"{'='*70}\n")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Config
    config = get_transformer_config(args.preset)
    config.max_len = args.max_len
    config.vocab_size = tokenizer.vocab_size
    
    # Model
    print("Loading model...")
    model = TransformerLM(config)
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = get_dataset(args.dataset, tokenizer, split="train", max_len=args.max_len)
    val_dataset = get_dataset(args.dataset, tokenizer, split="validation", max_len=args.max_len)
    
    # Output directory
    if args.output_dir is None:
        ds = f"_{args.dataset}" if args.dataset != "tinystories" else ""
        args.output_dir = f"output/transformer_{args.preset}{ds}_{args.max_steps}steps"
    
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
