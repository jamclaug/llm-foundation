#!/usr/bin/env python3
"""
Training script for Mamba/Mamba2 language models.

Uses the shared Trainer module for:
- Mixed precision training (fp16/bf16) - ~2x speedup
- torch.compile support - ~30% speedup
- Gradient checkpointing - enables longer sequences
- Early stopping, W&B logging, auto-checkpointing

Usage:
    # Basic training on TinyStories
    python train.py --use_fast --max_steps 5000

    # Long-sequence training on PG-19
    python train.py --use_fast --dataset pg19 --max_steps 5000
    
    # With all optimizations
    python train.py --use_fast --dataset pg19 --mixed_precision fp16 --compile
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
from config import MambaConfig, Mamba130MConfig, Mamba30MConfig, Mamba2_130MConfig


def main():
    parser = argparse.ArgumentParser(description="Train Mamba language model")
    
    # Model
    parser.add_argument("--model_type", type=str, default="mamba",
                        choices=["mamba", "mamba2"],
                        help="Mamba variant")
    parser.add_argument("--preset", type=str, default="30m",
                        choices=["30m", "130m"],
                        help="Model size preset")
    parser.add_argument("--use_fast", action="store_true",
                        help="Use mamba-ssm CUDA kernels (10-50x faster)")
    
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
    print("MAMBA LANGUAGE MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Model: {args.model_type} ({args.preset})")
    print(f"Implementation: {'mamba-ssm (CUDA)' if args.use_fast else 'Pure PyTorch'}")
    print(f"Dataset: {args.dataset} (max_len={args.max_len})")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"{'='*70}\n")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Config
    if args.preset == "30m":
        config = Mamba30MConfig()
    elif args.preset == "130m":
        if args.model_type == "mamba":
            config = Mamba130MConfig()
        else:
            config = Mamba2_130MConfig()
    
    config.model_type = args.model_type
    config.max_len = args.max_len
    config.vocab_size = tokenizer.vocab_size
    
    # Model
    print("Loading model...")
    if args.use_fast:
        try:
            from model_mamba_fast import FastMambaLM, MAMBA_SSM_AVAILABLE
            if not MAMBA_SSM_AVAILABLE:
                raise ImportError("mamba-ssm not available")
            model = FastMambaLM(config)
            print("✓ Using mamba-ssm CUDA kernels")
        except ImportError as e:
            print(f"⚠ mamba-ssm not available: {e}")
            print("Falling back to pure PyTorch...")
            from model_mamba import MambaLM
            model = MambaLM(config)
    else:
        from model_mamba import MambaLM
        model = MambaLM(config)
    
    print(f"Model parameters: {model.num_parameters():,}")
    
    # Datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = get_dataset(args.dataset, tokenizer, split="train", max_len=args.max_len)
    val_dataset = get_dataset(args.dataset, tokenizer, split="validation", max_len=args.max_len)
    
    # Output directory
    if args.output_dir is None:
        impl = "fast" if args.use_fast else "pytorch"
        ds = f"_{args.dataset}" if args.dataset != "tinystories" else ""
        args.output_dir = f"output/{args.model_type}_{args.preset}_{impl}{ds}_{args.max_steps}steps"
    
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
