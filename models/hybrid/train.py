#!/usr/bin/env python3
"""
Training script for Hymba hybrid Transformer-Mamba models.

Uses the shared Trainer module for:
- Mixed precision training (fp16/bf16) - ~2x speedup
- torch.compile support - ~30% speedup
- Gradient checkpointing - enables longer sequences
- Early stopping, W&B logging, auto-checkpointing

Supports:
- HymbaLM: Parallel attention + SSM heads with learned mixing
- JambaLM: Interleaved attention and SSM layers

Usage:
    # Basic training on TinyStories
    python train.py --model hymba --max_steps 5000

    # Long-sequence training on PG-19
    python train.py --model hymba --dataset pg19 --max_steps 5000
    
    # With all optimizations
    python train.py --model hymba --dataset wikitext --mixed_precision fp16 --compile
    
    # Disable mamba-ssm (use pure PyTorch SSM)
    python train.py --model hymba --no_fast --max_steps 5000
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
from model_hymba import HymbaLM, JambaLM, get_hymba_config


def main():
    parser = argparse.ArgumentParser(description="Train Hymba hybrid model")
    
    # Model
    parser.add_argument("--model", type=str, default="hymba",
                        choices=["hymba", "jamba"],
                        help="Model type: hymba (parallel heads) or jamba (interleaved)")
    parser.add_argument("--preset", type=str, default="30m",
                        choices=["30m", "125m", "30m-fast"],
                        help="Model size preset")
    parser.add_argument("--no_fast", action="store_true",
                        help="Disable mamba-ssm CUDA kernels (use pure PyTorch SSM)")
    
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
    
    use_fast = not args.no_fast
    
    print(f"\n{'='*70}")
    print("HYMBA HYBRID MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Model: {args.model} ({args.preset})")
    print(f"SSM: {'mamba-ssm (CUDA)' if use_fast else 'Pure PyTorch'}")
    print(f"Dataset: {args.dataset} (max_len={args.max_len})")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"{'='*70}\n")
    
    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Config
    config = get_hymba_config(args.preset)
    config.max_len = args.max_len
    config.vocab_size = tokenizer.vocab_size
    config.use_fast_ssm = use_fast
    
    # Model
    print("Loading model...")
    if args.model == "hymba":
        model = HymbaLM(config)
    else:
        model = JambaLM(config)
    
    print(f"Model parameters: {model.num_params:,}")
    
    # Datasets
    print(f"\nLoading {args.dataset} dataset...")
    train_dataset = get_dataset(args.dataset, tokenizer, split="train", max_len=args.max_len)
    val_dataset = get_dataset(args.dataset, tokenizer, split="validation", max_len=args.max_len)
    
    # Output directory
    if args.output_dir is None:
        impl = "fast" if use_fast else "pytorch"
        ds = f"_{args.dataset}" if args.dataset != "tinystories" else ""
        args.output_dir = f"output/hybrid/{args.model}_{args.preset}_{impl}{ds}_{args.max_steps}steps"
    
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
