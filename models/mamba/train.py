#!/usr/bin/env python3
"""
Training script for Mamba/Mamba2 language models.

Trains Mamba (state space models) on TinyStories dataset for comparison
with transformer-based architectures.

Supports two implementations:
1. Pure PyTorch (slower, educational, works everywhere)
2. mamba-ssm CUDA kernels (10-50x faster, requires Linux/WSL)

Usage:
    # Pure PyTorch (slow but educational)
    python train.py --model_type mamba --max_steps 5000

    # CUDA-optimized (fast training)
    python train.py --model_type mamba --use_fast --max_steps 5000
    
    # Mamba2 with CUDA kernels
    python train.py --model_type mamba2 --use_fast --max_steps 5000
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import MambaConfig, Mamba130MConfig, Mamba30MConfig, Mamba2_130MConfig
from dataset import TinyStoriesDataset


def train_mamba(config: MambaConfig, resume_from: Optional[str] = None, use_fast: bool = False):
    """
    Training loop for Mamba models.
    
    Implements standard training with:
    - Gradient accumulation for effective batch size
    - Learning rate warmup and cosine decay
    - Periodic validation and checkpointing
    
    Args:
        config: MambaConfig with model and training parameters
        resume_from: Path to checkpoint to resume from
        use_fast: If True, use mamba-ssm CUDA kernels (10-50x faster)
    """
    print(f"\n{'='*70}")
    print(f"MAMBA LANGUAGE MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Model type: {config.model_type}")
    print(f"Implementation: {'mamba-ssm (CUDA)' if use_fast else 'Pure PyTorch'}")
    print(f"Hidden dim: {config.d_model}")
    print(f"Layers: {config.n_layers}")
    print(f"State dim: {config.d_state}")
    print(f"{'='*70}\n")
    
    # Setup
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Tokenizer (using GPT-2 tokenizer for compatibility)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Model - choose implementation based on use_fast flag
    if use_fast:
        try:
            from model_mamba_fast import FastMambaLM, MAMBA_SSM_AVAILABLE
            if not MAMBA_SSM_AVAILABLE:
                raise ImportError("mamba-ssm not available")
            model = FastMambaLM(config).to(device)
        except ImportError as e:
            print(f"\n❌ Error: {e}")
            print("\nTo install mamba-ssm for fast training, run:")
            print("  pip install causal-conv1d>=1.4.0")
            print("  pip install mamba-ssm>=2.0.0")
            print("\nFalling back to pure PyTorch implementation...")
            from model_mamba import MambaLM
            model = MambaLM(config).to(device)
    else:
        try:
            from model_mamba import MambaLM
            model = MambaLM(config).to(device)
        except ImportError as e:
            print(f"\n❌ Error: {e}")
            print("\nNote: mamba-ssm requires Linux or WSL on Windows.")
            return
    
    n_params = model.num_parameters()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Check for NaN in initial weights
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"⚠️  Warning: NaN/Inf in {name}")
            has_nan = True
    if has_nan:
        print("ERROR: Model has NaN weights after initialization!")
        return
    
    # Datasets - use cached data from sparse_moe_transformer
    cache_dir = Path(__file__).parent.parent.parent / "sparse_moe_transformer" / "cached_tokenized"
    train_dataset = TinyStoriesDataset(
        tokenizer, 
        max_len=config.max_len, 
        split="train",
        cache_dir=str(cache_dir)
    )
    val_dataset = TinyStoriesDataset(
        tokenizer, 
        max_len=config.max_len, 
        split="validation",
        cache_dir=str(cache_dir)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )
    
    # Resume from checkpoint
    start_step = 0
    best_val_loss = float('inf')
    if resume_from and os.path.exists(resume_from):
        print(f"\nLoading checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from step {start_step}")
    
    # Training loop
    model.train()
    step = start_step
    epoch = 0
    train_iter = iter(train_loader)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"\nStarting training from step {start_step} to {config.max_steps}")
    print(f"Effective batch size: {config.batch_size * config.grad_accumulation_steps}")
    print(f"Validation every {config.eval_every} steps\n")
    
    start_time = time.time()
    
    while step < config.max_steps:
        epoch_loss = 0
        batches_in_epoch = 0
        
        for micro_step in range(config.grad_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                epoch += 1
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            output = model(input_ids, labels=labels)
            loss = output["loss"] / config.grad_accumulation_steps
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/Inf loss at step {step}, skipping batch")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            loss.backward()
            
            epoch_loss += loss.item() * config.grad_accumulation_steps
            batches_in_epoch += 1
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        step += 1
        
        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta_seconds = (config.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Step {step}/{config.max_steps} | Loss: {epoch_loss/batches_in_epoch:.4f} | "
                  f"LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h")
        
        # Validation
        if step % config.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            
            print(f"\n{'='*70}")
            print(f"VALIDATION @ Step {step}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")
            print(f"{'='*70}\n")
            
            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler, step, val_loss, best_val_loss, 
                config, os.path.join(config.output_dir, f"checkpoint_step_{step}.pt")
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, step, val_loss, best_val_loss,
                    config, os.path.join(config.output_dir, "best_checkpoint.pt")
                )
                print(f"✓ New best model! Val loss: {val_loss:.4f}")
            
            model.train()
        
        # Generate sample text periodically
        if step % config.generate_every == 0:
            generate_sample(model, tokenizer, device)
    
    # Final validation
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - Final Validation")
    print(f"{'='*70}")
    
    val_loss = evaluate(model, val_loader, device, max_batches=200)
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Final Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    print(f"\nTraining completed in {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best checkpoint: {os.path.join(config.output_dir, 'best_checkpoint.pt')}")
    print(f"{'='*70}\n")


def evaluate(model, val_loader, device, max_batches=100):
    """Run evaluation on validation set."""
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for val_batch in val_loader:
            input_ids = val_batch['input_ids'].to(device)
            labels = val_batch['labels'].to(device)
            output = model(input_ids, labels=labels)
            val_loss += output["loss"].item()
            val_batches += 1
            
            if val_batches >= max_batches:
                break
    
    return val_loss / val_batches


def save_checkpoint(model, optimizer, scheduler, step, val_loss, best_val_loss, config, path):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': {
            'model_type': config.model_type,
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'd_state': config.d_state,
            'd_conv': config.d_conv,
            'expand': config.expand,
            'vocab_size': config.vocab_size,
            'max_len': config.max_len,
        }
    }
    
    try:
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")


def generate_sample(model, tokenizer, device, prompt="Once upon a time"):
    """Generate sample text from the model."""
    model.eval()
    
    print(f"\n--- Generated Sample ---")
    print(f"Prompt: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50
        )
    
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Output: {text}")
    print(f"------------------------\n")
    
    model.train()


def main():
    parser = argparse.ArgumentParser(description="Train Mamba language model")
    
    # Model selection
    parser.add_argument("--model_type", choices=["mamba", "mamba2"], default="mamba",
                       help="Model type: mamba or mamba2")
    parser.add_argument("--preset", choices=["30m", "130m", "custom"], default="30m",
                       help="Model size preset (30m matches BDH/SparseMoE size)")
    parser.add_argument("--use_fast", action="store_true",
                       help="Use mamba-ssm CUDA kernels for 10-50x speedup (requires Linux/WSL)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for checkpoints")
    
    # Model architecture (for custom preset)
    parser.add_argument("--d_model", type=int, default=768,
                       help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=24,
                       help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16,
                       help="SSM state dimension")
    parser.add_argument("--d_conv", type=int, default=4,
                       help="Convolution kernel size")
    parser.add_argument("--expand", type=int, default=2,
                       help="Expansion factor")
    
    # Training
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--grad_acc_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--eval_every", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--generate_every", type=int, default=1000,
                       help="Generate sample every N steps")
    
    # Resume
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create config based on preset
    if args.preset == "30m":
        config = Mamba30MConfig()
    elif args.preset == "130m":
        if args.model_type == "mamba":
            config = Mamba130MConfig()
        else:
            config = Mamba2_130MConfig()
    else:
        config = MambaConfig(
            model_type=args.model_type,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
        )
    
    # Override with command-line args
    config.max_steps = args.max_steps
    config.batch_size = args.batch_size
    config.grad_accumulation_steps = args.grad_acc_steps
    config.learning_rate = args.lr
    config.warmup_steps = args.warmup_steps
    config.eval_every = args.eval_every
    config.generate_every = args.generate_every
    
    # Set output directory
    if args.output_dir:
        config.output_dir = args.output_dir
    else:
        impl_suffix = "_fast" if args.use_fast else "_pytorch"
        config.output_dir = f"output/{args.model_type}_{args.preset}{impl_suffix}_{args.max_steps}steps"
    
    # Print config
    print("\nConfiguration:")
    print(f"  Implementation: {'mamba-ssm (CUDA)' if args.use_fast else 'Pure PyTorch'}")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    print()
    
    # Train
    train_mamba(config, resume_from=args.resume_from, use_fast=args.use_fast)


if __name__ == "__main__":
    main()
