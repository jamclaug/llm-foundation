#!/usr/bin/env python3
"""
Training script for BDH-SLIM Transformer with Hebbian learning.

Supports three training modes:
1. backprop: Standard gradient descent (like GPT)
2. hebbian: Pure local Hebbian learning
3. hybrid: Both mechanisms simultaneously
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

# Import BDH model and config
from config import Config
from model_bdh import BDHTransformer
from dataset import TinyStoriesDataset


def train_bdh(config: Config, resume_from: Optional[str] = None):
    """
    Training loop for BDH model with Hebbian learning support.
    
    Implements custom training logic for BDH with:
    - Hebbian weight updates after each batch
    - Activity-dependent pruning every N steps  
    - Homeostatic scaling for stable activity
    """
    print(f"\n{'='*70}")
    print(f"BDH-SLIM TRANSFORMER TRAINING")
    print(f"{'='*70}")
    print(f"Learning mode: {config.learning_mode}")
    print(f"Hebbian LR: {config.hebbian_lr}")
    print(f"Biological pruning: Every {config.prune_every} steps")
    print(f"{'='*70}\n")
    
    # Setup
    torch.manual_seed(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    model = BDHTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
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
    
    # Datasets
    train_dataset = TinyStoriesDataset(tokenizer, max_len=config.max_len, split="train")
    val_dataset = TinyStoriesDataset(tokenizer, max_len=config.max_len, split="validation")
    
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
    
    # Optimizer (only for backprop/hybrid modes)
    optimizer = None
    if config.learning_mode in ['backprop', 'hybrid']:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
    
    # Learning rate scheduler
    scheduler = None
    if optimizer:
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
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
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
    print(f"Effective batch size: {config.batch_size * config.grad_acc_steps}")
    print(f"Validation every {config.eval_every} steps\n")
    
    start_time = time.time()
    
    while step < config.max_steps:
        epoch_loss = 0
        batches_in_epoch = 0
        
        for micro_step in range(config.grad_acc_steps):
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
            loss = output["loss"] / config.grad_acc_steps
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/Inf loss at step {step}, skipping batch")
                if optimizer:
                    optimizer.zero_grad()
                continue
            
            # Backprop update (if enabled)
            if config.learning_mode in ['backprop', 'hybrid']:
                loss.backward()
            
            epoch_loss += loss.item() * config.grad_acc_steps
            batches_in_epoch += 1
        
        # Optimizer step (if using backprop)
        if optimizer:
            grad_clip = getattr(config, 'grad_clip', 1.0)  # Default to 1.0 if not set
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        # Hebbian update (if enabled)
        if config.learning_mode in ['hebbian', 'hybrid']:
            model.hebbian_update()
        
        # Biological plasticity (periodic)
        if step % config.prune_every == 0 and step > 0:
            model.apply_biological_plasticity()
            stats = model.get_sparsity_stats()
            print(f"  [Plasticity] Sparsity: {stats['weight_sparsity']:.2%}, Active: {stats['active_params']:,}")
        
        step += 1
        
        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta_seconds = (config.max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            
            print(f"Step {step}/{config.max_steps} | Loss: {epoch_loss/batches_in_epoch:.4f} | "
                  f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h")
        
        # Validation
        if step % config.eval_every == 0:
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
                    
                    if val_batches >= 100:  # Limit validation batches
                        break
            
            val_loss /= val_batches
            print(f"\n{'='*70}")
            print(f"VALIDATION @ Step {step}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"{'='*70}\n")
            
            # Save checkpoint
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config.__dict__
            }
            
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_step_{step}.pt")
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save checkpoint: {e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.output_dir, "best_checkpoint.pt")
                try:
                    torch.save(checkpoint, best_path)
                    print(f"✓ New best model! Val loss: {val_loss:.4f}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to save best checkpoint: {e}")
            
            model.train()
        
        # Save periodic checkpoint
        if step % 1000 == 0:
            checkpoint_path = os.path.join(config.output_dir, f"checkpoint_step_{step}.pt")
            try:
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__
                }, checkpoint_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to save periodic checkpoint: {e}")
    
    # Final validation
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - Final Validation")
    print(f"{'='*70}")
    
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
            
            if val_batches >= 200:
                break
    
    val_loss /= val_batches
    print(f"Final Val Loss: {val_loss:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    # Get sparsity statistics
    stats = model.get_sparsity_stats()
    print(f"\nFinal Sparsity Statistics:")
    print(f"  Weight sparsity: {stats['weight_sparsity']:.2%}")
    print(f"  Active parameters: {stats['active_params']:,} / {stats['total_params']:,}")
    print(f"  Zero weights: {stats['zero_params']:,}")
    
    print(f"\nTraining completed in {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best checkpoint: {os.path.join(config.output_dir, 'best_checkpoint.pt')}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    
    # Mode
    parser.add_argument("--mode", choices=["train", "benchmark"], default="train")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output/bdh_transformer")
    parser.add_argument("--log_wandb", action="store_true")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, help="Embedding dimension (default: 512)")
    parser.add_argument("--n_heads", type=int, help="Number of attention heads (default: 8)")
    parser.add_argument("--d_ff", type=int, help="FFN hidden dimension (default: 1024)")
    parser.add_argument("--n_layers", type=int, help="Number of transformer layers (default: 6)")
    
    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, help="Total training steps (default: 5000)")
    parser.add_argument("--batch_size", type=int, help="Physical batch size (default: 4)")
    parser.add_argument("--lr", type=float, help="Learning rate (default: 3e-4)")
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--eval_every", type=int, help="Evaluate every N steps (default: 500)")
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume training from")
    
    # BDH-specific parameters
    parser.add_argument("--learning_mode", choices=["backprop", "hebbian", "hybrid"], 
                       default="hybrid", help="Learning paradigm")
    parser.add_argument("--hebbian_lr", type=float, default=0.01, 
                       help="Hebbian learning rate")
    parser.add_argument("--prune_every", type=int, default=1000,
                       help="Apply biological pruning every N steps")
    parser.add_argument("--prune_threshold", type=float, default=0.01,
                       help="Threshold for weak connection pruning")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override with command-line args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.log_wandb:
        config.log_wandb = args.log_wandb
    
    # Model architecture overrides
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_heads is not None:
        config.n_heads = args.n_heads
        # Validate
        if config.d_model % config.n_heads != 0:
            raise ValueError(f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})")
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    
    # Training hyperparameter overrides
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.grad_acc_steps is not None:
        config.grad_acc_steps = args.grad_acc_steps
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    
    # BDH-specific overrides
    config.learning_mode = args.learning_mode
    config.hebbian_lr = args.hebbian_lr
    config.prune_every = args.prune_every
    config.prune_threshold = args.prune_threshold
    
    # Set model type flag
    config.model_type = 'bdh'  # Signal to train.py to use BDH model
    
    if args.mode == "train":
        train_bdh(config, resume_from=args.resume_from)
    else:
        # Benchmark mode not yet implemented for BDH
        print("Benchmark mode not yet implemented for BDH")


if __name__ == "__main__":
    main()
