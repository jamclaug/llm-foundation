#!/usr/bin/env python3
"""
Training script for Hymba hybrid Transformer-Mamba models.

Supports:
- HymbaLM: Parallel attention + SSM heads with learned mixing
- JambaLM: Interleaved attention and SSM layers

Usage:
    python train.py --model hymba --max_steps 5000
    python train.py --model jamba --max_steps 5000
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

from dataset import TinyStoriesDataset
from model_hymba import HymbaLM, JambaLM, HymbaConfig, get_hymba_config


def train_hymba(
    model_type: str = "hymba",
    preset: str = "30m",
    max_steps: int = 5000,
    batch_size: int = 4,
    grad_acc_steps: int = 8,
    lr: float = 3e-4,
    eval_every: int = 50,
    output_dir: str = None,
    resume_from: Optional[str] = None,
):
    """
    Training loop for Hymba/Jamba hybrid models.
    """
    print(f"\n{'='*70}")
    print(f"HYMBA HYBRID MODEL TRAINING")
    print(f"{'='*70}")
    print(f"Model type: {model_type}")
    print(f"Preset: {preset}")
    print(f"Max steps: {max_steps}")
    print(f"Effective batch size: {batch_size * grad_acc_steps}")
    print(f"{'='*70}\n")
    
    # Setup
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Config
    config = get_hymba_config(preset)
    
    # Output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "output" / "hybrid" / f"{model_type}_{preset}_{max_steps//1000}k"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Auto-resume: check for latest checkpoint if no explicit resume given
    if resume_from is None:
        auto_resume_path = output_dir / "latest_checkpoint.pt"
        if auto_resume_path.exists():
            resume_from = str(auto_resume_path)
            print(f"Found existing checkpoint, will auto-resume from: {resume_from}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model
    if model_type == "hymba":
        model = HymbaLM(config).to(device)
    elif model_type == "jamba":
        model = JambaLM(config).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    n_params = model.num_params
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = TinyStoriesDataset(tokenizer, max_len=config.max_len, split="train")
    val_dataset = TinyStoriesDataset(tokenizer, max_len=config.max_len, split="validation")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    warmup_steps = max_steps // 10
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
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
    train_iter = iter(train_loader)
    
    print(f"\nStarting training from step {start_step} to {max_steps}")
    print(f"Validation every {eval_every} steps\n")
    
    start_time = time.time()
    accumulated_loss = 0
    
    while step < max_steps:
        optimizer.zero_grad()
        batch_loss = 0
        
        for micro_step in range(grad_acc_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            loss, logits = model(input_ids, labels=labels)
            loss = loss / grad_acc_steps
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️  Warning: NaN/Inf loss at step {step}, skipping batch")
                continue
            
            loss.backward()
            batch_loss += loss.item() * grad_acc_steps
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        accumulated_loss += batch_loss
        step += 1
        
        # Logging
        if step % 10 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step - start_step) / elapsed if elapsed > 0 else 0
            eta_seconds = (max_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600
            avg_loss = accumulated_loss / 10
            accumulated_loss = 0
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}/{max_steps} | Loss: {avg_loss:.4f} | "
                  f"LR: {current_lr:.2e} | Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h")
        
        # Validation
        if step % eval_every == 0:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    input_ids = val_batch['input_ids'].to(device)
                    labels = val_batch['labels'].to(device)
                    loss, _ = model(input_ids, labels=labels)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    if val_batches >= 100:
                        break
            
            val_loss /= val_batches
            
            print(f"\n{'='*70}")
            print(f"VALIDATION @ Step {step}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Print mixing stats for Hymba
            if model_type == "hymba":
                print(f"\nAttention vs SSM mixing (learned weights):")
                for layer_name, stats in model.get_mixing_stats().items():
                    print(f"  {layer_name}: attn_ratio={stats['attn_ratio_mean']:.3f}")
            
            print(f"{'='*70}\n")
            
            # Save checkpoint (overwrite single file to save disk space)
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'config': config.__dict__,
                'model_type': model_type,
            }
            
            # Always save latest checkpoint (overwrites previous)
            checkpoint_path = output_dir / "latest_checkpoint.pt"
            try:
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"⚠️  Warning: Failed to save checkpoint: {e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best_checkpoint.pt"
                try:
                    torch.save(checkpoint, best_path)
                    print(f"✓ New best model! Val loss: {val_loss:.4f}")
                except Exception as e:
                    print(f"⚠️  Warning: Failed to save best checkpoint: {e}")
            
            model.train()
    
    # Final validation and generation
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    # Generate sample text
    print(f"\nSample generation:")
    model.eval()
    
    prompts = ["Once upon a time", "The little dog"]
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt: '{prompt}'")
        print(f"Output: {text[:300]}...")
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train Hymba hybrid model")
    parser.add_argument("--model", type=str, default="hymba", choices=["hymba", "jamba"],
                        help="Model type: hymba (parallel heads) or jamba (interleaved)")
    parser.add_argument("--preset", type=str, default="30m", choices=["30m", "125m", "30m-fast"],
                        help="Model size preset")
    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--grad_acc_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=50,
                        help="Validation frequency")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    train_hymba(
        model_type=args.model,
        preset=args.preset,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_acc_steps=args.grad_acc_steps,
        lr=args.lr,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
