#!/usr/bin/env python3
"""
Training script for Sparse MoE Transformer.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Add shared folder to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Imports
from config import Config
from model import SparseMoETransformer
from dataset import TinyStoriesDataset
from benchmark import benchmark


def train(config: Config, resume_from: Optional[str] = None):
    """
    Main training loop for Sparse MoE Transformer.
    
    Process:
    1. Setup tokenizer and model
    2. Create dataloaders with shifted labels for causal LM
    3. Train with gradient accumulation (effective batch size = 32)
    4. Validate every 500 steps and save best checkpoint
    5. Log to W&B if enabled
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from (optional)
    """
    # Setup random seed for reproducibility
    torch.manual_seed(config.seed)
    
    # Force NVIDIA GPU usage if available (not Intel integrated graphics)
    if torch.cuda.is_available():
        # Explicitly select CUDA device 0 (NVIDIA GPU)
        device = torch.device("cuda:0")
        # Print GPU info for verification
        print(f"Using device: {device} - {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Create a test tensor on GPU to verify it's working
        test_tensor = torch.randn(1000, 1000, device=device)
        _ = test_tensor @ test_tensor  # Simple matmul to activate GPU
        print(f"✓ GPU test passed - {torch.cuda.memory_allocated(0) / 1e6:.1f} MB allocated")
        del test_tensor
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CUDA not available)")
        print("⚠️  WARNING: Training on CPU will be very slow!")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = SparseMoETransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    # Dataloaders
    print("Loading datasets...")
    train_ds = TinyStoriesDataset(tokenizer, split=config.split, max_len=config.max_len)
    val_ds = TinyStoriesDataset(tokenizer, split=config.val_split, max_len=config.max_len)
    
    # Use num_workers=0 on Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 2
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)
    print(f"✓ Loaded {len(train_ds)} training samples, {len(val_ds)} validation samples")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    # Logging
    if config.log_wandb and WANDB_AVAILABLE:
        wandb.init(project="sparse-moe-transformer", config=config.__dict__)

    # Training state
    start_step = 0
    best_val_loss = float("inf")
    
    # Resume from checkpoint if provided
    if resume_from:
        print(f"\n{'='*60}")
        print(f"RESUMING FROM CHECKPOINT: {resume_from}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step'] + 1  # Continue from next step
        best_val_loss = checkpoint['best_val_loss']
        
        # Skip RNG state restoration (not critical for training continuation)
        # The random sampling order will differ but model quality is unaffected
        
        print(f"✓ Resumed from step {start_step}")
        print(f"✓ Best validation loss: {best_val_loss:.4f}")
        print(f"✓ Will train until step {config.max_steps}\n")

    # Training timing
    training_start_time = time.time()
    last_log_time = training_start_time
    steps_at_last_log = start_step
    
    model.train()
    step = start_step

    while step < config.max_steps:
        for batch in train_dl:
            if step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, labels=labels)
            loss = out["loss"] / config.grad_acc_steps
            loss.backward()

            if (step + 1) % config.grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                current_time = time.time()
                elapsed_since_last_log = current_time - last_log_time
                steps_since_last_log = step - steps_at_last_log
                steps_per_sec = steps_since_last_log / elapsed_since_last_log if elapsed_since_last_log > 0 else 0
                
                lr = optimizer.param_groups[0]["lr"]
                # Show GPU memory usage to confirm NVIDIA GPU is being used
                if device.type == "cuda":
                    mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    print(f"[Step {step}] loss={loss.item():.4f}, lr={lr:.2e}, {steps_per_sec:.2f} steps/s | GPU mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB")
                else:
                    print(f"[Step {step}] loss={loss.item():.4f}, lr={lr:.2e}, {steps_per_sec:.2f} steps/s")
                
                last_log_time = current_time
                steps_at_last_log = step
                if config.log_wandb and WANDB_AVAILABLE:
                    wandb.log({"train_loss": loss.item(), "lr": lr}, step=step)

            if step % 500 == 0 and step > 0:
                # Validation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for val_batch in val_dl:
                        input_ids = val_batch["input_ids"].to(device)
                        labels = val_batch["labels"].to(device)
                        out = model(input_ids, labels=labels)
                        val_losses.append(out["loss"].item())
                val_loss = sum(val_losses) / len(val_losses)
                
                # Calculate timing statistics
                elapsed_total = time.time() - training_start_time
                steps_completed = step - start_step
                avg_steps_per_sec = steps_completed / elapsed_total if elapsed_total > 0 else 0
                steps_remaining = config.max_steps - step
                eta_seconds = steps_remaining / avg_steps_per_sec if avg_steps_per_sec > 0 else 0
                eta_hours = eta_seconds / 3600
                
                print(f"→ Val loss: {val_loss:.4f} | Elapsed: {elapsed_total/3600:.1f}h, Avg: {avg_steps_per_sec:.2f} steps/s, ETA: {eta_hours:.1f}h")
                if config.log_wandb and WANDB_AVAILABLE:
                    wandb.log({"val_loss": val_loss}, step=step)

                # Create checkpoint
                save_path = Path(config.output_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__,
                    'rng_state': torch.get_rng_state().cpu(),  # Ensure on CPU for saving
                }
                if torch.cuda.is_available():
                    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state().cpu()
                
                # Always save latest checkpoint (for resuming)
                torch.save(checkpoint, save_path / "latest_checkpoint.pt")
                
                # Save best model (for inference)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, save_path / "best_checkpoint.pt")
                    # Also save just the model weights for easy loading
                    torch.save(model.state_dict(), save_path / "best_model.pt")
                    print(f"✓ Saved best model (val loss: {val_loss:.4f})")

                model.train()

            step += 1

    # Final timing summary
    total_training_time = time.time() - training_start_time
    total_steps = step - start_step
    avg_steps_per_sec = total_steps / total_training_time if total_training_time > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print(f"Average speed: {avg_steps_per_sec:.2f} steps/sec")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    # Final save
    save_path = Path(config.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    final_checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config.__dict__,
        'rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        final_checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
    
    torch.save(final_checkpoint, save_path / "final_checkpoint.pt")
    torch.save(model.state_dict(), save_path / "final_model.pt")
    print("✅ Training completed.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "benchmark"], default="train")
    parser.add_argument("--model_path", type=str, default="output/sparse_moe_transformer/best_model.pt")
    parser.add_argument("--output_dir", type=str, default="output/sparse_moe_transformer")
    parser.add_argument("--log_wandb", action="store_true")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, help="Embedding dimension (default: 512)")
    parser.add_argument("--n_heads", type=int, help="Number of attention heads (default: 8)")
    parser.add_argument("--d_ff", type=int, help="FFN hidden dimension (default: 1024)")
    parser.add_argument("--n_layers", type=int, help="Number of transformer layers (default: 6)")
    parser.add_argument("--n_experts", type=int, help="Number of expert networks (default: 16)")
    parser.add_argument("--top_k", type=int, help="Active experts per token (default: 2)")
    
    # Training hyperparameters
    parser.add_argument("--max_steps", type=int, help="Total training steps (default: 5000)")
    parser.add_argument("--batch_size", type=int, help="Physical batch size (default: 4)")
    parser.add_argument("--lr", type=float, help="Learning rate (default: 3e-4)")
    parser.add_argument("--grad_acc_steps", type=int, help="Gradient accumulation steps (default: 8)")
    
    # Checkpoint resuming
    parser.add_argument("--resume_from", type=str, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()

    config = Config()
    config.output_dir = args.output_dir
    config.log_wandb = args.log_wandb
    
    # Override model architecture
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.n_heads is not None:
        config.n_heads = args.n_heads
    if args.d_ff is not None:
        config.d_ff = args.d_ff
    if args.n_layers is not None:
        config.n_layers = args.n_layers
    if args.n_experts is not None:
        config.n_experts = args.n_experts
    if args.top_k is not None:
        config.top_k = args.top_k
    
    # Override training hyperparameters
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.grad_acc_steps is not None:
        config.grad_acc_steps = args.grad_acc_steps
    
    # Validate model architecture
    if config.d_model % config.n_heads != 0:
        print(f"Error: d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads})")
        return

    if args.mode == "train":
        train(config, resume_from=args.resume_from)
    elif args.mode == "benchmark":
        benchmark(config, args.model_path)


if __name__ == "__main__":
    main()
