"""
Training Script for Time Series Transformer.

This script demonstrates how to train the TimeSeriesTransformer model
on synthetic data. It uses a simple training loop (not the shared trainer)
for clarity and full control.

Understanding the Loss Function
-------------------------------

We use Mean Squared Error (MSE) loss for next-window prediction:

    loss = MSE(predicted_windows, actual_windows)

Why MSE is Appropriate for Time Series Forecasting:

1. **Quadratic Penalty**: MSE penalizes large errors more than small ones.
   A prediction error of 2 contributes 4 to the loss, while error of 0.5
   contributes only 0.25. This is desirable because big prediction misses
   (e.g., predicting market goes up 5% when it crashes 10%) are much worse
   than small misses.

2. **Proper Scoring Rule**: MSE is "proper" meaning the optimal prediction
   under MSE is the true conditional expectation E[y|x]. The model can't
   game the metric by predicting something other than its best estimate.

3. **Decomposition**: MSE = Bias² + Variance + Irreducible Error
   - Bias²: systematic over/under-prediction
   - Variance: prediction instability
   - Irreducible: inherent noise in data
   This helps diagnose what's wrong if the model underperforms.

4. **Scale Interpretability**: RMSE = √MSE has the same units as the data.
   If your time series is temperature in °C and RMSE = 0.5, your predictions
   are off by about 0.5°C on average.

Interpreting Training Progress
------------------------------

What loss values mean (depends on data scale!):
- Loss decreasing: Model is learning patterns
- Loss plateau: Model converged (or needs more capacity/data)
- Loss increasing: Overfitting (use more regularization/dropout)

Baseline Comparisons (computed during training):
- **Random Walk Baseline**: Predict previous value (x[t+1] = x[t])
  If model MSE > baseline MSE, model isn't learning useful patterns!
- **Skill Score**: 1 - (model_mse / baseline_mse)
  > 0: Better than baseline
  = 0: Same as baseline (useless model)
  < 0: Worse than baseline (broken model)

For synthetic data:
- Sine waves: Model should easily beat baseline (patterns are deterministic)
- Random walk: Model can only match baseline (no predictable pattern)
- AR process: Model should beat baseline (there are learnable dependencies)

Usage
-----
    # Basic training
    python -m models.timeseries.train
    
    # With custom settings
    python -m models.timeseries.train --num_streams 5 --d_model 128 --n_layers 4
"""

import argparse
import gzip
import io
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.timeseries.model import TimeSeriesTransformer, TimeSeriesTransformerConfig
from models.timeseries.model_mamba import TimeSeriesMamba, TimeSeriesMambaConfig
from models.timeseries.data import create_train_val_datasets


def save_checkpoint_compressed(checkpoint: Dict, path: Path) -> None:
    """Save checkpoint with gzip compression (~60-70% smaller)."""
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer, _use_new_zipfile_serialization=True)
    buffer.seek(0)
    with gzip.open(path, 'wb', compresslevel=6) as f:
        f.write(buffer.read())


def load_checkpoint_compressed(path: Path) -> Dict:
    """Load gzip-compressed checkpoint."""
    with gzip.open(path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer, weights_only=False)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 50,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns dict with average metrics for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_baseline_mse = 0.0
    num_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        values = batch["values"].to(device)
        targets = batch["targets"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(values, targets=targets)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        if outputs["loss_details"]:
            total_mse += outputs["loss_details"]["mse"]
            total_baseline_mse += outputs["loss_details"]["baseline_mse"]
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * dataloader.batch_size / elapsed
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.6f} | {samples_per_sec:.1f} samples/sec")
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_baseline_mse = total_baseline_mse / num_batches
    skill_score = 1.0 - (avg_mse / (avg_baseline_mse + 1e-8))
    
    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "rmse": avg_mse ** 0.5,
        "baseline_mse": avg_baseline_mse,
        "baseline_rmse": avg_baseline_mse ** 0.5,
        "skill_score": skill_score,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Returns dict with average metrics.
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_baseline_mse = 0.0
    num_batches = 0
    
    for batch in dataloader:
        values = batch["values"].to(device)
        targets = batch["targets"].to(device)
        
        outputs = model(values, targets=targets)
        
        total_loss += outputs["loss"].item()
        if outputs["loss_details"]:
            total_mse += outputs["loss_details"]["mse"]
            total_mae += outputs["loss_details"]["mae"]
            total_baseline_mse += outputs["loss_details"]["baseline_mse"]
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches
    avg_baseline_mse = total_baseline_mse / num_batches
    skill_score = 1.0 - (avg_mse / (avg_baseline_mse + 1e-8))
    
    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "rmse": avg_mse ** 0.5,
        "mae": avg_mae,
        "baseline_mse": avg_baseline_mse,
        "baseline_rmse": avg_baseline_mse ** 0.5,
        "skill_score": skill_score,
    }


def train(
    # Data settings
    generator: str = "sine_waves",
    train_samples: int = 10000,
    val_samples: int = 1000,
    time_steps: int = 200,
    num_streams: int = 10,
    
    # Model settings
    backend: str = "transformer",  # "transformer" or "mamba"
    window_size: int = 5,
    num_bins: int = 50,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,  # Transformer only
    d_state: int = 16,  # Mamba only
    d_conv: int = 4,    # Mamba only
    expand: int = 2,    # Mamba only
    dropout: float = 0.1,
    delta_mode: str = "percent",
    
    # Training settings
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    save_every: int = 5,  # Save checkpoint every N epochs
    resume: Optional[str] = None,  # Path to checkpoint to resume from
    
    # System settings
    device: Optional[str] = None,
    output_dir: str = "output/timeseries",
    seed: int = 42,
):
    """
    Main training function.
    
    Supports both Transformer and Mamba backends:
    - Transformer: O(n²) attention, good for short sequences (<200 steps)
    - Mamba: O(n) SSM, good for long sequences (200-5000+ steps)
    
    This function:
    1. Creates synthetic data
    2. Initializes the model
    3. Trains for specified epochs
    4. Saves the best model checkpoint
    
    The printed metrics help you understand if the model is learning:
    - **Loss decreasing**: Good, model is learning
    - **Skill Score > 0**: Model beats the "predict previous value" baseline
    - **Skill Score ≈ 0**: Model no better than baseline (might be expected for random walks)
    - **Skill Score < 0**: Something is wrong
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # =================================================================
    # Create Data
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Creating {generator} dataset...")
    print(f"{'='*60}")
    
    # Centralized cache directory (shared across all runs)
    cache_dir = Path("output/timeseries_cache")
    
    train_dataset, val_dataset = create_train_val_datasets(
        generator=generator,
        train_samples=train_samples,
        val_samples=val_samples,
        time_steps=time_steps,
        num_streams=num_streams,
        normalize=True,
        seed=seed,
        cache_dir=str(cache_dir),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep simple for debugging
        pin_memory=(device.type == "cuda"),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # =================================================================
    # Create Model
    # =================================================================
    print(f"\n{'='*60}")
    print(f"Creating {backend} model...")
    print(f"{'='*60}")
    
    if backend == "transformer":
        config = TimeSeriesTransformerConfig(
            num_streams=num_streams,
            window_size=window_size,
            num_bins=num_bins,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            delta_mode=delta_mode,
        )
        model = TimeSeriesTransformer(config)
    elif backend == "mamba":
        config = TimeSeriesMambaConfig(
            num_streams=num_streams,
            window_size=window_size,
            num_bins=num_bins,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            delta_mode=delta_mode,
        )
        model = TimeSeriesMamba(config)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'transformer' or 'mamba'")
    
    model = model.to(device)
    
    # =================================================================
    # Setup Optimizer and Scheduler
    # =================================================================
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Cosine annealing: LR decreases smoothly to 0 over training
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader),
        eta_min=learning_rate * 0.01,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    
    if resume:
        print(f"\nResuming from checkpoint: {resume}")
        # Support both compressed (.gz) and uncompressed formats
        if resume.endswith('.gz'):
            checkpoint = load_checkpoint_compressed(Path(resume))
        else:
            checkpoint = torch.load(resume, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_epoch = checkpoint.get("best_epoch", 0)
        print(f"  Resuming from epoch {start_epoch}, best_val_loss: {best_val_loss:.6f}")
    
    # =================================================================
    # Training Loop
    # =================================================================
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    print(f"\nMetrics explanation:")
    print(f"  - MSE/RMSE: Lower is better (prediction error)")
    print(f"  - Baseline: 'Predict previous value' performance")
    print(f"  - Skill Score: >0 means beating baseline, <0 means losing")
    print()
    
    for epoch in range(start_epoch, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            log_interval=max(1, len(train_loader) // 5),
        )
        
        # Update scheduler
        scheduler.step()
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        
        # Print epoch summary
        print(f"\n  Train | Loss: {train_metrics['loss']:.6f} | "
              f"RMSE: {train_metrics['rmse']:.4f} | "
              f"Baseline RMSE: {train_metrics['baseline_rmse']:.4f} | "
              f"Skill: {train_metrics['skill_score']:.4f}")
        print(f"  Val   | Loss: {val_metrics['loss']:.6f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | "
              f"Baseline RMSE: {val_metrics['baseline_rmse']:.4f} | "
              f"Skill: {val_metrics['skill_score']:.4f}")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
            save_checkpoint_compressed(checkpoint, output_path / "best_model.pt.gz")
            print(f"  ✓ Saved new best model (val_loss: {best_val_loss:.6f})")
        
        # Save latest checkpoint for resuming (overwrites previous)
        if save_every > 0 and epoch % save_every == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
            # Always overwrite to save disk space - only keep latest checkpoint
            save_checkpoint_compressed(checkpoint, output_path / "latest_checkpoint.pt.gz")
            print(f"  ✓ Saved latest checkpoint (epoch {epoch})")
    
    # =================================================================
    # Final Summary
    # =================================================================
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_path / 'best_model.pt.gz'}")
    
    # Use last epoch's val metrics for final summary (model still in memory)
    final_metrics = val_metrics
    
    print(f"\nFinal Evaluation:")
    print(f"  MSE: {final_metrics['mse']:.6f}")
    print(f"  RMSE: {final_metrics['rmse']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  Skill Score: {final_metrics['skill_score']:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if final_metrics["skill_score"] > 0.1:
        print(f"  ✓ Model significantly beats baseline (skill={final_metrics['skill_score']:.2f})")
        print(f"    The model has learned predictable patterns in the data.")
    elif final_metrics["skill_score"] > 0:
        print(f"  ~ Model slightly beats baseline (skill={final_metrics['skill_score']:.2f})")
        print(f"    Some learning, but patterns may be weak or noisy.")
    else:
        print(f"  ✗ Model does not beat baseline (skill={final_metrics['skill_score']:.2f})")
        print(f"    Either data has no predictable patterns (random walk),")
        print(f"    or model needs tuning (more capacity, different hyperparams).")
    
    return model, final_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Time Series Model (Transformer or Mamba)")
    
    # Data arguments
    parser.add_argument("--generator", type=str, default="sine_waves",
                        choices=["sine_waves", "random_walk", "ar_process", 
                                "trend_seasonal", "regime_switching", "regime_ar", "mixed"],
                        help="Type of synthetic data to generate")
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--val_samples", type=int, default=1000)
    parser.add_argument("--time_steps", type=int, default=200)
    parser.add_argument("--num_streams", type=int, default=10)
    
    # Model arguments
    parser.add_argument("--backend", type=str, default="transformer",
                        choices=["transformer", "mamba"],
                        help="Model backend: transformer (O(n²), short seq) or mamba (O(n), long seq)")
    parser.add_argument("--window_size", type=int, default=5,
                        help="Time steps per window (use larger for mamba, e.g., 50-200)")
    parser.add_argument("--num_bins", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Attention heads (transformer only)")
    parser.add_argument("--d_state", type=int, default=16,
                        help="SSM state dimension (mamba only)")
    parser.add_argument("--d_conv", type=int, default=4,
                        help="Local convolution width (mamba only)")
    parser.add_argument("--expand", type=int, default=2,
                        help="Inner dimension expansion (mamba only)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta_mode", type=str, default="percent",
                        choices=["absolute", "percent", "log_return"])
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs (0 to disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # System arguments
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/timeseries")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(**vars(args))


if __name__ == "__main__":
    main()
