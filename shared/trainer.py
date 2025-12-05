#!/usr/bin/env python3
"""
Shared Training Module for LLM Foundation.

A unified training harness that all models can use. Features:
- Mixed precision training (fp16/bf16) for ~2x speedup
- torch.compile support for ~30% speedup
- Gradient checkpointing for longer sequences
- Early stopping to save time on bad runs
- Unified logging (console + optional W&B)
- Automatic checkpointing and resume
- Profiling support

Usage:
    from shared.trainer import Trainer, TrainerConfig
    
    config = TrainerConfig(
        max_steps=5000,
        batch_size=4,
        learning_rate=3e-4,
        mixed_precision="fp16",  # or "bf16", "none"
        compile_model=True,
    )
    
    trainer = Trainer(model, train_dataset, val_dataset, config)
    trainer.train()
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainerConfig:
    """Configuration for the unified Trainer."""
    
    # Training steps
    max_steps: int = 5000
    warmup_steps: int = 500
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 10
    generate_every: int = 1000
    
    # Batch size
    batch_size: int = 4
    grad_accumulation_steps: int = 8
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    betas: tuple = (0.9, 0.999)
    
    # Performance optimizations
    mixed_precision: str = "fp16"  # "fp16", "bf16", or "none"
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    gradient_checkpointing: bool = False  # Trade compute for memory
    num_workers: int = 0  # DataLoader workers (0 for Windows compatibility)
    
    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Logging
    log_wandb: bool = False
    wandb_project: str = "llm-foundation"
    wandb_run_name: Optional[str] = None
    
    # Checkpointing
    output_dir: str = "output"
    save_total_limit: int = 3  # Keep only N best checkpoints
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: Optional[str] = None  # Auto-detect if None
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accumulation_steps


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self, log_wandb: bool = False):
        self.log_wandb = log_wandb and WANDB_AVAILABLE
        self.history: Dict[str, List[float]] = {}
    
    def log(self, metrics: Dict[str, float], step: int):
        """Log metrics to console and optionally W&B."""
        # Store in history
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))
        
        # Log to W&B
        if self.log_wandb:
            wandb.log(metrics, step=step)
    
    def get_history(self, key: str) -> List[tuple]:
        """Get history for a specific metric."""
        return self.history.get(key, [])


class CheckpointManager:
    """Manage model checkpoints with automatic cleanup."""
    
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_total_limit = save_total_limit
        self.checkpoints: List[tuple] = []  # (val_loss, path)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step: int,
        val_loss: float,
        config: dict,
        is_best: bool = False,
    ):
        """Save a checkpoint and manage old ones."""
        # Always save latest
        latest_path = self.output_dir / "latest_checkpoint.pt"
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'config': config,
        }
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best checkpoint (val_loss={val_loss:.4f})")
        
        # Save numbered checkpoint
        step_path = self.output_dir / f"checkpoint_step{step}.pt"
        torch.save(checkpoint, step_path)
        self.checkpoints.append((val_loss, step_path))
        
        # Cleanup old checkpoints (keep best N)
        if len(self.checkpoints) > self.save_total_limit:
            self.checkpoints.sort(key=lambda x: x[0])  # Sort by val_loss
            while len(self.checkpoints) > self.save_total_limit:
                _, old_path = self.checkpoints.pop()
                if old_path.exists() and "best" not in str(old_path):
                    old_path.unlink()
    
    def load_latest(self, model, optimizer=None, scheduler=None, device='cpu'):
        """Load the latest checkpoint if it exists."""
        latest_path = self.output_dir / "latest_checkpoint.pt"
        if not latest_path.exists():
            return None
        
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint


class Trainer:
    """
    Unified trainer for all LLM Foundation models.
    
    Handles:
    - Training loop with gradient accumulation
    - Mixed precision (fp16/bf16)
    - Model compilation (torch.compile)
    - Checkpointing and auto-resume
    - Early stopping
    - Metrics logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainerConfig,
        tokenizer=None,  # For text generation during training
        compute_metrics: Optional[Callable] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.compute_metrics = compute_metrics
        
        # Setup device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reproducibility
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Model setup
        self.model = model.to(self.device)
        
        # Gradient checkpointing
        if config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            else:
                print("⚠ Model doesn't support gradient_checkpointing_enable()")
        
        # Compile model (PyTorch 2.0+)
        if config.compile_model:
            if hasattr(torch, 'compile'):
                print("Compiling model with torch.compile...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("✓ Model compiled")
            else:
                print("⚠ torch.compile requires PyTorch 2.0+")
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )
        
        # Mixed precision
        self.scaler = None
        self.autocast_dtype = None
        if config.mixed_precision != "none" and torch.cuda.is_available():
            if config.mixed_precision == "fp16":
                self.autocast_dtype = torch.float16
                self.scaler = GradScaler()
                print("✓ Mixed precision: fp16")
            elif config.mixed_precision == "bf16":
                if torch.cuda.is_bf16_supported():
                    self.autocast_dtype = torch.bfloat16
                    # bf16 doesn't need GradScaler
                    print("✓ Mixed precision: bf16")
                else:
                    print("⚠ bf16 not supported on this GPU, using fp32")
        
        # Early stopping
        self.early_stopping = None
        if config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        
        # Metrics and checkpointing
        self.metrics = MetricsTracker(log_wandb=config.log_wandb)
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            save_total_limit=config.save_total_limit,
        )
        
        # W&B init
        if config.log_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )
        
        # State
        self.step = 0
        self.best_val_loss = float('inf')
        self.start_time = None
    
    def _get_autocast_context(self):
        """Get the appropriate autocast context for mixed precision."""
        if self.autocast_dtype is not None:
            return autocast(dtype=self.autocast_dtype)
        return nullcontext()
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with gradient accumulation."""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with self._get_autocast_context():
            output = self.model(input_ids, labels=labels)
            loss = output['loss'] / self.config.grad_accumulation_steps
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠ NaN/Inf loss detected, skipping batch")
            return 0.0
        
        # Backward
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.grad_accumulation_steps
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            with self._get_autocast_context():
                output = self.model(input_ids, labels=labels)
            
            total_loss += output['loss'].item()
            num_batches += 1
            
            # Limit eval batches for speed
            if num_batches >= 100:
                break
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def generate_samples(self, prompts: List[str], max_new_tokens: int = 100):
        """Generate sample text during training."""
        if self.tokenizer is None:
            return
        
        self.model.eval()
        print("\n📝 Sample generations:")
        
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Check if model has generate method
            if hasattr(self.model, 'generate'):
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    top_k=50,
                )
                text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            else:
                text = prompt + " [generate() not implemented]"
            
            print(f"  Prompt: '{prompt}'")
            print(f"  Output: {text[:200]}...")
        
        self.model.train()
    
    def train(self, resume: bool = True) -> float:
        """
        Main training loop.
        
        Args:
            resume: If True, attempt to resume from latest checkpoint.
        
        Returns:
            Best validation loss achieved.
        """
        # Print training info
        print(f"\n{'='*70}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Max steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.batch_size} x {self.config.grad_accumulation_steps} = {self.config.effective_batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision: {self.config.mixed_precision}")
        print(f"Compile: {self.config.compile_model}")
        print(f"Gradient checkpointing: {self.config.gradient_checkpointing}")
        print(f"Output: {self.config.output_dir}")
        print(f"{'='*70}\n")
        
        # Resume from checkpoint
        if resume:
            checkpoint = self.checkpoint_manager.load_latest(
                self.model, self.optimizer, self.scheduler, self.device
            )
            if checkpoint:
                self.step = checkpoint.get('step', 0)
                self.best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"✓ Resumed from step {self.step} (val_loss={self.best_val_loss:.4f})")
        
        # Training loop
        self.model.train()
        self.start_time = time.time()
        train_iter = iter(self.train_loader)
        accumulated_loss = 0
        
        while self.step < self.config.max_steps:
            # Accumulate gradients
            for micro_step in range(self.config.grad_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                
                loss = self._train_step(batch)
                accumulated_loss += loss
            
            # Optimizer step
            self._optimizer_step()
            self.step += 1
            avg_loss = accumulated_loss / self.config.grad_accumulation_steps
            accumulated_loss = 0
            
            # Logging
            if self.step % self.config.log_every == 0:
                elapsed = time.time() - self.start_time
                steps_per_sec = self.step / elapsed
                eta_hours = (self.config.max_steps - self.step) / steps_per_sec / 3600
                lr = self.scheduler.get_last_lr()[0]
                
                self.metrics.log({
                    'train_loss': avg_loss,
                    'learning_rate': lr,
                    'steps_per_sec': steps_per_sec,
                }, step=self.step)
                
                print(f"Step {self.step}/{self.config.max_steps} | "
                      f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                      f"Speed: {steps_per_sec:.2f} steps/s | ETA: {eta_hours:.1f}h")
            
            # Evaluation
            if self.step % self.config.eval_every == 0:
                val_loss = self.evaluate()
                
                print(f"\n{'='*50}")
                print(f"VALIDATION @ Step {self.step}")
                print(f"Val Loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")
                
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print("🎉 New best!")
                
                self.metrics.log({'val_loss': val_loss}, step=self.step)
                print(f"{'='*50}\n")
                
                # Save checkpoint
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.step,
                    val_loss=val_loss,
                    config=vars(self.config),
                    is_best=is_best,
                )
                
                # Early stopping
                if self.early_stopping and self.early_stopping(val_loss):
                    print(f"⏹ Early stopping triggered after {self.early_stopping.counter} "
                          f"evaluations without improvement")
                    break
            
            # Generate samples
            if self.step % self.config.generate_every == 0 and self.tokenizer:
                self.generate_samples(["Once upon a time", "The little dog"])
        
        # Training complete
        elapsed = time.time() - self.start_time
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total steps: {self.step}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Training time: {elapsed/3600:.2f} hours")
        print(f"Final checkpoint: {self.config.output_dir}")
        print(f"{'='*70}\n")
        
        # Generate final samples
        if self.tokenizer:
            self.generate_samples(
                ["Once upon a time", "The little dog", "In a magical forest"],
                max_new_tokens=150
            )
        
        # Close W&B
        if self.config.log_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        return self.best_val_loss


def create_trainer(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer=None,
    output_dir: str = "output",
    max_steps: int = 5000,
    batch_size: int = 4,
    grad_accumulation_steps: int = 8,
    learning_rate: float = 3e-4,
    mixed_precision: str = "fp16",
    compile_model: bool = False,
    gradient_checkpointing: bool = False,
    early_stopping: bool = False,
    log_wandb: bool = False,
    **kwargs,
) -> Trainer:
    """
    Convenience function to create a Trainer with common defaults.
    
    Example:
        trainer = create_trainer(
            model=my_model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            tokenizer=tokenizer,
            output_dir="output/my_experiment",
            max_steps=5000,
            mixed_precision="fp16",
        )
        trainer.train()
    """
    config = TrainerConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        mixed_precision=mixed_precision,
        compile_model=compile_model,
        gradient_checkpointing=gradient_checkpointing,
        early_stopping=early_stopping,
        log_wandb=log_wandb,
        **kwargs,
    )
    
    return Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        tokenizer=tokenizer,
    )
