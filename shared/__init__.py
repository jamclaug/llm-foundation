"""Shared utilities for all models."""

# Core imports (always available)
from .config import Config, MambaConfig
from .dataset import TinyStoriesDataset, WikiTextDataset, PG19Dataset, get_dataset
from .trainer import Trainer, TrainerConfig, create_trainer
from .utils import count_parameters

# Optional imports (may have additional dependencies)
try:
    from .benchmark import benchmark
except ImportError:
    benchmark = None
