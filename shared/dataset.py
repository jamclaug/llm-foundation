#!/usr/bin/env python3
"""
Dataset utilities for model training.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# Default cache directory - relative to repo root
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "tinystories"


class TinyStoriesDataset(Dataset):
    """
    Dataset wrapper for TinyStories (simple children's stories).
    
    Creates causal language modeling pairs:
    - Input: tokens[:-1]
    - Label: tokens[1:]  (shifted by 1 position)
    
    This trains the model to predict the next token given previous tokens.
    
    Implements chunked tokenization with disk caching to handle large datasets
    without running out of memory.
    """
    def __init__(self, tokenizer, split="train", max_len=256, cache_dir=None):
        if cache_dir is None:
            cache_dir = _DEFAULT_CACHE_DIR
        print(f"  [DEBUG] TinyStoriesDataset.__init__ called for split='{split}'")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [DEBUG] Cache directory: {self.cache_dir.absolute()}")
        
        # Create unique cache filename based on split and max_len
        cache_file = self.cache_dir / f"tinystories_{split}_maxlen{max_len}.arrow"
        print(f"  [DEBUG] Cache file: {cache_file}")
        
        print(f"  Loading {split} split from HuggingFace...")
        print(f"  (First time: downloads ~500MB dataset from internet)")
        print(f"  (Future runs: loads from cache instantly)")
        print(f"  [DEBUG] About to call load_dataset()...")
        ds = load_dataset("roneneldan/TinyStories", split=split)
        print(f"  [DEBUG] load_dataset() returned successfully")
        print(f"  ✓ Downloaded/loaded {len(ds)} samples")
        
        # Check if already tokenized and cached
        if cache_file.exists():
            print(f"  Found cached tokenized dataset at {cache_file}")
            print(f"  Loading from cache...")
            from datasets import Dataset as HFDataset
            self.dataset = HFDataset.from_file(str(cache_file))
            print(f"  ✓ Loaded {len(self.dataset)} samples from cache")
        else:
            print(f"  No cache found. Tokenizing {len(ds)} samples...")
            print(f"  This will be saved to {cache_file} for future runs")
            print(f"  Processing in chunks of 10,000 to avoid memory issues...")
            
            # Tokenize in chunks to avoid memory issues
            def tokenize_fn(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_len
                )
            
            # Process with progress tracking
            self.dataset = ds.map(
                tokenize_fn,
                batched=True,
                batch_size=10000,  # Process 10k at a time
                remove_columns=["text"],
                desc=f"Tokenizing {split}"
            )
            
            # Save to cache for next time
            print(f"  Saving tokenized dataset to cache...")
            self.dataset.save_to_disk(str(cache_file.parent / f"{cache_file.stem}_temp"))
            # Arrow format is more efficient
            import shutil
            arrow_file = cache_file.parent / f"{cache_file.stem}_temp" / "data-00000-of-00001.arrow"
            if arrow_file.exists():
                shutil.move(str(arrow_file), str(cache_file))
                shutil.rmtree(str(cache_file.parent / f"{cache_file.stem}_temp"))
            
            print(f"  ✓ Tokenized and cached {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # HuggingFace datasets return lists, convert to tensor
        input_ids = item["input_ids"]
        
        # Ensure it's a list before slicing
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)
        
        # Create (input, label) shifted pair for causal LM
        return {
            "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(input_ids[1:], dtype=torch.long)
        }
