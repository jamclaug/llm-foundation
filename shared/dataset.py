#!/usr/bin/env python3
"""
Dataset utilities for model training.

Supports:
- TinyStories: Short children's stories (~100-500 tokens) - good for quick experiments
- PG-19: Full books from Project Gutenberg (~50K-100K tokens) - tests long-range memory
- WikiText-103: Wikipedia articles (~3-4K tokens) - medium length, factual

For long-sequence experiments, use PG-19 or WikiText with larger max_len (1024+).

Sequence Length Recommendations:
- 256 tokens: TinyStories (fast training, short context)
- 1024 tokens: WikiText-103 (medium, tests attention scaling)
- 2048+ tokens: PG-19 (long, tests Mamba vs Transformer efficiency)

Memory Usage (approximate, batch_size=4):
- 256 tokens: ~500 MB GPU
- 1024 tokens: ~2 GB GPU
- 2048 tokens: ~4 GB GPU (may need batch_size=2)

Future Improvement: RoPE (Rotary Position Embeddings)
- Current: Learned positional embeddings limited to max_len
- RoPE: Encodes position in attention computation, scales to any length
- Used by: LLaMA, Mistral, most modern LLMs
- Benefits: No retraining needed for longer sequences, better extrapolation
- Implementation: Apply rotation matrix to Q and K based on position
- Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# Default cache directory - relative to repo root
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data"


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
            cache_dir = _DEFAULT_CACHE_DIR / "tinystories"
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


class WikiTextDataset(Dataset):
    """
    Dataset wrapper for WikiText-103 (Wikipedia articles).
    
    Characteristics:
    - ~103M tokens from "Good" and "Featured" Wikipedia articles
    - Average document: ~3-4K tokens (well-suited for 1024 max_len)
    - Factual, encyclopedic content (different style from TinyStories)
    - Good for testing medium-range attention patterns
    
    Why use this:
    - Tests whether model learns factual relationships
    - Medium-length documents stress attention more than TinyStories
    - Common benchmark for perplexity comparison
    """
    def __init__(self, tokenizer, split="train", max_len=1024, cache_dir=None):
        if cache_dir is None:
            cache_dir = _DEFAULT_CACHE_DIR / "wikitext"
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Map split names (HuggingFace uses different naming)
        split_map = {"train": "train", "validation": "validation", "val": "validation", "test": "test"}
        hf_split = split_map.get(split, split)
        
        cache_file = self.cache_dir / f"wikitext103_{split}_maxlen{max_len}.arrow"
        
        print(f"  Loading WikiText-103 {split} split from HuggingFace...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=hf_split)
        print(f"  ✓ Loaded {len(ds)} samples")
        
        # WikiText has many empty lines, filter them out
        ds = ds.filter(lambda x: len(x["text"].strip()) > 50)
        print(f"  ✓ Filtered to {len(ds)} non-empty samples")
        
        if cache_file.exists():
            print(f"  Loading from cache: {cache_file}")
            from datasets import Dataset as HFDataset
            self.dataset = HFDataset.from_file(str(cache_file))
            print(f"  ✓ Loaded {len(self.dataset)} samples from cache")
        else:
            print(f"  Tokenizing {len(ds)} samples...")
            
            def tokenize_fn(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_len
                )
            
            self.dataset = ds.map(
                tokenize_fn,
                batched=True,
                batch_size=5000,
                remove_columns=["text"],
                desc=f"Tokenizing WikiText {split}"
            )
            
            # Save cache
            print(f"  Saving to cache...")
            self.dataset.save_to_disk(str(cache_file.parent / f"{cache_file.stem}_temp"))
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
        input_ids = item["input_ids"]
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)
        return {
            "input_ids": torch.tensor(input_ids[:-1], dtype=torch.long),
            "labels": torch.tensor(input_ids[1:], dtype=torch.long)
        }


class PG19Dataset(Dataset):
    """
    Dataset wrapper for PG-19 (Project Gutenberg books).
    
    Characteristics:
    - Full books, average ~70K tokens per book
    - 28,602 books spanning many genres and time periods
    - THE benchmark for long-range language modeling
    - Tests true long-term memory (references span thousands of tokens)
    
    Why use this:
    - Mamba vs Transformer efficiency test: O(L) vs O(L²) scaling
    - Tests if model can track characters, plot across long context
    - At 2048+ tokens, Transformer memory explodes, Mamba stays linear
    
    Memory considerations:
    - Books are very long, so we sample random 2048-token chunks
    - Even this is expensive: batch_size=2 recommended for 4GB GPU
    - Consider using gradient accumulation to increase effective batch
    
    Processing approach:
    - Unlike TinyStories, each "sample" is a chunk from a book
    - We create overlapping or non-overlapping chunks from each book
    - This gives us many training examples from each long book
    """
    def __init__(self, tokenizer, split="train", max_len=2048, cache_dir=None, 
                 chunks_per_book=50, overlap=256):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            split: "train", "validation", or "test"
            max_len: Length of each chunk (2048 recommended for long-range)
            cache_dir: Where to save processed data
            chunks_per_book: Max chunks to extract per book (books are very long)
            overlap: Token overlap between consecutive chunks (helps context)
        """
        if cache_dir is None:
            cache_dir = _DEFAULT_CACHE_DIR / "pg19"
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = self.cache_dir / f"pg19_{split}_maxlen{max_len}_chunks{chunks_per_book}.pt"
        
        if cache_file.exists():
            print(f"  Loading PG-19 from cache: {cache_file}")
            self.chunks = torch.load(cache_file)
            print(f"  ✓ Loaded {len(self.chunks)} chunks from cache")
        else:
            print(f"  Loading PG-19 {split} split from HuggingFace...")
            print(f"  (This dataset is ~10GB, first download takes a while)")
            ds = load_dataset("pg19", split=split, trust_remote_code=True)
            print(f"  ✓ Loaded {len(ds)} books")
            
            print(f"  Chunking books into {max_len}-token segments...")
            print(f"  (Extracting up to {chunks_per_book} chunks per book, overlap={overlap})")
            
            self.chunks = []
            for i, example in enumerate(ds):
                if i % 500 == 0:
                    print(f"    Processing book {i}/{len(ds)}...")
                
                text = example["text"]
                
                # Tokenize full book
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # Skip very short books
                if len(tokens) < max_len:
                    continue
                
                # Extract chunks with stride
                stride = max_len - overlap
                num_chunks = min(chunks_per_book, (len(tokens) - max_len) // stride + 1)
                
                for j in range(num_chunks):
                    start = j * stride
                    end = start + max_len
                    if end <= len(tokens):
                        self.chunks.append(tokens[start:end])
            
            print(f"  ✓ Created {len(self.chunks)} chunks from {len(ds)} books")
            print(f"  Saving to cache...")
            torch.save(self.chunks, cache_file)
            print(f"  ✓ Saved to {cache_file}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long)
        }


def get_dataset(name: str, tokenizer, split: str = "train", max_len: int = 256, **kwargs):
    """
    Factory function to get a dataset by name.
    
    Args:
        name: Dataset name - "tinystories", "wikitext", "pg19"
        tokenizer: HuggingFace tokenizer
        split: "train", "validation", or "test"
        max_len: Maximum sequence length
        **kwargs: Additional args passed to dataset constructor
    
    Returns:
        Dataset instance
    
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> 
        >>> # Quick experiment with TinyStories
        >>> train_ds = get_dataset("tinystories", tokenizer, max_len=256)
        >>> 
        >>> # Long-range experiment with PG-19
        >>> train_ds = get_dataset("pg19", tokenizer, max_len=2048, chunks_per_book=20)
    """
    name = name.lower().strip()
    
    if name in ("tinystories", "tiny_stories", "tiny-stories"):
        return TinyStoriesDataset(tokenizer, split=split, max_len=max_len, **kwargs)
    elif name in ("wikitext", "wikitext-103", "wikitext103"):
        return WikiTextDataset(tokenizer, split=split, max_len=max_len, **kwargs)
    elif name in ("pg19", "pg-19", "gutenberg"):
        return PG19Dataset(tokenizer, split=split, max_len=max_len, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: 'tinystories', 'wikitext', 'pg19'"
        )