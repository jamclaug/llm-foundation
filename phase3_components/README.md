# Phase 3: Components

Supporting infrastructure - datasets, tokenizers, and optimizers.

## Learning Objectives
- Implement or integrate tokenization strategies (BPE, WordPiece)
- Build data loading pipelines for language modeling
- Understand optimizer variants (Adam, AdamW, learning rate schedules)

## Directory Structure

### `notebooks/`
Interactive exploration (to be created):
- Tokenization strategies comparison
- Dataset preprocessing workflows
- Optimizer behavior visualization

### `src/`
Implementation files (placeholders):
- `tokenizers.py` - BPE, WordPiece, character-level tokenizers
- `datasets.py` - Data loaders, batching, text preprocessing
- `optimizers.py` - Adam, AdamW with warmup/decay schedules

### `tests/`
- Validation scripts for data pipelines

## Implementation Notes
- May leverage HuggingFace tokenizers for reference
- Focus on understanding tokenization algorithms
- Build efficient data loading for training
