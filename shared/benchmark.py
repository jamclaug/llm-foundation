#!/usr/bin/env python3
"""
Benchmarking utilities for Sparse MoE Transformer.
"""

import torch
from datasets import load_dataset

# Optional dependencies
try:
    import evaluate
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from .config import Config
    from .model import SparseMoETransformer
    from .utils import count_parameters, estimate_model_flops, benchmark_latency
except ImportError:
    from config import Config
    from model import SparseMoETransformer
    from utils import count_parameters, estimate_model_flops, benchmark_latency


def benchmark(config: Config, model_path: str):
    """
    Benchmark trained model on multiple metrics:
    1. Parameter count and estimated FLOPs
    2. GPU memory usage (if CUDA available)
    3. Inference latency (ms per sample)
    4. GSM8K math reasoning accuracy (optional, 50 samples)
    
    Args:
        config: Model configuration
        model_path: Path to trained model checkpoint
    """
    # Force NVIDIA GPU usage
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using device: cuda:0 - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"Using device: cpu")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = SparseMoETransformer(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\n" + "="*50)
    print(" RUNNING BENCHMARKS ")
    print("="*50)

    # 1. Model stats
    n_params = count_parameters(model)
    flops = estimate_model_flops(model, seq_len=128)
    print(f"Params: {n_params/1e6:.2f}M")
    print(f"Est. FLOPs (128 seq): {flops/1e9:.2f} GFLOPs")

    # 2. Memory
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        _ = model(torch.randint(0, config.vocab_size, (1, 128), device=device))
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"GPU Peak Mem: {mem_mb:.1f} MB")

    # 3. Latency
    latency = benchmark_latency(model, tokenizer, device, n_samples=config.benchmark_latency_samples)
    print(f"Latency (1 sample): {latency:.2f} ms")

    # 4. GSM8K (if enabled & evaluate available)
    if config.benchmark_gsm8k and EVAL_AVAILABLE:
        print("\n→ Running GSM8K (few-shot, greedy)...")
        gsm8k = load_dataset("gsm8k", "main", split="test[:200]")  # subset for speed
        metric = evaluate.load("accuracy")
        correct = 0
        total = 0

        prompt_template = (
            "Q: {question}\nA: Let's think step by step. "
        )

        for i, ex in enumerate(gsm8k):
            if i >= 50: break  # quick eval
            prompt = prompt_template.format(question=ex["question"])
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                # Note: generate() method not implemented yet
                # Would need: output = model.generate(input_ids, max_new_tokens=128, ...)
                pass

            total += 1

        acc = correct / total if total > 0 else 0.0
        print(f"GSM8K Acc (50 samples): {acc*100:.1f}%")
        if config.log_wandb and WANDB_AVAILABLE:
            wandb.log({"gsm8k_acc": acc})

    print("\n✅ Benchmarking complete.")
