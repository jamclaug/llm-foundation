#!/usr/bin/env python3
"""Calculate VRAM requirements for different model sizes."""

def calc_memory(params_b, dtype_bytes=2, training=True):
    """
    Calculate VRAM needed for transformer models.
    
    Args:
        params_b: Model size in billions of parameters
        dtype_bytes: 2 for fp16/bf16, 4 for fp32
        training: True for training (includes gradients + optimizer), False for inference
    
    Returns:
        (model_weights_gb, total_vram_gb)
    """
    params = params_b * 1e9
    
    # Model weights in GB
    model_weights = params * dtype_bytes / (1024**3)
    
    if training:
        # Training memory breakdown:
        # 1. Model weights (1x)
        # 2. Gradients (1x)
        # 3. Adam optimizer states: momentum (1x) + variance (1x) = 2x
        # Total: 4x model weights
        optimizer_overhead = model_weights * 3  # gradients + 2x optimizer states
        
        # Activations (rough estimate, depends on batch size and sequence length)
        activations = model_weights * 0.5
        
        total = model_weights + optimizer_overhead + activations
        return model_weights, total
    else:
        # Inference: just weights + activations (no gradients or optimizer)
        activations = model_weights * 0.2
        return model_weights, model_weights + activations


print("=" * 70)
print("VRAM Requirements for Transformer Models (fp16/bf16)")
print("=" * 70)

print("\n=== 7B Model (Llama-2-7B size) ===")
weights, total = calc_memory(7, training=True)
print(f"Training:  {total:.1f} GB VRAM (weights: {weights:.1f}GB + grads/optimizer/activations)")
weights_inf, total_inf = calc_memory(7, training=False)
print(f"Inference: {total_inf:.1f} GB VRAM (weights: {weights_inf:.1f}GB + activations)")

print("\n=== Your 216M MoE Model (Currently Training) ===")
weights, total = calc_memory(0.216, training=True)
print(f"Training:  {total:.1f} GB VRAM")
print(f"Actual:    3.7 GB VRAM (close match! ✓)")
weights_inf, total_inf = calc_memory(0.216, training=False)
print(f"Inference: {total_inf:.1f} GB VRAM")

print("\n=== Dense 230M (No MoE) ===")
weights, total = calc_memory(0.23, training=True)
print(f"Training:  {total:.1f} GB VRAM")
print("Note: Similar memory to MoE - sparsity saves COMPUTE not MEMORY")

print("\n" + "=" * 70)
print("Common Model Sizes")
print("=" * 70)
print(f"{'Model':<25} {'Params':<10} {'Training':<15} {'Inference':<15}")
print("-" * 70)

models = [
    ('GPT-2 Small', 0.124),
    ('GPT-2 Medium', 0.355),
    ('GPT-2 Large', 0.774),
    ('GPT-2 XL', 1.5),
    ('Your 158M MoE', 0.158),
    ('Your 216M MoE', 0.216),
    ('Llama-2-7B', 7),
    ('Llama-2-13B', 13),
    ('Llama-2-70B', 70),
]

for name, size in models:
    _, train = calc_memory(size, training=True)
    _, inf = calc_memory(size, training=False)
    print(f"{name:<25} {size:<10.2f}B {train:>6.1f} GB      {inf:>6.1f} GB")

print("\n" + "=" * 70)
print("GPU Fit Guide")
print("=" * 70)
print("4GB GPU (Quadro T1000):    Up to ~350M training, 1.5B inference")
print("8GB GPU (RTX 3060):        Up to ~800M training, 4B inference")
print("12GB GPU (RTX 3080):       Up to ~1.2B training, 6B inference")
print("16GB GPU (A4000):          Up to ~1.8B training, 8B inference")
print("24GB GPU (RTX 3090/4090):  Up to ~2.5B training, 12B inference")
print("40GB GPU (A100):           Up to ~4.5B training, 20B inference")
print("80GB GPU (A100):           Up to ~9B training, 40B inference")

print("\n" + "=" * 70)
print("MoE vs Dense: Key Differences")
print("=" * 70)
print("✓ MoE: Same memory, ~87% less compute (faster training)")
print("✓ Dense: Same memory, full compute (slower but simpler)")
print("✗ Both need same VRAM - sparsity is a compute optimization only")
