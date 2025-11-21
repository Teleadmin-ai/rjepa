#!/usr/bin/env python
"""
Clear CUDA memory completely.

This script forces PyTorch to release all GPU memory.
"""
import gc
import torch

print("=" * 80)
print("CUDA MEMORY CLEANUP")
print("=" * 80)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    exit(1)

print(f"\nBefore cleanup:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Force garbage collection
print("\nRunning garbage collection...")
gc.collect()

# Empty CUDA cache
print("Emptying CUDA cache...")
torch.cuda.empty_cache()

# Reset memory stats
print("Resetting memory statistics...")
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()

# Synchronize
print("Synchronizing CUDA...")
torch.cuda.synchronize()

print(f"\nAfter cleanup:")
print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\nCUDA memory cleared successfully!")
print("=" * 80)
