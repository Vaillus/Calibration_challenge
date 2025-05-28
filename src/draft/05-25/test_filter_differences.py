#!/usr/bin/env python3
"""
Test to analyze the exact differences between sequential and batch filtering methods.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path

from flow_filter import FlowFilter

def test_filter_differences():
    """Test the differences between sequential and batch filtering."""
    print("Analyzing filter differences...")
    
    # Load one flow
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow_np = data['flow'][0]
    
    print(f"Flow shape: {single_flow_np.shape}")
    print(f"Flow dtype: {single_flow_np.dtype}")
    
    # Initialize filter
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    # Sequential filtering
    print("\n=== Sequential filtering ===")
    filtered_flow_seq, weights_seq = flow_filter.filter_by_norm(single_flow_np)
    print(f"Sequential filtered flow dtype: {filtered_flow_seq.dtype}")
    print(f"Sequential weights dtype: {weights_seq.dtype}")
    print(f"Sequential weights min/max: {np.min(weights_seq):.8f} / {np.max(weights_seq):.8f}")
    
    # Batch filtering
    print("\n=== Batch filtering ===")
    single_flow_batch = mx.array(single_flow_np[None, :, :, :])  # Add batch dimension
    filtered_flow_batch, weights_batch = flow_filter.filter_by_norm_batch(single_flow_batch)
    
    # Convert back to numpy for comparison
    filtered_flow_batch_np = np.array(filtered_flow_batch[0])
    weights_batch_np = np.array(weights_batch[0])
    
    print(f"Batch filtered flow dtype: {filtered_flow_batch_np.dtype}")
    print(f"Batch weights dtype: {weights_batch_np.dtype}")
    print(f"Batch weights min/max: {np.min(weights_batch_np):.8f} / {np.max(weights_batch_np):.8f}")
    
    # Compare results
    print("\n=== Comparison ===")
    flow_diff = np.abs(filtered_flow_seq - filtered_flow_batch_np)
    weights_diff = np.abs(weights_seq - weights_batch_np)
    
    print(f"Max flow difference: {np.max(flow_diff):.10f}")
    print(f"Max weights difference: {np.max(weights_diff):.10f}")
    print(f"Mean weights difference: {np.mean(weights_diff):.10f}")
    
    # Find where the biggest differences are
    if np.max(weights_diff) > 1e-10:
        max_diff_idx = np.unravel_index(np.argmax(weights_diff), weights_diff.shape)
        print(f"\nBiggest difference at pixel {max_diff_idx}:")
        print(f"  Sequential weight: {weights_seq[max_diff_idx]:.10f}")
        print(f"  Batch weight: {weights_batch_np[max_diff_idx]:.10f}")
        print(f"  Original flow: {single_flow_np[max_diff_idx]}")
        print(f"  Flow norm: {np.sqrt(single_flow_np[max_diff_idx[0], max_diff_idx[1], 0]**2 + single_flow_np[max_diff_idx[0], max_diff_idx[1], 1]**2):.10f}")

def test_normalization_differences():
    """Test the normalization step specifically."""
    print("\n" + "="*60)
    print("Testing normalization differences...")
    
    # Create a simple test case
    test_flow = np.array([
        [[1.0, 0.5], [2.0, 1.0], [0.0, 0.0]],
        [[0.5, 0.25], [1.5, 0.75], [3.0, 1.5]]
    ], dtype=np.float32)
    
    print(f"Test flow shape: {test_flow.shape}")
    print(f"Test flow:\n{test_flow}")
    
    # Sequential processing
    print("\n=== Sequential normalization ===")
    norms_seq = np.sqrt(test_flow[..., 0]**2 + test_flow[..., 1]**2)
    print(f"Sequential norms:\n{norms_seq}")
    print(f"Sequential max norm: {np.max(norms_seq):.8f}")
    
    normalized_norms_seq = norms_seq / np.max(norms_seq)
    print(f"Sequential normalized norms:\n{normalized_norms_seq}")
    
    # Batch processing
    print("\n=== Batch normalization ===")
    test_flow_batch = mx.array(test_flow[None, :, :, :])  # Add batch dimension
    norms_batch = mx.sqrt(test_flow_batch[..., 0]**2 + test_flow_batch[..., 1]**2)
    norms_batch_np = np.array(norms_batch[0])
    print(f"Batch norms:\n{norms_batch_np}")
    print(f"Batch max norm: {float(mx.max(norms_batch)):.8f}")
    
    normalized_norms_batch = norms_batch / mx.max(norms_batch)
    normalized_norms_batch_np = np.array(normalized_norms_batch[0])
    print(f"Batch normalized norms:\n{normalized_norms_batch_np}")
    
    # Compare
    print("\n=== Normalization comparison ===")
    norms_diff = np.abs(norms_seq - norms_batch_np)
    normalized_diff = np.abs(normalized_norms_seq - normalized_norms_batch_np)
    
    print(f"Max norms difference: {np.max(norms_diff):.10f}")
    print(f"Max normalized difference: {np.max(normalized_diff):.10f}")

def test_precision_effects():
    """Test the effects of different precisions."""
    print("\n" + "="*60)
    print("Testing precision effects...")
    
    # Load real data
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        flow_float16 = data['flow'][0]  # This is float16
    
    # Convert to float32 and float64
    flow_float32 = flow_float16.astype(np.float32)
    flow_float64 = flow_float16.astype(np.float64)
    
    print(f"Original dtype: {flow_float16.dtype}")
    print(f"Float32 dtype: {flow_float32.dtype}")
    print(f"Float64 dtype: {flow_float64.dtype}")
    
    # Test with different precisions
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    # Float16 (original)
    _, weights_16 = flow_filter.filter_by_norm(flow_float16)
    
    # Float32
    _, weights_32 = flow_filter.filter_by_norm(flow_float32)
    
    # Float64
    _, weights_64 = flow_filter.filter_by_norm(flow_float64)
    
    print(f"\nWeights comparison:")
    print(f"Float16 weights min/max: {np.min(weights_16):.10f} / {np.max(weights_16):.10f}")
    print(f"Float32 weights min/max: {np.min(weights_32):.10f} / {np.max(weights_32):.10f}")
    print(f"Float64 weights min/max: {np.min(weights_64):.10f} / {np.max(weights_64):.10f}")
    
    # Compare differences
    diff_16_32 = np.abs(weights_16.astype(np.float64) - weights_32.astype(np.float64))
    diff_32_64 = np.abs(weights_32.astype(np.float64) - weights_64.astype(np.float64))
    
    print(f"\nMax difference float16 vs float32: {np.max(diff_16_32):.10f}")
    print(f"Max difference float32 vs float64: {np.max(diff_32_64):.10f}")
    
    # Test MLX conversion effects
    print(f"\n=== MLX conversion effects ===")
    flow_mx = mx.array(flow_float16)
    flow_back = np.array(flow_mx)
    
    conversion_diff = np.abs(flow_float16.astype(np.float32) - flow_back)
    print(f"Max difference after MLX conversion: {np.max(conversion_diff):.10f}")

if __name__ == "__main__":
    print("Starting filter differences analysis...")
    
    try:
        # Test 1: Direct filter comparison
        test_filter_differences()
        
        # Test 2: Normalization differences
        test_normalization_differences()
        
        # Test 3: Precision effects
        test_precision_effects()
        
        print("\n" + "="*60)
        print("Filter differences analysis completed!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc() 