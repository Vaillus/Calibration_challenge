#!/usr/bin/env python3
"""
Simple test to isolate the exact difference between sequential and batch implementations.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator
from flow_filter import FlowFilter

def test_single_pixel():
    """Test colinearity computation on a single pixel to isolate the issue."""
    print("Testing single pixel colinearity computation...")
    
    # Create a simple 3x3 flow field for testing
    flow = np.zeros((3, 3, 2))
    flow[1, 1] = [1.0, 0.5]  # Single non-zero flow vector at center
    
    # Test point
    pt = (1.0, 1.0)  # Center point
    
    # Sequential computation
    seq_estimator = VanishingPointEstimator(3, 3, use_max_distance=False, use_reoptimization=False)
    
    # Manual computation of what sequential should do
    print("\nManual sequential computation:")
    print(f"Flow at (1,1): {flow[1, 1]}")
    print(f"Point: {pt}")
    
    # Get the vector from pt to pixel (1,1)
    pixel_vector = seq_estimator._get_vector_to_pixel(pt, (1, 1))
    print(f"Pixel vector from pt to (1,1): {pixel_vector}")
    
    # Compute colinearity
    flow_vector = (flow[1, 1, 0], flow[1, 1, 1])
    colinearity = seq_estimator._compute_colinearity(flow_vector, pixel_vector)
    print(f"Colinearity: {colinearity}")
    
    # Sequential colin_score
    seq_score = seq_estimator.colin_score(flow, pt, step=1, weights=None)
    print(f"Sequential colin_score: {seq_score}")
    
    # Batch computation
    batch_estimator = ParallelVanishingPointEstimator(3, 3, use_max_distance=False, use_reoptimization=False)
    
    # Convert to batch format
    flow_batch = mx.array(flow[None, :, :, :])  # Add batch dimension
    pt_batch = mx.array([[pt[0], pt[1]]], dtype=mx.float32)
    
    # Batch colin_score using the exact replication method
    batch_score = batch_estimator._colin_score_batch_internal(flow_batch, pt_batch, weights=None, step=1)[0]
    print(f"Batch colin_score: {float(batch_score)}")
    
    print(f"Difference: {abs(seq_score - float(batch_score))}")

def test_real_flow_single_point():
    """Test with real flow data but at a single point."""
    print("\n" + "="*50)
    print("Testing with real flow data at a single point...")
    
    # Load one flow
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow_np = data['flow'][0]
    
    # Filter flows
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    filtered_flow_seq, weights_seq = flow_filter.filter_by_norm(single_flow_np)
    
    # Test point
    test_point = (582, 437)
    
    print(f"Flow shape: {filtered_flow_seq.shape}")
    print(f"Test point: {test_point}")
    
    # Sequential computation
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    seq_score = seq_estimator.colin_score(filtered_flow_seq, test_point, step=1, weights=weights_seq)
    print(f"Sequential score: {seq_score}")
    
    # Batch computation using exact replication
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    batch_score = batch_estimator._compute_colin_score_sequential_exact(
        filtered_flow_seq, test_point, weights_seq, step=1
    )
    print(f"Batch exact replication score: {batch_score}")
    
    print(f"Difference: {abs(seq_score - batch_score)}")
    
    # Let's also test the colinearity map computation
    print("\nTesting colinearity map computation...")
    seq_map = seq_estimator.compute_colinearity_map(filtered_flow_seq, test_point, step=1)
    print(f"Sequential map shape: {seq_map.shape}")
    print(f"Sequential map non-zero elements: {np.sum(seq_map != 0)}")
    print(f"Sequential map min/max: {np.min(seq_map):.6f} / {np.max(seq_map):.6f}")
    
    # Manual computation of what the batch should produce
    h, w = filtered_flow_seq.shape[:2]
    manual_map = np.zeros((h, w))
    
    for y in range(0, h, 1):
        for x in range(0, w, 1):
            flow_vector = (filtered_flow_seq[y, x, 0], filtered_flow_seq[y, x, 1])
            if flow_vector[0] == 0 and flow_vector[1] == 0:
                continue
            pixel_vector = seq_estimator._get_vector_to_pixel(test_point, (x, y))
            colinearity_value = seq_estimator._compute_colinearity(flow_vector, pixel_vector)
            manual_map[y, x] = colinearity_value
    
    print(f"Manual map non-zero elements: {np.sum(manual_map != 0)}")
    print(f"Manual map min/max: {np.min(manual_map):.6f} / {np.max(manual_map):.6f}")
    
    # Compare maps
    map_diff = np.abs(seq_map - manual_map)
    print(f"Max difference between sequential and manual maps: {np.max(map_diff)}")
    
    if np.max(map_diff) > 1e-10:
        print("WARNING: Sequential and manual maps differ!")
        # Find where they differ
        diff_indices = np.where(map_diff > 1e-10)
        print(f"Number of differing pixels: {len(diff_indices[0])}")
        if len(diff_indices[0]) > 0:
            y, x = diff_indices[0][0], diff_indices[1][0]
            print(f"First differing pixel at ({x}, {y}):")
            print(f"  Sequential: {seq_map[y, x]}")
            print(f"  Manual: {manual_map[y, x]}")
            print(f"  Flow vector: {filtered_flow_seq[y, x]}")

if __name__ == "__main__":
    print("Starting exact replication test...")
    
    try:
        # Test 1: Simple single pixel test
        test_single_pixel()
        
        # Test 2: Real flow data test
        test_real_flow_single_point()
        
        print("\n" + "="*50)
        print("Exact replication test completed!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc() 