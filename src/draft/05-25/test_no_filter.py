#!/usr/bin/env python3
"""
Test without any filtering to isolate the core difference between sequential and batch.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator

def test_no_filter():
    """Test with raw flow data, no filtering at all."""
    print("Testing without any filtering...")
    
    # Load one flow
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        raw_flow = data['flow'][0]  # Raw flow, no filtering
    
    print(f"Raw flow shape: {raw_flow.shape}")
    print(f"Raw flow non-zero elements: {np.sum((raw_flow[:,:,0]**2 + raw_flow[:,:,1]**2) > 0)}")
    
    # Test point
    test_point = (582, 437)
    
    print(f"Test point: {test_point}")
    
    # Sequential computation (no weights, no filtering)
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    seq_score = seq_estimator.colin_score(raw_flow, test_point, step=1, weights=None)
    print(f"Sequential score (no filter): {seq_score:.8f}")
    
    # Batch computation (no weights, no filtering)
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    batch_score = batch_estimator.colin_score(raw_flow, test_point, weights=None)
    print(f"Batch score (no filter): {batch_score:.8f}")
    
    print(f"Difference: {abs(seq_score - batch_score):.8f}")
    
    # Test gradients without filtering
    print(f"\n=== Testing gradients without filtering ===")
    epsilon = 0.1
    
    # Sequential gradients
    f_center = -seq_estimator.colin_score(raw_flow, test_point, step=1, weights=None)
    f_x_plus = -seq_estimator.colin_score(raw_flow, (test_point[0] + epsilon, test_point[1]), step=1, weights=None)
    f_y_plus = -seq_estimator.colin_score(raw_flow, (test_point[0], test_point[1] + epsilon), step=1, weights=None)
    
    seq_grad_x = (f_x_plus - f_center) / epsilon
    seq_grad_y = (f_y_plus - f_center) / epsilon
    
    print(f"Sequential gradients: ({seq_grad_x:.8f}, {seq_grad_y:.8f})")
    
    # Batch gradients
    flow_batch = mx.array(raw_flow[None, :, :, :])
    test_points = mx.array([[test_point[0], test_point[1]]], dtype=mx.float32)
    
    batch_gradients = batch_estimator._compute_gradients_batch(
        test_points,
        flow_batch,
        weights=None,
        epsilon=epsilon,
        step=1
    )
    
    batch_grad_x = float(batch_gradients[0, 0])
    batch_grad_y = float(batch_gradients[0, 1])
    
    print(f"Batch gradients: ({batch_grad_x:.8f}, {batch_grad_y:.8f})")
    print(f"Gradient differences: ({abs(seq_grad_x - batch_grad_x):.8f}, {abs(seq_grad_y - batch_grad_y):.8f})")

def test_flow_vector_access():
    """Test if we're accessing flow vectors at the same positions."""
    print("\n" + "="*60)
    print("Testing flow vector access patterns...")
    
    # Create a simple test flow where we can track access
    test_flow = np.zeros((5, 5, 2))
    # Put unique values at each position so we can see what's being accessed
    for y in range(5):
        for x in range(5):
            test_flow[y, x, 0] = y * 10 + x  # Unique identifier
            test_flow[y, x, 1] = (y * 10 + x) * 0.1
    
    print("Test flow field (x component):")
    print(test_flow[:, :, 0])
    
    test_point = (2.0, 2.0)  # Center
    
    # Sequential access pattern
    print(f"\n=== Sequential access pattern ===")
    seq_estimator = VanishingPointEstimator(5, 5, use_max_distance=False, use_reoptimization=False)
    
    # Let's manually trace what the sequential method does
    h, w = test_flow.shape[:2]
    print(f"Sequential will process pixels in this order:")
    accessed_values = []
    
    for y in range(0, h, 1):  # step=1
        for x in range(0, w, 1):
            flow_vector = (test_flow[y, x, 0], test_flow[y, x, 1])
            if flow_vector[0] == 0 and flow_vector[1] == 0:
                continue
            print(f"  Pixel ({x}, {y}): flow=({flow_vector[0]}, {flow_vector[1]})")
            accessed_values.append((x, y, flow_vector))
    
    seq_score = seq_estimator.colin_score(test_flow, test_point, step=1, weights=None)
    print(f"Sequential score: {seq_score:.6f}")
    
    # Batch access pattern
    print(f"\n=== Batch access pattern ===")
    batch_estimator = ParallelVanishingPointEstimator(5, 5, use_max_distance=False, use_reoptimization=False)
    
    # The batch method should do the same thing
    batch_score = batch_estimator.colin_score(test_flow, test_point, weights=None)
    print(f"Batch score: {batch_score:.6f}")
    
    print(f"Score difference: {abs(seq_score - batch_score):.8f}")
    
    # Let's also check a specific computation manually
    print(f"\n=== Manual verification ===")
    # Take pixel (1, 1) as example
    x, y = 1, 1
    flow_vector = (test_flow[y, x, 0], test_flow[y, x, 1])
    pixel_vector = seq_estimator._get_vector_to_pixel(test_point, (x, y))
    colinearity = seq_estimator._compute_colinearity(flow_vector, pixel_vector)
    
    print(f"At pixel ({x}, {y}):")
    print(f"  Flow vector: {flow_vector}")
    print(f"  Pixel vector (from point to pixel): {pixel_vector}")
    print(f"  Colinearity: {colinearity:.6f}")

if __name__ == "__main__":
    print("Starting no-filter test...")
    
    try:
        # Test 1: No filtering
        test_no_filter()
        
        # Test 2: Flow vector access patterns
        test_flow_vector_access()
        
        print("\n" + "="*60)
        print("No-filter test completed!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc() 