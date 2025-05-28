#!/usr/bin/env python3
"""
Test script for batch vanishing point estimation.
This script tests the new batch optimization functionality with a small subset of data.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import time

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from flow_filter import FlowFilter

def test_batch_vs_sequential():
    """
    Test that batch optimization gives similar results to sequential optimization.
    """
    print("Testing batch vs sequential optimization...")
    
    # Initialize estimator
    pve = ParallelVanishingPointEstimator(
        frame_width=1164,
        frame_height=874,
        use_max_distance=False,
        use_reoptimization=False
    )
    
    # Initialize filter
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    # Load a small subset of flows for testing
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    if not npz_path.exists():
        print(f"Error: {npz_path} not found. Please ensure flows are pre-computed.")
        return
    
    with np.load(npz_path) as data:
        # Take only first 10 flows for testing
        flows_subset = mx.array(data['flow'][:10])
    
    print(f"Testing with {flows_subset.shape[0]} flows")
    
    # Filter flows
    filtered_flows, weights = flow_filter.filter_by_norm_batch(flows_subset)
    
    # Test batch optimization
    print("\nRunning batch optimization...")
    start_time = time.time()
    batch_points = pve.estimate_vanishing_point_batch(
        filtered_flows,
        weights=weights,
        max_iterations=50,
        learning_rate=0.1,
        tolerance=1e-8,
        chunk_size=5
    )
    batch_time = time.time() - start_time
    print(f"Batch optimization took {batch_time:.3f} seconds")
    
    # Test sequential optimization (for comparison)
    print("\nRunning sequential optimization...")
    start_time = time.time()
    sequential_points = []
    
    for i in range(flows_subset.shape[0]):
        # Convert single flow back to numpy for the original method
        single_flow = np.array(filtered_flows[i])
        single_weights = np.array(weights[i]) if weights is not None else None
        
        # Use the original estimate_vanishing_point method
        from colinearity_optimization import VanishingPointEstimator
        sequential_estimator = VanishingPointEstimator(
            frame_width=1164,
            frame_height=874,
            use_max_distance=False,
            use_reoptimization=False
        )
        
        point = sequential_estimator.find_vanishing_point_lbfgsb(
            single_flow, 
            weights=single_weights, 
            visualize=False
        )
        sequential_points.append(point)
    
    sequential_time = time.time() - start_time
    print(f"Sequential optimization took {sequential_time:.3f} seconds")
    
    # Compare results
    sequential_points = np.array(sequential_points)
    differences = np.sqrt(np.sum((batch_points - sequential_points)**2, axis=1))
    
    print(f"\nComparison Results:")
    print(f"Mean difference: {np.mean(differences):.2f} pixels")
    print(f"Max difference: {np.max(differences):.2f} pixels")
    print(f"Speedup: {sequential_time/batch_time:.2f}x")
    
    # Print some example points for manual inspection
    print(f"\nExample comparisons (first 5 flows):")
    for i in range(min(5, len(batch_points))):
        print(f"Flow {i}: Batch {batch_points[i]}, Sequential {sequential_points[i]}, Diff: {differences[i]:.2f}")
    
    return batch_points, sequential_points, differences

def test_optimization_convergence():
    """
    Test that the optimization actually converges and improves the objective function.
    """
    print("\n" + "="*50)
    print("Testing optimization convergence...")
    
    # Initialize estimator
    pve = ParallelVanishingPointEstimator(
        frame_width=1164,
        frame_height=874,
        use_max_distance=False,
        use_reoptimization=False
    )
    
    # Load one flow for detailed testing
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow = mx.array(data['flow'][0:1])  # Shape: (1, h, w, 2)
    
    # Filter flow
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    filtered_flow, weights = flow_filter.filter_by_norm_batch(single_flow)
    
    # Test with different starting points
    starting_points = mx.array([
        [582, 437],  # Center
        [602, 417],  # Best point from debug
        [590, 427],  # Intermediate point
    ])
    
    print(f"Testing convergence from {len(starting_points)} different starting points...")
    
    for i, start_point in enumerate(starting_points):
        print(f"\nStarting point {i+1}: {start_point}")
        
        # Expand flow to match number of starting points
        flows_expanded = mx.repeat(filtered_flow, len(starting_points), axis=0)
        weights_expanded = mx.repeat(weights, len(starting_points), axis=0) if weights is not None else None
        
        # Run optimization with this starting point
        result = pve._optimize_batch_mlx(
            flows_expanded[i:i+1],
            start_point[None, :],
            weights_expanded[i:i+1] if weights_expanded is not None else None,
            max_iterations=30,
            learning_rate=0.5,
            tolerance=1e-6
        )
        
        final_point = result[0]
        
        # Compute initial and final scores
        initial_score = pve._colin_score_batch_internal(
            flows_expanded[i:i+1], 
            start_point[None, :], 
            weights_expanded[i:i+1] if weights_expanded is not None else None
        )[0]
        
        final_score = pve._colin_score_batch_internal(
            flows_expanded[i:i+1], 
            final_point[None, :], 
            weights_expanded[i:i+1] if weights_expanded is not None else None
        )[0]
        
        print(f"  Final point: {final_point}")
        print(f"  Initial score: {float(initial_score):.4f}")
        print(f"  Final score: {float(final_score):.4f}")
        print(f"  Improvement: {float(final_score - initial_score):.4f}")

def debug_objective_function():
    """
    Debug the objective function to understand why gradients are so small.
    """
    print("\n" + "="*50)
    print("Debugging objective function...")
    
    # Initialize estimator
    pve = ParallelVanishingPointEstimator(
        frame_width=1164,
        frame_height=874,
        use_max_distance=False,
        use_reoptimization=False
    )
    
    # Load one flow for testing
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow = mx.array(data['flow'][0:1])  # Shape: (1, h, w, 2)
    
    # Filter flow
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    filtered_flow, weights = flow_filter.filter_by_norm_batch(single_flow)
    
    # Test points around the center
    center_x, center_y = 582, 437
    test_points = []
    test_scores = []
    
    # Create a grid of test points
    for dx in [-20, -10, -5, -1, 0, 1, 5, 10, 20]:
        for dy in [-20, -10, -5, -1, 0, 1, 5, 10, 20]:
            test_point = mx.array([[center_x + dx, center_y + dy]], dtype=mx.float32)
            score = pve._objective_function_batch(test_point, filtered_flow, weights)[0]
            test_points.append([center_x + dx, center_y + dy])
            test_scores.append(float(score))
            
    test_points = np.array(test_points)
    test_scores = np.array(test_scores)
    
    print(f"Score range: {np.min(test_scores):.6f} to {np.max(test_scores):.6f}")
    print(f"Score std: {np.std(test_scores):.6f}")
    print(f"Center score: {test_scores[len(test_scores)//2]:.6f}")
    
    # Find best and worst points
    best_idx = np.argmin(test_scores)  # Remember we minimize negative colinearity
    worst_idx = np.argmax(test_scores)
    
    print(f"Best point: {test_points[best_idx]} with score {test_scores[best_idx]:.6f}")
    print(f"Worst point: {test_points[worst_idx]} with score {test_scores[worst_idx]:.6f}")
    
    # Test manual gradient calculation
    center_point = mx.array([[center_x, center_y]], dtype=mx.float32)
    center_score = pve._objective_function_batch(center_point, filtered_flow, weights)[0]
    
    # Test different epsilon values
    epsilons = [0.01, 0.1, 1.0, 5.0, 10.0]
    print(f"\nTesting different epsilon values:")
    print(f"Center score: {float(center_score):.6f}")
    
    for eps in epsilons:
        # X gradient
        point_x_plus = mx.array([[center_x + eps, center_y]], dtype=mx.float32)
        score_x_plus = pve._objective_function_batch(point_x_plus, filtered_flow, weights)[0]
        grad_x = (score_x_plus - center_score) / eps
        
        # Y gradient  
        point_y_plus = mx.array([[center_x, center_y + eps]], dtype=mx.float32)
        score_y_plus = pve._objective_function_batch(point_y_plus, filtered_flow, weights)[0]
        grad_y = (score_y_plus - center_score) / eps
        
        print(f"  Epsilon {eps:4.2f}: grad_x={float(grad_x):8.6f}, grad_y={float(grad_y):8.6f}")

if __name__ == "__main__":
    print("Starting batch optimization tests...")
    
    try:
        # Test 1: Compare batch vs sequential
        batch_points, sequential_points, differences = test_batch_vs_sequential()
        
        # Test 2: Test convergence
        test_optimization_convergence()
        
        # Test 3: Debug objective function
        debug_objective_function()
        
        print("\n" + "="*50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 