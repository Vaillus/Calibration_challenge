#!/usr/bin/env python3
"""
Debug script to compare batch vs sequential optimization with identical logic.
This will help identify why there's a 43-pixel difference between the two methods.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import time

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator
from flow_filter import FlowFilter

def test_identical_logic():
    """
    Test if batch and sequential methods give identical results when using the same optimization logic.
    """
    print("Testing batch vs sequential with identical logic...")
    
    # Load one flow for detailed comparison
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow_np = data['flow'][0]  # Single flow as numpy
        single_flow_mx = mx.array(single_flow_np[None, :, :, :])  # Batch of 1 for MLX
    
    print(f"Flow shape: {single_flow_np.shape}")
    
    # Filter flows
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    # Sequential filtering
    filtered_flow_seq, weights_seq = flow_filter.filter_by_norm(single_flow_np)
    print(f"Sequential - Filtered flow shape: {filtered_flow_seq.shape}")
    print(f"Sequential - Weights shape: {weights_seq.shape if weights_seq is not None else None}")
    
    # Batch filtering
    filtered_flow_batch, weights_batch = flow_filter.filter_by_norm_batch(single_flow_mx)
    filtered_flow_batch_np = np.array(filtered_flow_batch[0])  # Extract first (and only) element
    weights_batch_np = np.array(weights_batch[0]) if weights_batch is not None else None
    
    print(f"Batch - Filtered flow shape: {filtered_flow_batch_np.shape}")
    print(f"Batch - Weights shape: {weights_batch_np.shape if weights_batch_np is not None else None}")
    
    # Compare filtered flows
    flow_diff = np.abs(filtered_flow_seq - filtered_flow_batch_np)
    print(f"Max difference in filtered flows: {np.max(flow_diff)}")
    
    if weights_seq is not None and weights_batch_np is not None:
        weights_diff = np.abs(weights_seq - weights_batch_np)
        print(f"Max difference in weights: {np.max(weights_diff)}")
    
    # Test colinearity computation at the same point with step=1
    test_point = (582, 437)  # Center point
    
    print(f"\n=== Testing with step=1 (correct value) ===")
    # Sequential colinearity with step=1
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    seq_score_step1 = seq_estimator.colin_score(filtered_flow_seq, test_point, step=1, weights=weights_seq)
    
    # Batch colinearity with step=1 using SAME weights as sequential
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    batch_score_step1 = batch_estimator.colin_score(filtered_flow_seq, test_point, weights=weights_seq)  # Use same flow and weights
    
    print(f"Colinearity scores at point {test_point} (step=1):")
    print(f"Sequential: {seq_score_step1:.6f}")
    print(f"Batch: {batch_score_step1:.6f}")
    print(f"Difference: {abs(seq_score_step1 - batch_score_step1):.6f}")
    
    # Test objective function with step=1
    seq_obj_step1 = -seq_score_step1  # Sequential uses negative for minimization
    
    # For batch objective, we need to use the same flow and weights
    flow_batch_same = mx.array(filtered_flow_seq[None, :, :, :])
    weights_batch_same = mx.array(weights_seq[None, :, :]) if weights_seq is not None else None
    
    batch_obj_step1 = batch_estimator._objective_function_batch(
        mx.array([[test_point[0], test_point[1]]], dtype=mx.float32),
        flow_batch_same,
        weights_batch_same,
        step=1
    )[0]
    
    print(f"\nObjective function values at point {test_point} (step=1):")
    print(f"Sequential: {seq_obj_step1:.6f}")
    print(f"Batch: {float(batch_obj_step1):.6f}")
    print(f"Difference: {abs(seq_obj_step1 - float(batch_obj_step1)):.6f}")

def test_gradient_computation():
    """
    Compare gradient computation between manual calculation and batch method.
    """
    print("\n" + "="*50)
    print("Testing gradient computation...")
    
    # Load one flow
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow_np = data['flow'][0]
    
    # Filter flows
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    filtered_flow_seq, weights_seq = flow_filter.filter_by_norm(single_flow_np)
    
    # Test point
    test_point = (582, 437)
    epsilon = 0.1
    
    print(f"\n=== Testing gradients with step=1 (correct value) ===")
    # Manual gradient calculation with step=1
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    f_center = -seq_estimator.colin_score(filtered_flow_seq, test_point, step=1, weights=weights_seq)
    f_x_plus = -seq_estimator.colin_score(filtered_flow_seq, (test_point[0] + epsilon, test_point[1]), step=1, weights=weights_seq)
    f_y_plus = -seq_estimator.colin_score(filtered_flow_seq, (test_point[0], test_point[1] + epsilon), step=1, weights=weights_seq)
    
    manual_grad_x = (f_x_plus - f_center) / epsilon
    manual_grad_y = (f_y_plus - f_center) / epsilon
    
    print(f"Manual gradient calculation (step=1):")
    print(f"  f_center: {f_center:.6f}")
    print(f"  f_x_plus: {f_x_plus:.6f}")
    print(f"  f_y_plus: {f_y_plus:.6f}")
    print(f"  grad_x: {manual_grad_x:.6f}")
    print(f"  grad_y: {manual_grad_y:.6f}")
    
    # Batch gradient calculation with step=1 using SAME flow and weights
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Use same flow and weights as sequential
    flow_batch_same = mx.array(filtered_flow_seq[None, :, :, :])
    weights_batch_same = mx.array(weights_seq[None, :, :]) if weights_seq is not None else None
    
    test_points = mx.array([[test_point[0], test_point[1]]], dtype=mx.float32)
    batch_gradients = batch_estimator._compute_gradients_batch(
        test_points,
        flow_batch_same,
        weights_batch_same,
        epsilon=epsilon,
        step=1
    )
    
    batch_grad_x = float(batch_gradients[0, 0])
    batch_grad_y = float(batch_gradients[0, 1])
    
    print(f"\nBatch gradient calculation (step=1):")
    print(f"  grad_x: {batch_grad_x:.6f}")
    print(f"  grad_y: {batch_grad_y:.6f}")
    
    print(f"\nGradient differences (step=1):")
    print(f"  grad_x diff: {abs(manual_grad_x - batch_grad_x):.6f}")
    print(f"  grad_y diff: {abs(manual_grad_y - batch_grad_y):.6f}")

def test_optimization_step_by_step():
    """
    Compare optimization step by step to see where divergence occurs.
    """
    print("\n" + "="*50)
    print("Testing optimization step by step...")
    
    # Load one flow
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow_np = data['flow'][0]
    
    # Filter flows
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    filtered_flow_seq, weights_seq = flow_filter.filter_by_norm(single_flow_np)
    
    # Starting point
    start_point = (582.0, 437.0)
    
    # Sequential optimization (manual implementation of what L-BFGS-B would do in first step)
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Compute initial score and gradient manually
    initial_score_seq = -seq_estimator.colin_score(filtered_flow_seq, start_point, step=1, weights=weights_seq)
    
    epsilon = 0.1
    f_x_plus = -seq_estimator.colin_score(filtered_flow_seq, (start_point[0] + epsilon, start_point[1]), step=1, weights=weights_seq)
    f_y_plus = -seq_estimator.colin_score(filtered_flow_seq, (start_point[0], start_point[1] + epsilon), step=1, weights=weights_seq)
    
    seq_grad_x = (f_x_plus - initial_score_seq) / epsilon
    seq_grad_y = (f_y_plus - initial_score_seq) / epsilon
    
    # Manual gradient descent step
    learning_rate = 0.1
    seq_new_x = start_point[0] - learning_rate * seq_grad_x * 1000  # Same scaling as batch
    seq_new_y = start_point[1] - learning_rate * seq_grad_y * 1000
    
    print(f"Sequential manual step:")
    print(f"  Initial point: {start_point}")
    print(f"  Initial score: {initial_score_seq:.6f}")
    print(f"  Gradients: ({seq_grad_x:.6f}, {seq_grad_y:.6f})")
    print(f"  New point: ({seq_new_x:.3f}, {seq_new_y:.3f})")
    
    # Batch optimization (one step) using SAME flow and weights
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Use same flow and weights as sequential
    flow_batch_same = mx.array(filtered_flow_seq[None, :, :, :])
    weights_batch_same = mx.array(weights_seq[None, :, :]) if weights_seq is not None else None
    
    test_points = mx.array([[start_point[0], start_point[1]]], dtype=mx.float32)
    initial_score_batch = batch_estimator._objective_function_batch(
        test_points,
        flow_batch_same,
        weights_batch_same,
        step=1
    )[0]
    
    batch_gradients = batch_estimator._compute_gradients_batch(
        test_points,
        flow_batch_same,
        weights_batch_same,
        epsilon=epsilon,
        step=1
    )
    
    # Manual step with same logic as batch
    scaled_gradients = batch_gradients * 1000.0  # Same scaling
    effective_lr = learning_rate / (1.0 + mx.max(mx.sqrt(mx.sum(scaled_gradients**2, axis=1))) * 0.01)
    new_points = test_points - effective_lr * scaled_gradients
    
    print(f"\nBatch step:")
    print(f"  Initial point: {start_point}")
    print(f"  Initial score: {float(initial_score_batch):.6f}")
    print(f"  Gradients: ({float(batch_gradients[0,0]):.6f}, {float(batch_gradients[0,1]):.6f})")
    print(f"  Effective LR: {float(effective_lr):.6f}")
    print(f"  New point: ({float(new_points[0,0]):.3f}, {float(new_points[0,1]):.3f})")
    
    print(f"\nComparison:")
    print(f"  Score difference: {abs(initial_score_seq - float(initial_score_batch)):.6f}")
    print(f"  Point difference: ({abs(seq_new_x - float(new_points[0,0])):.3f}, {abs(seq_new_y - float(new_points[0,1])):.3f})")

if __name__ == "__main__":
    print("Starting detailed batch vs sequential comparison...")
    
    try:
        # Test 1: Compare basic computations
        test_identical_logic()
        
        # Test 2: Compare gradient computation
        test_gradient_computation()
        
        # Test 3: Compare optimization steps
        test_optimization_step_by_step()
        
        print("\n" + "="*50)
        print("Detailed comparison completed!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc() 