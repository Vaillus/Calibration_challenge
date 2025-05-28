#!/usr/bin/env python3
"""
Test complet de descente de gradient : batch vs séquentiel
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import time

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator
from flow_filter import FlowFilter
from scipy.optimize import minimize

def test_full_optimization():
    """Compare une optimisation complète entre batch et séquentiel."""
    print("Test d'optimisation complète : batch vs séquentiel")
    print("="*60)
    
    # Load one flow
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        single_flow_np = data['flow'][0]
    
    print(f"Flow shape: {single_flow_np.shape}")
    
    # Filter flows
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    filtered_flow, weights = flow_filter.filter_by_norm(single_flow_np)
    
    print(f"Filtered flow non-zero elements: {np.sum((filtered_flow[:,:,0]**2 + filtered_flow[:,:,1]**2) > 0)}")
    
    # Starting point (center of image)
    start_point = (582.0, 437.0)
    print(f"Starting point: {start_point}")
    
    # ===== SEQUENTIAL OPTIMIZATION =====
    print(f"\n{'='*30} SEQUENTIAL {'='*30}")
    
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Manual gradient descent for sequential
    seq_point = np.array(start_point, dtype=np.float64)
    seq_trajectory = [seq_point.copy()]
    seq_scores = []
    
    learning_rate = 0.1
    max_iterations = 50
    tolerance = 1e-6
    
    print(f"Sequential gradient descent (LR={learning_rate}, max_iter={max_iterations})")
    
    start_time = time.time()
    for iteration in range(max_iterations):
        # Compute score and gradients
        score = -seq_estimator.colin_score(filtered_flow, tuple(seq_point), step=1, weights=weights)
        seq_scores.append(score)
        
        # Compute gradients manually
        epsilon = 0.1
        f_x_plus = -seq_estimator.colin_score(filtered_flow, (seq_point[0] + epsilon, seq_point[1]), step=1, weights=weights)
        f_y_plus = -seq_estimator.colin_score(filtered_flow, (seq_point[0], seq_point[1] + epsilon), step=1, weights=weights)
        
        grad_x = (f_x_plus - score) / epsilon
        grad_y = (f_y_plus - score) / epsilon
        
        # Check convergence
        if iteration > 0:
            point_movement = np.sqrt((seq_point[0] - seq_trajectory[-1][0])**2 + (seq_point[1] - seq_trajectory[-1][1])**2)
            if point_movement < tolerance:
                print(f"  Converged after {iteration} iterations (movement: {point_movement:.6f})")
                break
        
        # Update point with gradient scaling
        gradient_scale = 1000.0
        scaled_grad_x = grad_x * gradient_scale
        scaled_grad_y = grad_y * gradient_scale
        
        # Adaptive learning rate
        grad_magnitude = np.sqrt(scaled_grad_x**2 + scaled_grad_y**2)
        effective_lr = learning_rate / (1.0 + grad_magnitude * 0.01)
        
        seq_point[0] -= effective_lr * scaled_grad_x
        seq_point[1] -= effective_lr * scaled_grad_y
        
        # Apply constraints (keep in image bounds)
        seq_point[0] = np.clip(seq_point[0], 0, 1163)
        seq_point[1] = np.clip(seq_point[1], 0, 873)
        
        seq_trajectory.append(seq_point.copy())
        
        if iteration % 10 == 0:
            print(f"  Iter {iteration}: point=({seq_point[0]:.3f}, {seq_point[1]:.3f}), score={score:.6f}, grad=({grad_x:.6f}, {grad_y:.6f})")
    
    seq_time = time.time() - start_time
    final_seq_score = -seq_estimator.colin_score(filtered_flow, tuple(seq_point), step=1, weights=weights)
    
    print(f"Sequential final point: ({seq_point[0]:.6f}, {seq_point[1]:.6f})")
    print(f"Sequential final score: {final_seq_score:.6f}")
    print(f"Sequential time: {seq_time:.3f}s")
    print(f"Sequential iterations: {len(seq_trajectory)-1}")
    
    # ===== BATCH OPTIMIZATION =====
    print(f"\n{'='*30} BATCH {'='*30}")
    
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Use the batch optimization method
    filtered_flow_batch = mx.array(filtered_flow[None, :, :, :])
    weights_batch = mx.array(weights[None, :, :]) if weights is not None else None
    
    print(f"Batch optimization (LR={learning_rate}, max_iter={max_iterations})")
    
    start_time = time.time()
    batch_points = batch_estimator.estimate_vanishing_point_batch(
        filtered_flow_batch,
        weights=weights_batch,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        chunk_size=1
    )
    batch_time = time.time() - start_time
    
    batch_point = batch_points[0]
    final_batch_score = -batch_estimator.colin_score(filtered_flow, tuple(batch_point), weights=weights)
    
    print(f"Batch final point: ({batch_point[0]:.6f}, {batch_point[1]:.6f})")
    print(f"Batch final score: {final_batch_score:.6f}")
    print(f"Batch time: {batch_time:.3f}s")
    
    # ===== COMPARISON =====
    print(f"\n{'='*30} COMPARISON {'='*30}")
    
    point_diff = np.sqrt((seq_point[0] - batch_point[0])**2 + (seq_point[1] - batch_point[1])**2)
    score_diff = abs(final_seq_score - final_batch_score)
    
    print(f"Point difference: {point_diff:.6f} pixels")
    print(f"Score difference: {score_diff:.8f}")
    print(f"Time ratio (batch/seq): {batch_time/seq_time:.2f}x")
    
    # Distance from starting point
    seq_distance = np.sqrt((seq_point[0] - start_point[0])**2 + (seq_point[1] - start_point[1])**2)
    batch_distance = np.sqrt((batch_point[0] - start_point[0])**2 + (batch_point[1] - start_point[1])**2)
    
    print(f"Sequential moved: {seq_distance:.3f} pixels from start")
    print(f"Batch moved: {batch_distance:.3f} pixels from start")
    
    # ===== L-BFGS-B REFERENCE =====
    print(f"\n{'='*30} L-BFGS-B REFERENCE {'='*30}")
    
    # Use the original L-BFGS-B method as reference
    start_time = time.time()
    result = minimize(
        lambda point: seq_estimator.objective_function(point, filtered_flow, weights, step=1),
        start_point,
        method='L-BFGS-B',
        options={'disp': False}
    )
    lbfgs_time = time.time() - start_time
    
    lbfgs_point = result.x
    lbfgs_score = -seq_estimator.colin_score(filtered_flow, tuple(lbfgs_point), step=1, weights=weights)
    
    print(f"L-BFGS-B final point: ({lbfgs_point[0]:.6f}, {lbfgs_point[1]:.6f})")
    print(f"L-BFGS-B final score: {lbfgs_score:.6f}")
    print(f"L-BFGS-B time: {lbfgs_time:.3f}s")
    
    # Compare with L-BFGS-B
    seq_vs_lbfgs = np.sqrt((seq_point[0] - lbfgs_point[0])**2 + (seq_point[1] - lbfgs_point[1])**2)
    batch_vs_lbfgs = np.sqrt((batch_point[0] - lbfgs_point[0])**2 + (batch_point[1] - lbfgs_point[1])**2)
    
    print(f"\nDistance to L-BFGS-B reference:")
    print(f"Sequential: {seq_vs_lbfgs:.6f} pixels")
    print(f"Batch: {batch_vs_lbfgs:.6f} pixels")
    
    return {
        'seq_point': seq_point,
        'batch_point': batch_point,
        'lbfgs_point': lbfgs_point,
        'point_diff': point_diff,
        'score_diff': score_diff,
        'seq_vs_lbfgs': seq_vs_lbfgs,
        'batch_vs_lbfgs': batch_vs_lbfgs
    }

def test_multiple_flows():
    """Test sur plusieurs flows pour voir la consistance."""
    print(f"\n{'='*60}")
    print("Test sur plusieurs flows")
    print("="*60)
    
    # Load multiple flows
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        flows = data['flow'][:5]  # Test sur 5 flows
    
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    results = []
    
    for i, flow in enumerate(flows):
        print(f"\n--- Flow {i} ---")
        
        # Filter
        filtered_flow, weights = flow_filter.filter_by_norm(flow)
        
        # Starting point
        start_point = (582.0, 437.0)
        
        # Sequential
        seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
        seq_result = minimize(
            lambda point: seq_estimator.objective_function(point, filtered_flow, weights, step=1),
            start_point,
            method='L-BFGS-B',
            options={'disp': False}
        )
        seq_point = seq_result.x
        
        # Batch
        batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
        filtered_flow_batch = mx.array(filtered_flow[None, :, :, :])
        weights_batch = mx.array(weights[None, :, :]) if weights is not None else None
        
        batch_points = batch_estimator.estimate_vanishing_point_batch(
            filtered_flow_batch,
            weights=weights_batch,
            max_iterations=30,
            learning_rate=0.5,
            tolerance=1e-6,
            chunk_size=1
        )
        batch_point = batch_points[0]
        
        # Compare
        point_diff = np.sqrt((seq_point[0] - batch_point[0])**2 + (seq_point[1] - batch_point[1])**2)
        
        print(f"Sequential: ({seq_point[0]:.3f}, {seq_point[1]:.3f})")
        print(f"Batch: ({batch_point[0]:.3f}, {batch_point[1]:.3f})")
        print(f"Difference: {point_diff:.3f} pixels")
        
        results.append(point_diff)
    
    print(f"\nRésumé sur {len(results)} flows:")
    print(f"Différence moyenne: {np.mean(results):.3f} pixels")
    print(f"Différence médiane: {np.median(results):.3f} pixels")
    print(f"Différence max: {np.max(results):.3f} pixels")
    print(f"Différence min: {np.min(results):.3f} pixels")

if __name__ == "__main__":
    print("Test d'optimisation complète : batch vs séquentiel")
    
    try:
        # Test 1: Optimisation complète sur un flow
        result = test_full_optimization()
        
        # Test 2: Test sur plusieurs flows
        test_multiple_flows()
        
        print("\n" + "="*60)
        print("Test d'optimisation complète terminé !")
        
    except Exception as e:
        print(f"Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc() 