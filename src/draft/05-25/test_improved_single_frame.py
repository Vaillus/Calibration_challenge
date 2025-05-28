#!/usr/bin/env python3
"""
Test rapide de la méthode d'optimisation améliorée sur une seule frame
Compare l'ancienne et la nouvelle méthode d'optimisation batch
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import time
import sys
import os

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', '..')
sys.path.append(src_dir)

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator
from flow_filter import FlowFilter
from scipy.optimize import minimize

def test_single_frame_optimization():
    """Test rapide sur une seule frame avec différentes méthodes d'optimisation."""
    print("Test d'optimisation améliorée sur une seule frame")
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
    
    # Initialize estimator
    estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Prepare batch data
    filtered_flow_batch = mx.array(filtered_flow[None, :, :, :])
    weights_batch = mx.array(weights[None, :, :]) if weights is not None else None
    
    # ===== TEST 1: L-BFGS-B REFERENCE =====
    print(f"\n{'='*20} L-BFGS-B REFERENCE {'='*20}")
    
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
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
    
    print(f"L-BFGS-B point: ({lbfgs_point[0]:.3f}, {lbfgs_point[1]:.3f})")
    print(f"L-BFGS-B score: {lbfgs_score:.6f}")
    print(f"L-BFGS-B time: {lbfgs_time:.3f}s")
    print(f"L-BFGS-B iterations: {result.nit}")
    
    # ===== TEST 2: IMPROVED BATCH OPTIMIZATION =====
    print(f"\n{'='*20} IMPROVED BATCH {'='*20}")
    
    start_time = time.time()
    improved_points = estimator.estimate_vanishing_point_batch(
        filtered_flow_batch,
        weights=weights_batch,
        max_iterations=30,
        learning_rate=0.5,
        tolerance=1e-6,
        chunk_size=1,
        use_restarts=False  # Single run first
    )
    improved_time = time.time() - start_time
    
    improved_point = improved_points[0]
    improved_score = -estimator.colin_score(filtered_flow, tuple(improved_point), weights=weights)
    
    print(f"Improved point: ({improved_point[0]:.3f}, {improved_point[1]:.3f})")
    print(f"Improved score: {improved_score:.6f}")
    print(f"Improved time: {improved_time:.3f}s")
    
    # ===== TEST 3: IMPROVED BATCH WITH RESTARTS =====
    print(f"\n{'='*20} IMPROVED BATCH + RESTARTS {'='*20}")
    
    start_time = time.time()
    restart_points = estimator.estimate_vanishing_point_batch(
        filtered_flow_batch,
        weights=weights_batch,
        max_iterations=30,
        learning_rate=0.5,
        tolerance=1e-6,
        chunk_size=1,
        use_restarts=True,
        num_restarts=3
    )
    restart_time = time.time() - start_time
    
    restart_point = restart_points[0]
    restart_score = -estimator.colin_score(filtered_flow, tuple(restart_point), weights=weights)
    
    print(f"Restart point: ({restart_point[0]:.3f}, {restart_point[1]:.3f})")
    print(f"Restart score: {restart_score:.6f}")
    print(f"Restart time: {restart_time:.3f}s")
    
    # ===== COMPARISON =====
    print(f"\n{'='*20} COMPARISON {'='*20}")
    
    # Distances to L-BFGS-B reference
    improved_vs_lbfgs = np.sqrt((improved_point[0] - lbfgs_point[0])**2 + (improved_point[1] - lbfgs_point[1])**2)
    restart_vs_lbfgs = np.sqrt((restart_point[0] - lbfgs_point[0])**2 + (restart_point[1] - lbfgs_point[1])**2)
    
    # Score differences
    improved_score_diff = abs(improved_score - lbfgs_score)
    restart_score_diff = abs(restart_score - lbfgs_score)
    
    print(f"Distance to L-BFGS-B:")
    print(f"  Improved: {improved_vs_lbfgs:.3f} pixels")
    print(f"  Restart:  {restart_vs_lbfgs:.3f} pixels")
    
    print(f"Score difference from L-BFGS-B:")
    print(f"  Improved: {improved_score_diff:.6f}")
    print(f"  Restart:  {restart_score_diff:.6f}")
    
    print(f"Time comparison (vs L-BFGS-B):")
    print(f"  Improved: {improved_time/lbfgs_time:.2f}x")
    print(f"  Restart:  {restart_time/lbfgs_time:.2f}x")
    
    # Movement from starting point
    start_point_np = np.array(start_point)
    lbfgs_movement = np.sqrt(np.sum((lbfgs_point - start_point_np)**2))
    improved_movement = np.sqrt(np.sum((improved_point - start_point_np)**2))
    restart_movement = np.sqrt(np.sum((restart_point - start_point_np)**2))
    
    print(f"Movement from starting point:")
    print(f"  L-BFGS-B: {lbfgs_movement:.3f} pixels")
    print(f"  Improved: {improved_movement:.3f} pixels")
    print(f"  Restart:  {restart_movement:.3f} pixels")
    
    # ===== CONCLUSION =====
    print(f"\n{'='*20} CONCLUSION {'='*20}")
    
    best_method = "L-BFGS-B"
    best_score = lbfgs_score
    
    if improved_score > best_score:
        best_method = "Improved Batch"
        best_score = improved_score
    
    if restart_score > best_score:
        best_method = "Restart Batch"
        best_score = restart_score
    
    print(f"Best method: {best_method}")
    print(f"Best score: {best_score:.6f}")
    
    # Check if improvements are significant
    if improved_vs_lbfgs < 5.0:
        print("✅ Improved batch method is very close to L-BFGS-B")
    elif improved_vs_lbfgs < 20.0:
        print("⚠️  Improved batch method is reasonably close to L-BFGS-B")
    else:
        print("❌ Improved batch method differs significantly from L-BFGS-B")
    
    if restart_vs_lbfgs < 5.0:
        print("✅ Restart batch method is very close to L-BFGS-B")
    elif restart_vs_lbfgs < 20.0:
        print("⚠️  Restart batch method is reasonably close to L-BFGS-B")
    else:
        print("❌ Restart batch method differs significantly from L-BFGS-B")
    
    return {
        'lbfgs': {'point': lbfgs_point, 'score': lbfgs_score, 'time': lbfgs_time},
        'improved': {'point': improved_point, 'score': improved_score, 'time': improved_time},
        'restart': {'point': restart_point, 'score': restart_score, 'time': restart_time}
    }

if __name__ == "__main__":
    try:
        results = test_single_frame_optimization()
        print("\nTest terminé avec succès !")
        
    except Exception as e:
        print(f"Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc() 