#!/usr/bin/env python3
"""
Test rapide d'optimisation : batch vs séquentiel
Compare juste les résultats finaux de L-BFGS-B
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import time

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator
from flow_filter import FlowFilter
from scipy.optimize import minimize

def test_quick_optimization():
    """Test rapide : compare juste les résultats finaux."""
    print("Test rapide d'optimisation : batch vs séquentiel")
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
    
    # ===== SEQUENTIAL L-BFGS-B =====
    print(f"\n--- Sequential L-BFGS-B ---")
    
    seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    start_time = time.time()
    seq_result = minimize(
        lambda point: seq_estimator.objective_function(point, filtered_flow, weights, step=1),
        start_point,
        method='L-BFGS-B',
        options={'disp': False}
    )
    seq_time = time.time() - start_time
    
    seq_point = seq_result.x
    seq_score = -seq_estimator.colin_score(filtered_flow, tuple(seq_point), step=1, weights=weights)
    
    print(f"Sequential point: ({seq_point[0]:.6f}, {seq_point[1]:.6f})")
    print(f"Sequential score: {seq_score:.6f}")
    print(f"Sequential time: {seq_time:.3f}s")
    print(f"Sequential success: {seq_result.success}")
    print(f"Sequential iterations: {seq_result.nit}")
    
    # ===== BATCH OPTIMIZATION (simplified) =====
    print(f"\n--- Batch optimization ---")
    
    batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
    
    # Use same filtered flow and weights for fair comparison
    filtered_flow_batch = mx.array(filtered_flow[None, :, :, :])
    weights_batch = mx.array(weights[None, :, :]) if weights is not None else None
    
    start_time = time.time()
    batch_points = batch_estimator.estimate_vanishing_point_batch(
        filtered_flow_batch,
        weights=weights_batch,
        max_iterations=20,  # Reduced iterations
        learning_rate=0.5,
        tolerance=1e-6,
        chunk_size=1
    )
    batch_time = time.time() - start_time
    
    batch_point = batch_points[0]
    batch_score = -batch_estimator.colin_score(filtered_flow, tuple(batch_point), weights=weights)
    
    print(f"Batch point: ({batch_point[0]:.6f}, {batch_point[1]:.6f})")
    print(f"Batch score: {batch_score:.6f}")
    print(f"Batch time: {batch_time:.3f}s")
    
    # ===== COMPARISON =====
    print(f"\n--- Comparison ---")
    
    point_diff = np.sqrt((seq_point[0] - batch_point[0])**2 + (seq_point[1] - batch_point[1])**2)
    score_diff = abs(seq_score - batch_score)
    
    print(f"Point difference: {point_diff:.3f} pixels")
    print(f"Score difference: {score_diff:.6f}")
    print(f"Time ratio (batch/seq): {batch_time/seq_time:.2f}x")
    
    # Distance from starting point
    seq_distance = np.sqrt((seq_point[0] - start_point[0])**2 + (seq_point[1] - start_point[1])**2)
    batch_distance = np.sqrt((batch_point[0] - start_point[0])**2 + (batch_point[1] - start_point[1])**2)
    
    print(f"Sequential moved: {seq_distance:.3f} pixels from start")
    print(f"Batch moved: {batch_distance:.3f} pixels from start")
    
    return point_diff, score_diff

def test_multiple_flows_quick():
    """Test rapide sur plusieurs flows."""
    print(f"\n{'='*60}")
    print("Test rapide sur plusieurs flows")
    print("="*60)
    
    # Load multiple flows
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    with np.load(npz_path) as data:
        flows = data['flow'][:3]  # Juste 3 flows pour aller vite
    
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    point_diffs = []
    score_diffs = []
    
    for i, flow in enumerate(flows):
        print(f"\n--- Flow {i} ---")
        
        # Filter
        filtered_flow, weights = flow_filter.filter_by_norm(flow)
        
        # Starting point
        start_point = (582.0, 437.0)
        
        # Sequential (L-BFGS-B)
        seq_estimator = VanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
        seq_result = minimize(
            lambda point: seq_estimator.objective_function(point, filtered_flow, weights, step=1),
            start_point,
            method='L-BFGS-B',
            options={'disp': False}
        )
        seq_point = seq_result.x
        seq_score = -seq_estimator.colin_score(filtered_flow, tuple(seq_point), step=1, weights=weights)
        
        # Batch (simplified)
        batch_estimator = ParallelVanishingPointEstimator(1164, 874, use_max_distance=False, use_reoptimization=False)
        filtered_flow_batch = mx.array(filtered_flow[None, :, :, :])
        weights_batch = mx.array(weights[None, :, :]) if weights is not None else None
        
        batch_points = batch_estimator.estimate_vanishing_point_batch(
            filtered_flow_batch,
            weights=weights_batch,
            max_iterations=15,  # Encore plus réduit
            learning_rate=0.5,
            tolerance=1e-6,
            chunk_size=1
        )
        batch_point = batch_points[0]
        batch_score = -batch_estimator.colin_score(filtered_flow, tuple(batch_point), weights=weights)
        
        # Compare
        point_diff = np.sqrt((seq_point[0] - batch_point[0])**2 + (seq_point[1] - batch_point[1])**2)
        score_diff = abs(seq_score - batch_score)
        
        print(f"Sequential: ({seq_point[0]:.3f}, {seq_point[1]:.3f}), score: {seq_score:.4f}")
        print(f"Batch: ({batch_point[0]:.3f}, {batch_point[1]:.3f}), score: {batch_score:.4f}")
        print(f"Difference: {point_diff:.3f} pixels, score diff: {score_diff:.6f}")
        
        point_diffs.append(point_diff)
        score_diffs.append(score_diff)
    
    print(f"\nRésumé sur {len(point_diffs)} flows:")
    print(f"Point difference - Moyenne: {np.mean(point_diffs):.3f} pixels")
    print(f"Point difference - Max: {np.max(point_diffs):.3f} pixels")
    print(f"Score difference - Moyenne: {np.mean(score_diffs):.6f}")
    print(f"Score difference - Max: {np.max(score_diffs):.6f}")
    
    return point_diffs, score_diffs

if __name__ == "__main__":
    print("Test rapide d'optimisation : batch vs séquentiel")
    
    try:
        # Test 1: Un flow
        point_diff, score_diff = test_quick_optimization()
        
        # Test 2: Plusieurs flows
        point_diffs, score_diffs = test_multiple_flows_quick()
        
        print("\n" + "="*60)
        print("CONCLUSION:")
        if np.mean(point_diffs) < 5.0:  # Moins de 5 pixels de différence
            print("✅ Les méthodes batch et séquentielle donnent des résultats très similaires")
        elif np.mean(point_diffs) < 20.0:  # Moins de 20 pixels
            print("⚠️  Les méthodes donnent des résultats proches mais avec quelques différences")
        else:
            print("❌ Les méthodes donnent des résultats significativement différents")
        
        print(f"Différence moyenne: {np.mean(point_diffs):.3f} pixels")
        print("Test rapide terminé !")
        
    except Exception as e:
        print(f"Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc() 