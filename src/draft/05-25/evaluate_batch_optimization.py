#!/usr/bin/env python3
"""
Evaluation script for batch vanishing point estimation.
This script evaluates the performance of batch optimization against ground truth labels.
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import time
import matplotlib.pyplot as plt

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from flow_filter import FlowFilter

def load_ground_truth(video_id=0):
    """Load ground truth vanishing points for a video."""
    try:
        from ground_truth import read_ground_truth_pixels
        gt_pixels = read_ground_truth_pixels(video_id)[1:]  # Skip first frame
        return np.array(gt_pixels)
    except ImportError:
        print("Warning: Could not import ground_truth module")
        return None

def evaluate_parameter_sensitivity():
    """
    Test different optimization parameters to find the best configuration.
    """
    print("Evaluating parameter sensitivity...")
    
    # Initialize estimator
    pve = ParallelVanishingPointEstimator(
        frame_width=1164,
        frame_height=874,
        use_max_distance=False,
        use_reoptimization=False
    )
    
    # Initialize filter
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    # Load a subset of flows for testing
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    if not npz_path.exists():
        print(f"Error: {npz_path} not found.")
        return
    
    with np.load(npz_path) as data:
        # Take first 50 flows for parameter testing
        flows_subset = mx.array(data['flow'][:50])
    
    # Load ground truth
    gt_points = load_ground_truth(0)
    if gt_points is None:
        print("Cannot evaluate without ground truth")
        return
    
    gt_subset = gt_points[:50]
    
    # Filter flows
    filtered_flows, weights = flow_filter.filter_by_norm_batch(flows_subset)
    
    # Test different parameter combinations
    parameter_configs = [
        {"max_iterations": 20, "learning_rate": 0.3, "name": "Conservative"},
        {"max_iterations": 30, "learning_rate": 0.5, "name": "Balanced"},
        {"max_iterations": 50, "learning_rate": 0.7, "name": "Aggressive"},
        {"max_iterations": 40, "learning_rate": 1.0, "name": "High LR"},
        {"max_iterations": 60, "learning_rate": 0.2, "name": "Low LR"},
    ]
    
    results = []
    
    for config in parameter_configs:
        print(f"\nTesting {config['name']} configuration...")
        print(f"  Max iterations: {config['max_iterations']}")
        print(f"  Learning rate: {config['learning_rate']}")
        
        start_time = time.time()
        estimated_points = pve.estimate_vanishing_point_batch(
            filtered_flows,
            weights=weights,
            max_iterations=config['max_iterations'],
            learning_rate=config['learning_rate'],
            tolerance=1e-6,
            chunk_size=10
        )
        optimization_time = time.time() - start_time
        
        # Compute distances to ground truth
        distances = np.sqrt(np.sum((estimated_points - gt_subset)**2, axis=1))
        
        # Compute colinearity scores
        estimated_points_mx = mx.array(estimated_points)
        scores = pve.colin_score_batch(filtered_flows, estimated_points_mx, weights=weights, chunk_size=10)
        
        result = {
            'name': config['name'],
            'config': config,
            'mean_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'std_distance': np.std(distances),
            'mean_score': np.mean(scores),
            'time': optimization_time,
            'distances': distances,
            'scores': scores
        }
        
        results.append(result)
        
        print(f"  Mean distance to GT: {result['mean_distance']:.2f} pixels")
        print(f"  Median distance to GT: {result['median_distance']:.2f} pixels")
        print(f"  Mean colinearity score: {result['mean_score']:.4f}")
        print(f"  Optimization time: {result['time']:.2f} seconds")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['mean_distance'])
    print(f"\nBest configuration: {best_config['name']}")
    print(f"  Mean distance: {best_config['mean_distance']:.2f} pixels")
    print(f"  Mean score: {best_config['mean_score']:.4f}")
    
    return results

def evaluate_full_video(video_id=0, config=None):
    """
    Evaluate the batch optimization on a full video.
    """
    print(f"\nEvaluating full video {video_id}...")
    
    if config is None:
        config = {"max_iterations": 30, "learning_rate": 0.5}
    
    # Initialize estimator
    pve = ParallelVanishingPointEstimator(
        frame_width=1164,
        frame_height=874,
        use_max_distance=False,
        use_reoptimization=False
    )
    
    # Initialize filter
    flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
    
    # Load flows
    npz_path = Path(f"calib_challenge/flows/{video_id}_float16.npz")
    if not npz_path.exists():
        print(f"Error: {npz_path} not found.")
        return None
    
    with np.load(npz_path) as data:
        flows_data = mx.array(data['flow'])
    
    # Load ground truth
    gt_points = load_ground_truth(video_id)
    if gt_points is None:
        print("Cannot evaluate without ground truth")
        return None
    
    print(f"Processing {flows_data.shape[0]} flows...")
    
    # Process in batches
    batch_size = 100
    all_estimated_points = []
    all_distances = []
    all_scores = []
    
    total_time = 0
    
    for start_idx in range(0, flows_data.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, flows_data.shape[0])
        
        # Get batch
        flows_batch = flows_data[start_idx:end_idx]
        gt_batch = gt_points[start_idx:end_idx]
        
        # Filter flows
        filtered_flows, weights = flow_filter.filter_by_norm_batch(flows_batch)
        
        # Estimate vanishing points
        start_time = time.time()
        estimated_points = pve.estimate_vanishing_point_batch(
            filtered_flows,
            weights=weights,
            max_iterations=config['max_iterations'],
            learning_rate=config['learning_rate'],
            tolerance=1e-6,
            chunk_size=20
        )
        total_time += time.time() - start_time
        
        # Compute metrics
        distances = np.sqrt(np.sum((estimated_points - gt_batch)**2, axis=1))
        
        estimated_points_mx = mx.array(estimated_points)
        scores = pve.colin_score_batch(filtered_flows, estimated_points_mx, weights=weights, chunk_size=20)
        
        all_estimated_points.extend(estimated_points)
        all_distances.extend(distances)
        all_scores.extend(scores)
        
        if (start_idx // batch_size) % 5 == 0:
            print(f"  Processed {end_idx}/{flows_data.shape[0]} flows...")
    
    # Compute overall statistics
    all_estimated_points = np.array(all_estimated_points)
    all_distances = np.array(all_distances)
    all_scores = np.array(all_scores)
    
    results = {
        'video_id': video_id,
        'config': config,
        'estimated_points': all_estimated_points,
        'gt_points': gt_points,
        'distances': all_distances,
        'scores': all_scores,
        'total_time': total_time,
        'mean_distance': np.mean(all_distances),
        'median_distance': np.median(all_distances),
        'std_distance': np.std(all_distances),
        'mean_score': np.mean(all_scores),
        'fps': len(all_distances) / total_time
    }
    
    print(f"\nResults for video {video_id}:")
    print(f"  Mean distance to GT: {results['mean_distance']:.2f} pixels")
    print(f"  Median distance to GT: {results['median_distance']:.2f} pixels")
    print(f"  Std distance to GT: {results['std_distance']:.2f} pixels")
    print(f"  Mean colinearity score: {results['mean_score']:.4f}")
    print(f"  Total optimization time: {results['total_time']:.2f} seconds")
    print(f"  Processing speed: {results['fps']:.1f} FPS")
    
    return results

def plot_results(results):
    """
    Plot evaluation results.
    """
    if results is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Distance histogram
    axes[0, 0].hist(results['distances'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(results['mean_distance'], color='red', linestyle='--', 
                       label=f'Mean: {results["mean_distance"]:.1f}px')
    axes[0, 0].axvline(results['median_distance'], color='orange', linestyle='--', 
                       label=f'Median: {results["median_distance"]:.1f}px')
    axes[0, 0].set_xlabel('Distance to Ground Truth (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distance Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score histogram
    axes[0, 1].hist(results['scores'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 1].axvline(results['mean_score'], color='red', linestyle='--', 
                       label=f'Mean: {results["mean_score"]:.3f}')
    axes[0, 1].set_xlabel('Colinearity Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Colinearity Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distance over time
    axes[1, 0].plot(results['distances'], alpha=0.7)
    axes[1, 0].set_xlabel('Frame Number')
    axes[1, 0].set_ylabel('Distance to Ground Truth (pixels)')
    axes[1, 0].set_title('Distance Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Score over time
    axes[1, 1].plot(results['scores'], alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('Colinearity Score')
    axes[1, 1].set_title('Colinearity Score Over Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'evaluation_results_video_{results["video_id"]}.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting batch optimization evaluation...")
    
    try:
        # Step 1: Parameter sensitivity analysis
        param_results = evaluate_parameter_sensitivity()
        
        if param_results:
            # Find best configuration
            best_config = min(param_results, key=lambda x: x['mean_distance'])
            print(f"\nUsing best configuration for full evaluation: {best_config['name']}")
            
            # Step 2: Full video evaluation
            full_results = evaluate_full_video(video_id=0, config=best_config['config'])
            
            if full_results:
                # Step 3: Plot results
                plot_results(full_results)
                
                # Save results
                np.save("batch_optimization_evaluation.npy", full_results)
                print(f"\nResults saved to batch_optimization_evaluation.npy")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc() 