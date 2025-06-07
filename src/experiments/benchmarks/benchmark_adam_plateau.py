#!/usr/bin/env python3
"""
Benchmark Adam Plateau Detection Parameters

Ce script teste différentes configurations d'early stopping (plateau detection) 
pour l'optimiseur Adam afin de trouver le meilleur compromis vitesse/précision.

Utilisation:
    python benchmark_adam_plateau.py

Ou depuis Python:
    from src.experiments.benchmark_adam_plateau import benchmark_plateau_detection
    results = benchmark_plateau_detection(max_frames=50, video_id=2)
"""

import time
import numpy as np
import mlx.core as mx
from pathlib import Path
import gc

# Imports absolus
from src.utilities.ground_truth import read_ground_truth_pixels
from src.core.optimizers import optimize_batch


def benchmark_plateau_detection(max_frames=20, video_id=0):
    """
    Benchmark different early stopping configurations for optimization.
    Useful for validating optimal parameters or testing new ones.
    
    Args:
        max_frames: Number of frames to test (default: 20)
        video_id: Video ID to use for testing (default: 0)
        
    Returns:
        List of benchmark results with timing and accuracy metrics
    """
    print("=== PLATEAU DETECTION COMPARISON ===")
    print("Testing different early stopping configurations for optimization")
    print(f"Using {max_frames} frames from video {video_id}")
    print()
    
    # Load and prepare data
    npz_path = Path(f"flows/{video_id}_float16.npz")
    print(f"Loading video from: {npz_path}")
    
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return []
    
    # Load and convert float16 to float32
    print(f"Loading NPZ file...")
    with np.load(npz_path) as data:
        flows_data_f16 = mx.array(data['flow'])
    mx.eval(flows_data_f16)
    
    print("Converting to float32...")
    flows_data = flows_data_f16.astype(mx.float32)
    mx.eval(flows_data)
    
    del flows_data_f16
    gc.collect()
    
    # Limit frames for testing
    max_frames = min(max_frames, flows_data.shape[0])
    flows_data = flows_data[:max_frames]
    mx.eval(flows_data)
    print(f"Testing with {max_frames} frames")
    
    # Load ground truth
    labels = mx.array(read_ground_truth_pixels(video_id)[1:max_frames+1])
    mx.eval(labels)
    
    # Test configurations
    test_configs = [
        {"name": "No early stopping", "threshold": 0, "patience": 5},  # threshold=0 disables early stopping
        {"name": "Conservative", "threshold": 1e-7, "patience": 5},
        {"name": "Moderate", "threshold": 1e-6, "patience": 5},
        {"name": "Aggressive", "threshold": 1e-5, "patience": 5},
        {"name": "Very aggressive (OPTIMAL)", "threshold": 1e-4, "patience": 3},  # Current optimal
    ]
    
    results_summary = []
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Threshold: {config['threshold']}, Patience: {config['patience']}")
        print(f"{'='*60}")
        
        # Process in batches
        batch_size = 10
        all_predictions = []
        total_time = 0
        
        for start_idx in range(0, max_frames, batch_size):
            end_idx = min(start_idx + batch_size, max_frames)
            batch_frames = end_idx - start_idx
            
            print(f"Processing frames {start_idx} to {end_idx-1}...")
            
            flow_batch = flows_data[start_idx:end_idx]
            mx.eval(flow_batch)
            
            start_time = time.time()
            preds_batch = optimize_batch(
                flow_batch, 
                plateau_threshold=config['threshold'], 
                plateau_patience=config['patience']
            )
            mx.eval(preds_batch)
            batch_time = time.time() - start_time
            
            total_time += batch_time
            all_predictions.extend(preds_batch)
            
            print(f"  Completed in {batch_time:.3f}s ({batch_time/batch_frames:.4f}s per frame)")
            
            del flow_batch, preds_batch
            gc.collect()
        
        # Calculate results
        all_predictions = mx.stack(all_predictions, axis=0)
        mx.eval(all_predictions)
        
        # Accuracy
        distances = mx.sqrt(mx.sum(mx.square(all_predictions - labels), axis=1))
        mx.eval(distances)
        
        avg_distance = float(mx.mean(distances))
        std_distance = float(mx.std(distances))
        avg_time_per_frame = total_time / max_frames
        fps = max_frames / total_time
        
        # Store results
        result = {
            'name': config['name'],
            'threshold': config['threshold'],
            'patience': config['patience'],
            'total_time': total_time,
            'avg_time_per_frame': avg_time_per_frame,
            'fps': fps,
            'avg_distance': avg_distance,
            'std_distance': std_distance
        }
        results_summary.append(result)
        
        print(f"\n--- Results for {config['name']} ---")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per frame: {avg_time_per_frame:.4f}s")
        print(f"FPS: {fps:.2f}")
        print(f"Average distance: {avg_distance:.2f} pixels")
        print(f"Distance std dev: {std_distance:.2f} pixels")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("PLATEAU DETECTION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"{'Configuration':<25} {'Time/Frame':<12} {'FPS':<8} {'Avg Dist':<10} {'Speedup':<10}")
    print(f"{'-'*80}")
    
    baseline_time = results_summary[0]['avg_time_per_frame']  # No early stopping
    
    for result in results_summary:
        speedup = baseline_time / result['avg_time_per_frame']
        print(f"{result['name']:<25} {result['avg_time_per_frame']:<12.4f} {result['fps']:<8.2f} {result['avg_distance']:<10.2f} {speedup:<10.2f}x")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    
    # Find best speedup with minimal accuracy loss
    baseline_accuracy = results_summary[0]['avg_distance']
    best_config = None
    best_speedup = 1.0
    
    for result in results_summary[1:]:  # Skip baseline
        accuracy_loss = result['avg_distance'] - baseline_accuracy
        speedup = baseline_time / result['avg_time_per_frame']
        
        # Accept if speedup > 1.1x and accuracy loss < 2 pixels
        if speedup > best_speedup and accuracy_loss < 2.0:
            best_config = result
            best_speedup = speedup
    
    if best_config:
        print(f"✅ Best configuration: {best_config['name']}")
        print(f"   Speedup: {best_speedup:.2f}x")
        print(f"   Accuracy loss: {best_config['avg_distance'] - baseline_accuracy:.2f} pixels")
        print(f"   Recommended settings: threshold={best_config['threshold']}, patience={best_config['patience']}")
    else:
        print("⚠️  No configuration provides significant speedup without accuracy loss.")
        print("   Recommend using baseline (no early stopping) for best accuracy.")
    
    print(f"{'='*80}")
    print("✅ Plateau detection benchmarking completed!")
    
    return results_summary


if __name__ == "__main__":
    # Run benchmark when called directly
    benchmark_plateau_detection() 