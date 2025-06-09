#!/usr/bin/env python3
"""
Benchmark complet : L-BFGS-B vs Adam
Compare les performances, vitesse et précision des deux optimiseurs

NOTE: La version L-BFGS-B parallèle (MLX) a été abandonnée car l'incompatibilité
entre scipy.optimize et MLX rendait l'implémentation plus lente que l'original.
Ce benchmark compare maintenant seulement :
- L-BFGS-B original (scipy) : précision maximale
- Adam MLX : vitesse maximale avec support batch
"""

import time
import numpy as np
import mlx.core as mx
from pathlib import Path
import gc

from src.utilities.ground_truth import read_ground_truth_pixels
from src.utilities.paths import get_flows_dir
from src.core.flow_filter import FlowFilterBatch
from src.core.colinearity_optimization_parallel import ParallelVanishingPointEstimator
from src.core.colinearity_optimization import VanishingPointEstimator
from src.core.optimizers import optimize_batch as adam_optimize_batch, LBFGSOptimizer


def benchmark_single_frame(flow_data, ground_truth_point, frame_idx=0):
    """
    Benchmark détaillé sur une seule frame pour analyse approfondie.
    
    Args:
        flow_data: Single flow field of shape (h, w, 2)
        ground_truth_point: Ground truth vanishing point (x, y)
        frame_idx: Frame index for logging
        
    Returns:
        dict: Detailed results for this frame
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK FRAME {frame_idx}")
    print(f"{'='*60}")
    
    # Convert to MLX for filtering
    flow_mx = mx.array(flow_data, dtype=mx.float32)
    
    # Apply filtering - using production parameters
    filter_config = {
        'filtering': {
            'norm': {'is_used': True, 'min_threshold': 13},
            'colinearity': {'is_used': True, 'min_threshold': 0.96}
        },
        'weighting': {
            'norm': {'is_used': False},
            'colinearity': {'is_used': False}
        }
    }
    flow_filter = FlowFilterBatch(filter_config)
    filtered_flow_batch, weights_batch = flow_filter.filter_and_weight(flow_mx[None, :, :, :])
    
    # Extract single filtered flow
    filtered_flow = filtered_flow_batch[0]
    weights = weights_batch[0] if weights_batch is not None else None
    
    # Convert back to numpy for L-BFGS-B
    filtered_flow_np = np.array(filtered_flow)
    weights_np = np.array(weights) if weights is not None else None
    
    results = {}
    
    # ===== L-BFGS-B (Version non-parallèle) =====
    print(f"\n--- L-BFGS-B (Version non-parallèle) ---")
    
    seq_estimator = VanishingPointEstimator()
    optimizer = LBFGSOptimizer(max_iter=100, display_warnings=False)
    
    start_time = time.time()
    lbfgs_point = optimizer.optimize_single(seq_estimator, filtered_flow_np, weights=weights_np)
    lbfgs_time = time.time() - start_time
    
    lbfgs_score = seq_estimator.colin_score(filtered_flow_np, lbfgs_point, weights=weights_np)
    lbfgs_distance = np.sqrt((lbfgs_point[0] - ground_truth_point[0])**2 + 
                            (lbfgs_point[1] - ground_truth_point[1])**2)
    
    results['lbfgs'] = {
        'point': lbfgs_point,
        'score': float(lbfgs_score),
        'distance_to_gt': float(lbfgs_distance),
        'time': lbfgs_time,
        'method': 'L-BFGS-B (scipy)'
    }
    
    print(f"  Point: ({lbfgs_point[0]:.3f}, {lbfgs_point[1]:.3f})")
    print(f"  Score: {lbfgs_score:.6f}")
    print(f"  Distance to GT: {lbfgs_distance:.3f} pixels")
    print(f"  Time: {lbfgs_time:.4f}s")
    
    # ===== L-BFGS-B (Version parallèle) - ABANDONNÉE =====
    # Cette version a été abandonnée car l'incompatibilité scipy/MLX
    # rendait l'implémentation plus lente que l'original.
    # Voir les commentaires dans colinearity_optimization_parallel.py
    
    # ===== Adam (Version parallèle) =====
    print(f"\n--- Adam (Version parallèle) ---")
    
    # Create estimator for score calculation
    par_estimator = ParallelVanishingPointEstimator(
        flow_data.shape[1], flow_data.shape[0], 
        use_max_distance=False, use_reoptimization=False
    )
    
    start_time = time.time()
    adam_points = adam_optimize_batch(
        filtered_flow_batch, 
        plateau_threshold=0,  # Disable early stopping - run full 50 iterations
        plateau_patience=3
    )
    adam_time = time.time() - start_time
    
    adam_point = adam_points[0]
    adam_score = float(par_estimator.colin_score(filtered_flow, adam_point, weights=weights))
    adam_distance = np.sqrt((float(adam_point[0]) - ground_truth_point[0])**2 + 
                           (float(adam_point[1]) - ground_truth_point[1])**2)
    
    results['adam'] = {
        'point': (float(adam_point[0]), float(adam_point[1])),
        'score': adam_score,
        'distance_to_gt': float(adam_distance),
        'time': adam_time,
        'method': 'Adam (MLX + early stopping)'
    }
    
    print(f"  Point: ({float(adam_point[0]):.3f}, {float(adam_point[1]):.3f})")
    print(f"  Score: {adam_score:.6f}")
    print(f"  Distance to GT: {adam_distance:.3f} pixels")
    print(f"  Time: {adam_time:.4f}s")
    
    # ===== Comparaison =====
    print(f"\n--- Comparaison ---")
    
    # Meilleur score
    best_score = max(results['lbfgs']['score'], results['adam']['score'])
    best_method = 'lbfgs' if results['lbfgs']['score'] == best_score else 'adam'
    
    # Meilleure précision
    best_distance = min(results['lbfgs']['distance_to_gt'], results['adam']['distance_to_gt'])
    best_accuracy_method = 'lbfgs' if results['lbfgs']['distance_to_gt'] == best_distance else 'adam'
    
    # Vitesse
    fastest_time = min(results['lbfgs']['time'], results['adam']['time'])
    fastest_method = 'lbfgs' if results['lbfgs']['time'] == fastest_time else 'adam'
    
    print(f"  Meilleur score: {best_method.upper()} ({best_score:.6f})")
    print(f"  Meilleure précision: {best_accuracy_method.upper()} ({best_distance:.3f} pixels)")
    print(f"  Plus rapide: {fastest_method.upper()} ({fastest_time:.4f}s)")
    
    # Différences entre méthodes
    lbfgs_vs_adam_distance = abs(results['lbfgs']['distance_to_gt'] - results['adam']['distance_to_gt'])
    lbfgs_vs_adam_score = abs(results['lbfgs']['score'] - results['adam']['score'])
    
    print(f"  Différence L-BFGS-B vs Adam:")
    print(f"    Distance: {lbfgs_vs_adam_distance:.3f} pixels")
    print(f"    Score: {lbfgs_vs_adam_score:.6f}")
    print(f"    Speedup Adam: {results['lbfgs']['time'] / results['adam']['time']:.2f}x")
    
    results['comparison'] = {
        'best_score_method': best_method,
        'best_accuracy_method': best_accuracy_method,
        'fastest_method': fastest_method,
        'lbfgs_vs_adam_distance_diff': float(lbfgs_vs_adam_distance),
        'lbfgs_vs_adam_score_diff': float(lbfgs_vs_adam_score),
        'adam_speedup': float(results['lbfgs']['time'] / results['adam']['time'])
    }
    
    return results


def benchmark_multiple_frames(num_frames=10, video_id=0):
    """
    Benchmark sur plusieurs frames pour avoir des statistiques robustes.
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK MULTIPLE FRAMES")
    print(f"Video ID: {video_id}, Frames: {num_frames}")
    print(f"{'='*80}")
    
    # Load data
    npz_path = get_flows_dir() / f"{video_id}_float16.npz"
    if not npz_path.exists():
        print(f"❌ File not found: {npz_path}")
        return {}
    
    with np.load(npz_path) as data:
        flows_data = data['flow'][:num_frames]
    
    gt_pixels = read_ground_truth_pixels(video_id)[1:num_frames+1]
    
    all_results = []
    for i in range(num_frames):
        print(f"\nProcessing frame {i+1}/{num_frames}...")
        result = benchmark_single_frame(flows_data[i], gt_pixels[i], i)
        all_results.append(result)
        
        # Memory cleanup
        gc.collect()
    
    # Aggregate results  
    methods = ['lbfgs', 'adam']  # Removed lbfgs_parallel (abandoned)
    summary = {}
    
    for method in methods:
        distances = [r[method]['distance_to_gt'] for r in all_results]
        scores = [r[method]['score'] for r in all_results]
        times = [r[method]['time'] for r in all_results]
        
        summary[method] = {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'median_distance': float(np.median(distances)),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'mean_time': float(np.mean(times)),
            'total_time': float(np.sum(times)),
            'distances': distances,
            'scores': scores,
            'times': times
        }
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ STATISTIQUE ({num_frames} frames)")
    print(f"{'='*80}")
    
    for method in methods:
        data = summary[method]
        method_name = {'lbfgs': 'L-BFGS-B (original)', 
                      'adam': 'Adam (MLX)'}[method]
        
        print(f"\n{method_name}:")
        print(f"  Distance: {data['mean_distance']:.2f} ± {data['std_distance']:.2f} pixels (médiane: {data['median_distance']:.2f})")
        print(f"  Score: {data['mean_score']:.6f} ± {data['std_score']:.6f}")
        print(f"  Temps: {data['mean_time']:.4f}s par frame (total: {data['total_time']:.3f}s)")
    
    # Comparisons
    print(f"\n{'='*40}")
    print(f"COMPARAISONS")
    print(f"{'='*40}")
    
    # Accuracy comparison
    lbfgs_acc = summary['lbfgs']['mean_distance']
    adam_acc = summary['adam']['mean_distance']
    acc_diff = abs(lbfgs_acc - adam_acc)
    
    print(f"\nPrécision:")
    if adam_acc < lbfgs_acc:
        print(f"  🎯 Adam plus précis de {lbfgs_acc - adam_acc:.2f} pixels")
    elif lbfgs_acc < adam_acc:
        print(f"  🎯 L-BFGS-B plus précis de {adam_acc - lbfgs_acc:.2f} pixels")
    else:
        print(f"  🎯 Précision équivalente")
    
    # Speed comparison
    lbfgs_speed = summary['lbfgs']['mean_time']
    adam_speed = summary['adam']['mean_time']
    speedup = lbfgs_speed / adam_speed
    
    print(f"\nVitesse:")
    if speedup > 1.2:
        print(f"  ⚡ Adam {speedup:.2f}x plus rapide")
    elif speedup < 0.8:
        print(f"  ⚡ L-BFGS-B {1/speedup:.2f}x plus rapide")
    else:
        print(f"  ⚡ Vitesses comparables")
    
    # Consistency comparison
    lbfgs_consistency = summary['lbfgs']['std_distance']
    adam_consistency = summary['adam']['std_distance']
    
    print(f"\nConsistance (écart-type des distances):")
    if adam_consistency < lbfgs_consistency:
        print(f"  📊 Adam plus consistant ({adam_consistency:.2f} vs {lbfgs_consistency:.2f})")
    elif lbfgs_consistency < adam_consistency:
        print(f"  📊 L-BFGS-B plus consistant ({lbfgs_consistency:.2f} vs {adam_consistency:.2f})")
    else:
        print(f"  📊 Consistance équivalente")
    
    return {
        'individual_results': all_results,
        'summary': summary,
        'num_frames': num_frames,
        'video_id': video_id
    }


if __name__ == "__main__":
    print(f"🚀 BENCHMARK L-BFGS-B vs ADAM")
    print(f"{'='*80}")
    
    # Test rapide sur une frame
    print(f"\n1️⃣ Test rapide sur une frame...")
    npz_path = get_flows_dir() / "3_float16.npz"
    with np.load(npz_path) as data:
        flow_data = data['flow'][0]
    
    gt_pixels = read_ground_truth_pixels(3)[1]
    single_result = benchmark_single_frame(flow_data, gt_pixels, 0)
    
    # Test sur plusieurs frames
    print(f"\n2️⃣ Test sur plusieurs frames...")
    multi_result = benchmark_multiple_frames(num_frames=5, video_id=3)
    
    print(f"\n✅ Benchmark terminé !")
    print(f"\n💡 CONCLUSIONS:")
    
    if multi_result:
        summary = multi_result['summary']
        lbfgs_acc = summary['lbfgs']['mean_distance']
        adam_acc = summary['adam']['mean_distance']
        speedup = summary['lbfgs']['mean_time'] / summary['adam']['mean_time']
        
        print(f"  • Précision: {'Adam' if adam_acc < lbfgs_acc else 'L-BFGS-B'} gagne")
        print(f"  • Vitesse: Adam {speedup:.1f}x plus rapide")
        print(f"  • Recommandation: {'Adam pour production' if speedup > 1.5 and abs(adam_acc - lbfgs_acc) < 5 else 'L-BFGS-B pour précision maximale'}") 