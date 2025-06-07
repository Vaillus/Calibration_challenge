"""
🧪 FILTER BENCHMARK MODULE

Framework complet pour tester et analyser les performances des configurations de filtrage.
Contient 6 fonctions principales pour optimiser les paramètres de filtrage des flows optiques.

Fonctions disponibles:
- test_multiple_thresholds() : Test de seuils de norme avec visualisation
- compare_weight_modes() : Comparaison des modes de pondération
- test_colinearity_filtering() : Test spécifique du filtrage par colinéarité  
- test_parameter_grid_3d() : Exploration exhaustive des paramètres en 3D
- test_specific_parameters() : Rapport détaillé pour paramètres spécifiques
- process_mixed_batch_comparison() : Comparaison avec/sans filtrage sur mixed batch

Usage:
    from src.experiments.filter_benchmark import test_multiple_thresholds, process_mixed_batch_comparison
    test_multiple_thresholds(num_thresholds=15)
    process_mixed_batch_comparison(config=my_config)
"""

import time
import numpy as np
import mlx.core as mx
from pathlib import Path
import gc

# Import the unified optimization function
from src.optimization_with_filtering import optimize_batch
from src.utilities.paths import get_flows_dir, get_intermediate_dir, get_outputs_dir

def test_multiple_thresholds(weight_mode='constant', num_thresholds=10, threshold_min=1e-2, threshold_max=1e2):
    """
    Test multiple threshold values and visualize the median improvement.
    
    Args:
        weight_mode: Weight mode for filtering ('constant', 'linear', 'quadratic')
        num_thresholds: Number of threshold values to test (default: 10)
        threshold_min: Minimum threshold value (default: 1e-2)
        threshold_max: Maximum threshold value (default: 1e2)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    print("🧪 TESTING MULTIPLE THRESHOLD VALUES")
    print(f"Testing {num_thresholds} threshold values between {threshold_min:.1e} and {threshold_max:.1e}")
    print(f"Weight mode: '{weight_mode}'")
    print("="*60)
    
    # ========================================
    # LOAD NO-FILTER DISTANCES ONCE (EFFICIENT)
    # ========================================
    print("📂 Loading baseline data (no filtering)...")
    
    # Create distances directory
    distances_dir = get_outputs_dir() / "distances"
    distances_dir.mkdir(exist_ok=True)
    distances_no_filter_file = distances_dir / "mixed_batch_no_filter_distances.npy"
    
    # Load mixed batch flows and labels
    flows_file = get_flows_dir() / "mixed_batch_flows.npz"
    labels_file = get_flows_dir() / "mixed_batch_labels.npy"
    
    if not flows_file.exists() or not labels_file.exists():
        print("❌ Mixed batch files not found. Run create_mixed_batch.py first.")
        return
    
    print(f"Loading flows from: {flows_file}")
    with np.load(flows_file) as data:
        flows_np = data['flows']
    flows_data = mx.array(flows_np, dtype=mx.float32)
    mx.eval(flows_data)
    
    print(f"Loading labels from: {labels_file}")
    labels_np = np.load(labels_file)
    labels_data = mx.array(labels_np, dtype=mx.float32)
    mx.eval(labels_data)
    
    total_frames = flows_data.shape[0]
    print(f"Mixed batch loaded: {total_frames} frames")
    
    # Load or compute no-filter distances ONCE
    if distances_no_filter_file.exists():
        print("✅ Loading cached no-filter distances...")
        distances_no_filter_np = np.load(distances_no_filter_file)
        distances_no_filter = mx.array(distances_no_filter_np)
    else:
        print("🔄 Computing no-filter distances (will be cached)...")
        predictions_no_filter = optimize_batch(flows_data, ground_truth_batch=None)
        mx.eval(predictions_no_filter)
        distances_no_filter = mx.sqrt(mx.sum(mx.square(predictions_no_filter - labels_data), axis=1))
        mx.eval(distances_no_filter)
        distances_no_filter_np = np.array(distances_no_filter)
        np.save(distances_no_filter_file, distances_no_filter_np)
        print(f"✅ No-filter distances cached to: {distances_no_filter_file}")
    
    # Compute no-filter stats once
    stats_no_filter = {
        'avg': float(mx.mean(distances_no_filter)),
        'median': float(np.median(distances_no_filter_np)),
        'min': float(mx.min(distances_no_filter)),
        'max': float(mx.max(distances_no_filter)),
        'std': float(mx.std(distances_no_filter))
    }
    
    print(f"✅ Baseline loaded - Median distance: {stats_no_filter['median']:.1f} pixels")
    print()
    
    # ========================================
    # TEST MULTIPLE THRESHOLDS EFFICIENTLY
    # ========================================
    
    # Generate threshold values logarithmically spaced
    thresholds = np.logspace(np.log10(threshold_min), np.log10(threshold_max), num_thresholds)
    
    results = []
    
    for i, threshold in enumerate(thresholds):
        print(f"🔄 Test {i+1}/{num_thresholds}: threshold = {threshold:.2e}")
        
        try:
            # Only compute WITH filtering part (no-filter is already loaded)
            start_time = time.time()
            
            predictions_with_filter, distances_with_filter, filtered_flows, weights = optimize_batch(
                flows_data, labels_data, min_norm_threshold=threshold, weight_mode=weight_mode
            )
            mx.eval(predictions_with_filter)
            mx.eval(distances_with_filter)
            
            time_with_filter = time.time() - start_time
            distances_with_filter_np = np.array(distances_with_filter)
            
            # Compute stats for this threshold
            stats_with_filter = {
                'avg': float(mx.mean(distances_with_filter)),
                'median': float(np.median(distances_with_filter_np)),
                'min': float(mx.min(distances_with_filter)),
                'max': float(mx.max(distances_with_filter)),
                'std': float(mx.std(distances_with_filter))
            }
            
            # Calculate improvements
            median_improvement = (stats_no_filter['median'] - stats_with_filter['median']) / stats_no_filter['median'] * 100
            avg_improvement = (stats_no_filter['avg'] - stats_with_filter['avg']) / stats_no_filter['avg'] * 100
            
            results.append({
                'threshold': threshold,
                'median_improvement': median_improvement,
                'avg_improvement': avg_improvement,
                'stats_with_filter': stats_with_filter,
                'stats_no_filter': stats_no_filter,
                'time': time_with_filter
            })
            
            print(f"   ✅ Median improvement: {median_improvement:+.1f}% ({stats_with_filter['median']:.1f}px)")
            print(f"   ✅ Average improvement: {avg_improvement:+.1f}% ({stats_with_filter['avg']:.1f}px)")
            print(f"   ⏱️  Time: {time_with_filter:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    if not results:
        print("❌ No results to visualize")
        return
    
    # Extract data for plotting
    thresholds_plot = [r['threshold'] for r in results]
    median_improvements = [r['median_improvement'] for r in results]
    avg_improvements = [r['avg_improvement'] for r in results]
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Median improvement
    plt.subplot(2, 1, 1)
    plt.semilogx(thresholds_plot, median_improvements, 'o-', linewidth=2, markersize=8, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Min Norm Threshold')
    plt.ylabel('Median Improvement (%)')
    plt.title(f'Median Distance Improvement vs Threshold (weight_mode="{weight_mode}")')
    
    # Annotate best result
    best_idx = np.argmax(median_improvements)
    best_threshold = thresholds_plot[best_idx]
    best_improvement = median_improvements[best_idx]
    plt.annotate(f'Best: {best_threshold:.2e}\n{best_improvement:+.1f}%', 
                xy=(best_threshold, best_improvement),
                xytext=(best_threshold*2, best_improvement+1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    # Plot 2: Average improvement
    plt.subplot(2, 1, 2)
    plt.semilogx(thresholds_plot, avg_improvements, 'o-', linewidth=2, markersize=8, color='green')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Min Norm Threshold')
    plt.ylabel('Average Improvement (%)')
    plt.title(f'Average Distance Improvement vs Threshold (weight_mode="{weight_mode}")')
    
    # Annotate best result for average
    best_avg_idx = np.argmax(avg_improvements)
    best_avg_threshold = thresholds_plot[best_avg_idx]
    best_avg_improvement = avg_improvements[best_avg_idx]
    plt.annotate(f'Best: {best_avg_threshold:.2e}\n{best_avg_improvement:+.1f}%', 
                xy=(best_avg_threshold, best_avg_improvement),
                xytext=(best_avg_threshold*2, best_avg_improvement+0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Save plot
    range_str = f"{threshold_min:.0e}_{threshold_max:.0e}"
    plot_filename = f"threshold_analysis_{weight_mode}_{range_str}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved as: {plot_filename}")
    plt.show()
    
    # Print summary table
    print(f"\n📋 SUMMARY TABLE (weight_mode='{weight_mode}', range={threshold_min:.1e}-{threshold_max:.1e})")
    print("="*90)
    print(f"{'Threshold':<12} {'Median Imp.':<12} {'Avg Imp.':<12} {'Median Dist.':<12} {'Avg Dist.':<12} {'Time':<8}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['threshold']:<12.2e} {r['median_improvement']:<12.1f}% {r['avg_improvement']:<12.1f}% "
              f"{r['stats_with_filter']['median']:<12.1f} {r['stats_with_filter']['avg']:<12.1f} {r['time']:<8.2f}s")
    
    # Find and highlight best results
    print(f"\n🏆 BEST RESULTS")
    print(f"Best median improvement: {best_improvement:+.1f}% at threshold {best_threshold:.2e}")
    print(f"Best average improvement: {best_avg_improvement:+.1f}% at threshold {best_avg_threshold:.2e}")
    print(f"Baseline (no filter): median={stats_no_filter['median']:.1f}px, avg={stats_no_filter['avg']:.1f}px")
    
    return results

def compare_weight_modes(threshold=2e1, weight_modes=['constant', 'linear']):
    """
    Compare different weight modes with a fixed threshold.
    
    Args:
        threshold: Fixed threshold value to use (default: 2e1)
        weight_modes: List of weight modes to compare (default: ['constant', 'linear'])
    """
    print("⚖️  COMPARING WEIGHT MODES")
    print(f"Fixed threshold: {threshold:.1e}")
    print(f"Weight modes to compare: {weight_modes}")
    print("="*60)
    
    # ========================================
    # LOAD NO-FILTER DISTANCES ONCE (EFFICIENT)
    # ========================================
    print("📂 Loading baseline data (no filtering)...")
    
    # Create distances directory
    distances_dir = get_outputs_dir() / "distances"
    distances_dir.mkdir(exist_ok=True)
    distances_no_filter_file = distances_dir / "mixed_batch_no_filter_distances.npy"
    
    # Load mixed batch flows and labels
    flows_file = get_flows_dir() / "mixed_batch_flows.npz"
    labels_file = get_flows_dir() / "mixed_batch_labels.npy"
    
    if not flows_file.exists() or not labels_file.exists():
        print("❌ Mixed batch files not found. Run create_mixed_batch.py first.")
        return
    
    print(f"Loading flows from: {flows_file}")
    with np.load(flows_file) as data:
        flows_np = data['flows']
    flows_data = mx.array(flows_np, dtype=mx.float32)
    mx.eval(flows_data)
    
    print(f"Loading labels from: {labels_file}")
    labels_np = np.load(labels_file)
    labels_data = mx.array(labels_np, dtype=mx.float32)
    mx.eval(labels_data)
    
    total_frames = flows_data.shape[0]
    print(f"Mixed batch loaded: {total_frames} frames")
    
    # Load or compute no-filter distances ONCE
    if distances_no_filter_file.exists():
        print("✅ Loading cached no-filter distances...")
        distances_no_filter_np = np.load(distances_no_filter_file)
        distances_no_filter = mx.array(distances_no_filter_np)
    else:
        print("🔄 Computing no-filter distances (will be cached)...")
        predictions_no_filter = optimize_batch(flows_data, ground_truth_batch=None)
        mx.eval(predictions_no_filter)
        distances_no_filter = mx.sqrt(mx.sum(mx.square(predictions_no_filter - labels_data), axis=1))
        mx.eval(distances_no_filter)
        distances_no_filter_np = np.array(distances_no_filter)
        np.save(distances_no_filter_file, distances_no_filter_np)
        print(f"✅ No-filter distances cached to: {distances_no_filter_file}")
    
    # Compute no-filter stats once
    stats_no_filter = {
        'avg': float(mx.mean(distances_no_filter)),
        'median': float(np.median(distances_no_filter_np)),
        'min': float(mx.min(distances_no_filter)),
        'max': float(mx.max(distances_no_filter)),
        'std': float(mx.std(distances_no_filter))
    }
    
    print(f"✅ Baseline loaded - Median distance: {stats_no_filter['median']:.1f} pixels")
    print()
    
    # ========================================
    # TEST DIFFERENT WEIGHT MODES
    # ========================================
    
    results = []
    
    for i, weight_mode in enumerate(weight_modes):
        print(f"🔄 Test {i+1}/{len(weight_modes)}: weight_mode = '{weight_mode}'")
        
        try:
            start_time = time.time()
            
            predictions_with_filter, distances_with_filter, filtered_flows, weights = optimize_batch(
                flows_data, labels_data, min_norm_threshold=threshold, weight_mode=weight_mode
            )
            mx.eval(predictions_with_filter)
            mx.eval(distances_with_filter)
            
            time_with_filter = time.time() - start_time
            distances_with_filter_np = np.array(distances_with_filter)
            
            # Compute stats for this weight mode
            stats_with_filter = {
                'avg': float(mx.mean(distances_with_filter)),
                'median': float(np.median(distances_with_filter_np)),
                'min': float(mx.min(distances_with_filter)),
                'max': float(mx.max(distances_with_filter)),
                'std': float(mx.std(distances_with_filter))
            }
            
            # Calculate improvements
            median_improvement = (stats_no_filter['median'] - stats_with_filter['median']) / stats_no_filter['median'] * 100
            avg_improvement = (stats_no_filter['avg'] - stats_with_filter['avg']) / stats_no_filter['avg'] * 100
            
            results.append({
                'weight_mode': weight_mode,
                'threshold': threshold,
                'median_improvement': median_improvement,
                'avg_improvement': avg_improvement,
                'stats_with_filter': stats_with_filter,
                'stats_no_filter': stats_no_filter,
                'time': time_with_filter
            })
            
            print(f"   ✅ Median improvement: {median_improvement:+.1f}% ({stats_with_filter['median']:.1f}px)")
            print(f"   ✅ Average improvement: {avg_improvement:+.1f}% ({stats_with_filter['avg']:.1f}px)")
            print(f"   ⏱️  Time: {time_with_filter:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
        
        print()
    
    if not results:
        print("❌ No results to compare")
        return
    
    # ========================================
    # COMPARISON ANALYSIS
    # ========================================
    
    print("📊 WEIGHT MODE COMPARISON RESULTS")
    print("="*70)
    
    # Print comparison table
    print(f"{'Weight Mode':<12} {'Median Imp.':<12} {'Avg Imp.':<12} {'Median Dist.':<12} {'Avg Dist.':<12} {'Time':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['weight_mode']:<12} {r['median_improvement']:<12.1f}% {r['avg_improvement']:<12.1f}% "
              f"{r['stats_with_filter']['median']:<12.1f} {r['stats_with_filter']['avg']:<12.1f} {r['time']:<8.2f}s")
    
    print()
    
    # Find best results
    best_median_idx = max(range(len(results)), key=lambda i: results[i]['median_improvement'])
    best_avg_idx = max(range(len(results)), key=lambda i: results[i]['avg_improvement'])
    
    best_median_result = results[best_median_idx]
    best_avg_result = results[best_avg_idx]
    
    print("🏆 BEST RESULTS")
    print(f"Best median improvement: {best_median_result['median_improvement']:+.1f}% with '{best_median_result['weight_mode']}' mode")
    print(f"Best average improvement: {best_avg_result['avg_improvement']:+.1f}% with '{best_avg_result['weight_mode']}' mode")
    print(f"Baseline (no filter): median={stats_no_filter['median']:.1f}px, avg={stats_no_filter['avg']:.1f}px")
    
    # Detailed comparison if we have exactly 2 modes
    if len(results) == 2:
        r1, r2 = results[0], results[1]
        print(f"\n🔍 DETAILED COMPARISON: '{r1['weight_mode']}' vs '{r2['weight_mode']}'")
        print("-" * 50)
        
        median_diff = r1['median_improvement'] - r2['median_improvement']
        avg_diff = r1['avg_improvement'] - r2['avg_improvement']
        time_diff = r1['time'] - r2['time']
        
        print(f"Median improvement difference: {median_diff:+.1f}% ('{r1['weight_mode']}' vs '{r2['weight_mode']}')")
        print(f"Average improvement difference: {avg_diff:+.1f}% ('{r1['weight_mode']}' vs '{r2['weight_mode']}')")
        print(f"Time difference: {time_diff:+.2f}s ('{r1['weight_mode']}' vs '{r2['weight_mode']}')")
        
        if abs(median_diff) < 1 and abs(avg_diff) < 1:
            print("⚖️  Results are very similar - both modes perform equally well")
        elif median_diff > 1 or avg_diff > 1:
            print(f"✅ '{r1['weight_mode']}' mode performs better")
        else:
            print(f"✅ '{r2['weight_mode']}' mode performs better")
    
    return results

def test_colinearity_filtering(min_colinearity_threshold=0.8, weight_mode='constant'):
    """
    Test le filtrage par colinéarité avec un score minimum élevé (0.8) et sans seuil sur la norme.
    Compare les résultats avec et sans filtrage par colinéarité.
    
    Args:
        min_colinearity_threshold: Seuil minimum de colinéarité (default: 0.8)
        weight_mode: Mode de pondération ('constant', 'linear', etc.) (default: 'constant')
    """
    print("🎯 TEST DU FILTRAGE PAR COLINÉARITÉ")
    print(f"Seuil de colinéarité minimum: {min_colinearity_threshold}")
    print(f"Mode de pondération: '{weight_mode}'")
    print("⚠️  SANS seuil sur la norme des vecteurs (min_norm_threshold=0)")
    print("="*70)
    
    # Create distances directory
    distances_dir = get_outputs_dir() / "distances"
    distances_dir.mkdir(exist_ok=True)
    
    # Check if no-filter distances already exist
    distances_no_filter_file = distances_dir / "mixed_batch_no_filter_distances.npy"
    
    # Load mixed batch flows and labels
    flows_file = get_flows_dir() / "mixed_batch_flows.npz"
    labels_file = get_flows_dir() / "mixed_batch_labels.npy"
    
    if not flows_file.exists() or not labels_file.exists():
        print("❌ Mixed batch files not found. Run create_mixed_batch.py first.")
        return
    
    print(f"Loading flows from: {flows_file}")
    with np.load(flows_file) as data:
        flows_np = data['flows']
    flows_data = mx.array(flows_np, dtype=mx.float32)
    mx.eval(flows_data)
    
    print(f"Loading labels from: {labels_file}")
    labels_np = np.load(labels_file)
    labels_data = mx.array(labels_np, dtype=mx.float32)
    mx.eval(labels_data)
    
    total_frames = flows_data.shape[0]
    print(f"Mixed batch loaded: {total_frames} frames")
    print(f"Flows shape: {flows_data.shape}")
    print(f"Labels shape: {labels_data.shape}")
    print()
    
    # ========================================
    # EVALUATION 1: WITHOUT FILTERING (CACHED)
    # ========================================
    if distances_no_filter_file.exists():
        print("🔄 EVALUATION 1: WITHOUT FILTERING (Loading cached distances)")
        distances_no_filter_np = np.load(distances_no_filter_file)
        distances_no_filter = mx.array(distances_no_filter_np)
        time_no_filter = 0  # Not measured when loading
        print(f"✅ No filtering distances loaded from cache: {distances_no_filter_file}")
    else:
        print("🔄 EVALUATION 1: WITHOUT FILTERING (Computing and caching)")
        print("Processing mixed batch WITHOUT filtering...")
        
        start_time = time.time()
        
        # Direct optimization without filtering
        predictions_no_filter = optimize_batch(flows_data, ground_truth_batch=None)
        mx.eval(predictions_no_filter)
        
        # Calculate distances to ground truth
        distances_no_filter = mx.sqrt(mx.sum(mx.square(predictions_no_filter - labels_data), axis=1))
        mx.eval(distances_no_filter)
        
        time_no_filter = time.time() - start_time
        
        # Save distances for future use
        distances_no_filter_np = np.array(distances_no_filter)
        np.save(distances_no_filter_file, distances_no_filter_np)
        print(f"✅ No filtering completed in {time_no_filter:.2f}s")
        print(f"✅ Distances cached to: {distances_no_filter_file}")
    
    print()
    
    # ========================================
    # EVALUATION 2: WITH COLINEARITY FILTERING
    # ========================================
    print("🔄 EVALUATION 2: WITH COLINEARITY FILTERING")
    print(f"Using colinearity threshold: {min_colinearity_threshold}")
    print(f"Using weight mode: '{weight_mode}'")
    print("⚠️  NO norm threshold (min_norm_threshold=0)")
    
    start_time = time.time()
    
    # Optimization with colinearity filtering (NO norm threshold)
    predictions_with_filter, distances_with_filter, filtered_flows, weights = optimize_batch(
        flows_data, 
        labels_data, 
        min_norm_threshold=0.0,  # PAS de seuil sur la norme
        weight_mode=weight_mode,
        filter_type='distance',  # Filtrage par colinéarité
        min_colinearity_threshold=min_colinearity_threshold
    )
    mx.eval(predictions_with_filter)
    mx.eval(distances_with_filter)
    
    time_with_filter = time.time() - start_time
    
    # Convert to numpy for analysis
    distances_with_filter_np = np.array(distances_with_filter)
    
    print(f"✅ Colinearity filtering completed in {time_with_filter:.2f}s")
    print()
    
    # ========================================
    # COMPARISON AND ANALYSIS
    # ========================================
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    # Basic statistics
    stats_no_filter = {
        'avg': float(mx.mean(distances_no_filter)),
        'median': float(np.median(distances_no_filter_np)),
        'min': float(mx.min(distances_no_filter)),
        'max': float(mx.max(distances_no_filter)),
        'std': float(mx.std(distances_no_filter))
    }
    
    stats_with_filter = {
        'avg': float(mx.mean(distances_with_filter)),
        'median': float(np.median(distances_with_filter_np)),
        'min': float(mx.min(distances_with_filter)),
        'max': float(mx.max(distances_with_filter)),
        'std': float(mx.std(distances_with_filter))
    }
    
    print("Distance Statistics Comparison:")
    print(f"{'Metric':<12} {'No Filter':<12} {'Colinearity':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in ['avg', 'median', 'min', 'max', 'std']:
        no_filter_val = stats_no_filter[metric]
        with_filter_val = stats_with_filter[metric]
        
        if metric in ['min']:  # For min, lower is better, but improvement calculation is tricky
            if no_filter_val > 0:
                improvement = (no_filter_val - with_filter_val) / no_filter_val * 100
            else:
                improvement = 0
        else:  # For avg, median, max, std - lower is better
            if no_filter_val > 0:
                improvement = (no_filter_val - with_filter_val) / no_filter_val * 100
            else:
                improvement = 0
        
        print(f"{metric.capitalize():<12} {no_filter_val:<12.2f} {with_filter_val:<12.2f} {improvement:<12.1f}%")
    
    print()
    
    # Analyze filtering effectiveness
    print("📈 FILTERING ANALYSIS")
    print("-" * 30)
    
    # Check how many pixels were filtered out
    original_non_zero = mx.sum(flows_data != 0)
    filtered_non_zero = mx.sum(filtered_flows != 0)
    filtered_out = original_non_zero - filtered_non_zero
    filter_percentage = float(filtered_out) / float(original_non_zero) * 100
    
    print(f"Original non-zero flow vectors: {original_non_zero}")
    print(f"After colinearity filtering: {filtered_non_zero}")
    print(f"Filtered out: {filtered_out} ({filter_percentage:.1f}%)")
    
    # Weight statistics
    print(f"\nWeight statistics:")
    print(f"Min weight: {float(mx.min(weights)):.3f}")
    print(f"Max weight: {float(mx.max(weights)):.3f}")
    print(f"Mean weight: {float(mx.mean(weights)):.3f}")
    print(f"Non-zero weights: {float(mx.sum(weights > 0))}")
    
    # Performance comparison
    if time_no_filter > 0:
        print(f"\n⏱️  PERFORMANCE COMPARISON")
        print(f"Time without filtering: {time_no_filter:.2f}s")
        print(f"Time with colinearity filtering: {time_with_filter:.2f}s")
        time_overhead = ((time_with_filter - time_no_filter) / time_no_filter) * 100
        print(f"Filtering overhead: {time_overhead:+.1f}%")
    else:
        print(f"\n⏱️  PERFORMANCE")
        print(f"Time with colinearity filtering: {time_with_filter:.2f}s")
    
    # Overall improvement summary
    overall_improvement = (stats_no_filter['avg'] - stats_with_filter['avg']) / stats_no_filter['avg'] * 100
    median_improvement = (stats_no_filter['median'] - stats_with_filter['median']) / stats_no_filter['median'] * 100
    
    print(f"\n🎯 SUMMARY")
    print("="*30)
    print(f"Colinearity threshold: {min_colinearity_threshold}")
    print(f"Weight mode: '{weight_mode}'")
    print(f"Norm threshold: 0.0 (disabled)")
    print(f"Overall average distance improvement: {overall_improvement:+.1f}%")
    print(f"Median distance improvement: {median_improvement:+.1f}%")
    print(f"Filtering effectiveness: {filter_percentage:.1f}% of vectors filtered")
    
    if overall_improvement > 5:
        print("✅ Colinearity filtering SIGNIFICATIVELY IMPROVES results!")
    elif overall_improvement > 0:
        print("✅ Colinearity filtering improves results")
    elif overall_improvement < -5:
        print("❌ Colinearity filtering WORSENS results significantly!")
    else:
        print("⚠️  Colinearity filtering has minimal impact")
    
    return {
        'no_filter': {
            'distances': distances_no_filter_np,
            'stats': stats_no_filter,
            'time': time_no_filter
        },
        'with_colinearity_filter': {
            'distances': distances_with_filter_np,
            'stats': stats_with_filter,
            'time': time_with_filter
        },
        'improvements': {
            'overall_avg': overall_improvement,
            'median': median_improvement,
            'filter_percentage': filter_percentage,
            'time_overhead': time_overhead if time_no_filter > 0 else None
        },
        'filter_params': {
            'min_colinearity_threshold': min_colinearity_threshold,
            'weight_mode': weight_mode,
            'min_norm_threshold': 0.0
        }
    }

def test_parameter_grid_3d(
    colinearity_range=(0.5, 0.95), 
    norm_range=(0.1, 100), 
    num_colinearity=10, 
    num_norm=10, 
    weight_mode='constant'
):
    """
    Test une grille de paramètres avec différents seuils de colinéarité et de norme.
    Visualise les résultats comme une surface 3D.
    
    Args:
        colinearity_range: Tuple (min, max) pour les seuils de colinéarité (default: (0.5, 0.95))
        norm_range: Tuple (min, max) pour les seuils de norme (default: (0.1, 100))
        num_colinearity: Nombre de valeurs de colinéarité à tester (default: 10)
        num_norm: Nombre de valeurs de norme à tester (default: 10)
        weight_mode: Mode de pondération (default: 'constant')
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    print("🧪 TEST DE GRILLE DE PARAMÈTRES 3D")
    print(f"Seuils de colinéarité: {colinearity_range[0]} à {colinearity_range[1]} ({num_colinearity} valeurs)")
    print(f"Seuils de norme: {norm_range[0]} à {norm_range[1]} ({num_norm} valeurs)")
    print(f"Mode de pondération: '{weight_mode}'")
    print(f"Total de tests: {num_colinearity * num_norm}")
    print("="*70)
    
    # Create parameter grids
    colinearity_values = np.linspace(colinearity_range[0], colinearity_range[1], num_colinearity)
    norm_values = np.logspace(np.log10(norm_range[0]), np.log10(norm_range[1]), num_norm)
    
    print(f"Colinearity values: {colinearity_values}")
    print(f"Norm values: {norm_values}")
    print()
    
    # ========================================
    # LOAD BASELINE DATA ONCE (EFFICIENT)
    # ========================================
    print("📂 Loading baseline data (no filtering)...")
    
    # Create distances directory
    distances_dir = get_outputs_dir() / "distances"
    distances_dir.mkdir(exist_ok=True)
    distances_no_filter_file = distances_dir / "mixed_batch_no_filter_distances.npy"
    
    # Load mixed batch flows and labels
    flows_file = get_flows_dir() / "mixed_batch_flows.npz"
    labels_file = get_flows_dir() / "mixed_batch_labels.npy"
    
    if not flows_file.exists() or not labels_file.exists():
        print("❌ Mixed batch files not found. Run create_mixed_batch.py first.")
        return
    
    print(f"Loading flows from: {flows_file}")
    with np.load(flows_file) as data:
        flows_np = data['flows']
    flows_data = mx.array(flows_np, dtype=mx.float32)
    mx.eval(flows_data)
    
    print(f"Loading labels from: {labels_file}")
    labels_np = np.load(labels_file)
    labels_data = mx.array(labels_np, dtype=mx.float32)
    mx.eval(labels_data)
    
    total_frames = flows_data.shape[0]
    print(f"Mixed batch loaded: {total_frames} frames")
    
    # Load or compute no-filter distances ONCE
    if distances_no_filter_file.exists():
        print("✅ Loading cached no-filter distances...")
        distances_no_filter_np = np.load(distances_no_filter_file)
        distances_no_filter = mx.array(distances_no_filter_np)
    else:
        print("🔄 Computing no-filter distances (will be cached)...")
        predictions_no_filter = optimize_batch(flows_data, ground_truth_batch=None)
        mx.eval(predictions_no_filter)
        distances_no_filter = mx.sqrt(mx.sum(mx.square(predictions_no_filter - labels_data), axis=1))
        mx.eval(distances_no_filter)
        distances_no_filter_np = np.array(distances_no_filter)
        np.save(distances_no_filter_file, distances_no_filter_np)
        print(f"✅ No-filter distances cached to: {distances_no_filter_file}")
    
    # Compute baseline stats once
    baseline_median = float(np.median(distances_no_filter_np))
    baseline_avg = float(mx.mean(distances_no_filter))
    
    print(f"✅ Baseline loaded - Median: {baseline_median:.1f}px, Average: {baseline_avg:.1f}px")
    print()
    
    # ========================================
    # TEST PARAMETER GRID
    # ========================================
    
    # Initialize result matrices
    median_improvements = np.zeros((num_colinearity, num_norm))
    avg_improvements = np.zeros((num_colinearity, num_norm))
    filter_percentages = np.zeros((num_colinearity, num_norm))
    processing_times = np.zeros((num_colinearity, num_norm))
    
    total_tests = num_colinearity * num_norm
    test_count = 0
    
    print("🔄 Testing parameter grid...")
    
    for i, colinearity_thresh in enumerate(colinearity_values):
        for j, norm_thresh in enumerate(norm_values):
            test_count += 1
            print(f"Test {test_count}/{total_tests}: colinearity={colinearity_thresh:.2f}, norm={norm_thresh:.2e}")
            
            try:
                start_time = time.time()
                
                # Test with both colinearity and norm filtering
                predictions_filtered, distances_filtered, filtered_flows, weights = optimize_batch(
                    flows_data, 
                    labels_data, 
                    min_norm_threshold=norm_thresh,
                    weight_mode=weight_mode,
                    filter_type='combined',  # Use COMBINED filtering (norm + colinearity)
                    min_colinearity_threshold=colinearity_thresh
                )
                mx.eval(predictions_filtered)
                mx.eval(distances_filtered)
                
                test_time = time.time() - start_time
                distances_filtered_np = np.array(distances_filtered)
                
                # Compute improvements
                filtered_median = float(np.median(distances_filtered_np))
                filtered_avg = float(mx.mean(distances_filtered))
                
                median_improvement = (baseline_median - filtered_median) / baseline_median * 100
                avg_improvement = (baseline_avg - filtered_avg) / baseline_avg * 100
                
                # Compute filtering effectiveness
                original_non_zero = mx.sum(flows_data != 0)
                filtered_non_zero = mx.sum(filtered_flows != 0)
                filter_percentage = float(original_non_zero - filtered_non_zero) / float(original_non_zero) * 100
                
                # Store results
                median_improvements[i, j] = median_improvement
                avg_improvements[i, j] = avg_improvement
                filter_percentages[i, j] = filter_percentage
                processing_times[i, j] = test_time
                
                print(f"   ✅ Median improvement: {median_improvement:+.1f}% ({filtered_median:.1f}px)")
                print(f"   ✅ Average improvement: {avg_improvement:+.1f}% ({filtered_avg:.1f}px)")
                print(f"   📊 Filtered: {filter_percentage:.1f}% of vectors")
                print(f"   ⏱️  Time: {test_time:.2f}s")
                
                # Memory cleanup
                del predictions_filtered, distances_filtered, filtered_flows, weights
                gc.collect()
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                # Fill with NaN for failed tests
                median_improvements[i, j] = np.nan
                avg_improvements[i, j] = np.nan
                filter_percentages[i, j] = np.nan
                processing_times[i, j] = np.nan
    
    print("\n✅ Parameter grid testing completed!")
    
    # ========================================
    # 3D VISUALIZATION
    # ========================================
    
    print("📊 Creating 3D visualizations...")
    
    # Create meshgrids for 3D plotting
    Norm, Colinearity = np.meshgrid(norm_values, colinearity_values)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Median Improvement
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(np.log10(Norm), Colinearity, median_improvements, 
                            cmap='RdYlGn', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Log10(Norm Threshold)')
    ax1.set_ylabel('Colinearity Threshold')
    ax1.set_zlabel('Median Improvement (%)')
    ax1.set_title('Median Distance Improvement')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot 2: Average Improvement
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(np.log10(Norm), Colinearity, avg_improvements, 
                            cmap='RdYlGn', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('Log10(Norm Threshold)')
    ax2.set_ylabel('Colinearity Threshold')
    ax2.set_zlabel('Average Improvement (%)')
    ax2.set_title('Average Distance Improvement')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot 3: Filter Percentage
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(np.log10(Norm), Colinearity, filter_percentages, 
                            cmap='viridis', alpha=0.8, edgecolor='none')
    ax3.set_xlabel('Log10(Norm Threshold)')
    ax3.set_ylabel('Colinearity Threshold')
    ax3.set_zlabel('Filtered Vectors (%)')
    ax3.set_title('Filtering Effectiveness')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # Plot 4: Processing Time
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    surf4 = ax4.plot_surface(np.log10(Norm), Colinearity, processing_times, 
                            cmap='plasma', alpha=0.8, edgecolor='none')
    ax4.set_xlabel('Log10(Norm Threshold)')
    ax4.set_ylabel('Colinearity Threshold')
    ax4.set_zlabel('Processing Time (s)')
    ax4.set_title('Processing Time')
    fig.colorbar(surf4, ax=ax4, shrink=0.5)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"parameter_grid_3d_{weight_mode}_{num_colinearity}x{num_norm}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 3D plots saved as: {plot_filename}")
    plt.show()
    
    # ========================================
    # FIND BEST PARAMETERS
    # ========================================
    
    print("\n🏆 BEST PARAMETER COMBINATIONS")
    print("="*50)
    
    # Find best median improvement
    best_median_idx = np.unravel_index(np.nanargmax(median_improvements), median_improvements.shape)
    best_median_colinearity = colinearity_values[best_median_idx[0]]
    best_median_norm = norm_values[best_median_idx[1]]
    best_median_improvement = median_improvements[best_median_idx]
    
    # Find best average improvement
    best_avg_idx = np.unravel_index(np.nanargmax(avg_improvements), avg_improvements.shape)
    best_avg_colinearity = colinearity_values[best_avg_idx[0]]
    best_avg_norm = norm_values[best_avg_idx[1]]
    best_avg_improvement = avg_improvements[best_avg_idx]
    
    print(f"Best median improvement: {best_median_improvement:.1f}%")
    print(f"  Colinearity threshold: {best_median_colinearity:.2f}")
    print(f"  Norm threshold: {best_median_norm:.2e}")
    print(f"  Filter percentage: {filter_percentages[best_median_idx]:.1f}%")
    print()
    
    print(f"Best average improvement: {best_avg_improvement:.1f}%")
    print(f"  Colinearity threshold: {best_avg_colinearity:.2f}")
    print(f"  Norm threshold: {best_avg_norm:.2e}")
    print(f"  Filter percentage: {filter_percentages[best_avg_idx]:.1f}%")
    print()
    
    # Summary statistics
    print("📈 SUMMARY STATISTICS")
    print("-" * 30)
    print(f"Median improvement range: [{np.nanmin(median_improvements):.1f}%, {np.nanmax(median_improvements):.1f}%]")
    print(f"Average improvement range: [{np.nanmin(avg_improvements):.1f}%, {np.nanmax(avg_improvements):.1f}%]")
    print(f"Filter percentage range: [{np.nanmin(filter_percentages):.1f}%, {np.nanmax(filter_percentages):.1f}%]")
    print(f"Processing time range: [{np.nanmin(processing_times):.1f}s, {np.nanmax(processing_times):.1f}s]")
    
    # Count positive improvements
    positive_median = np.sum(median_improvements > 0)
    positive_avg = np.sum(avg_improvements > 0)
    total_valid = np.sum(~np.isnan(median_improvements))
    
    print(f"\nPositive improvements: {positive_median}/{total_valid} median, {positive_avg}/{total_valid} average")
    print(f"Success rate: {positive_median/total_valid*100:.1f}% median, {positive_avg/total_valid*100:.1f}% average")
    
    return {
        'colinearity_values': colinearity_values,
        'norm_values': norm_values,
        'median_improvements': median_improvements,
        'avg_improvements': avg_improvements,
        'filter_percentages': filter_percentages,
        'processing_times': processing_times,
        'best_params': {
            'median': {
                'colinearity': best_median_colinearity,
                'norm': best_median_norm,
                'improvement': best_median_improvement
            },
            'average': {
                'colinearity': best_avg_colinearity,
                'norm': best_avg_norm,
                'improvement': best_avg_improvement
            }
        },
        'baseline': {
            'median': baseline_median,
            'avg': baseline_avg
        }
    }

def test_specific_parameters(colinearity_threshold=0.96, norm_threshold=13, weight_mode='constant'):
    """
    Test des paramètres spécifiques et génère un rapport d'amélioration détaillé.
    
    Args:
        colinearity_threshold: Seuil de colinéarité (default: 0.96)
        norm_threshold: Seuil de norme (default: 13)
        weight_mode: Mode de pondération (default: 'constant')
    """
    print("🎯 TEST DE PARAMÈTRES SPÉCIFIQUES")
    print(f"Seuil de colinéarité: {colinearity_threshold}")
    print(f"Seuil de norme: {norm_threshold}")
    print(f"Mode de pondération: '{weight_mode}'")
    print(f"Type de filtrage: COMBINÉ (norme + colinéarité)")
    print("="*60)
    
    # Create distances directory
    distances_dir = get_outputs_dir() / "distances"
    distances_dir.mkdir(exist_ok=True)
    
    # Check if no-filter distances already exist
    distances_no_filter_file = distances_dir / "mixed_batch_no_filter_distances.npy"
    
    # Load mixed batch flows and labels
    flows_file = get_flows_dir() / "mixed_batch_flows.npz"
    labels_file = get_flows_dir() / "mixed_batch_labels.npy"
    
    if not flows_file.exists() or not labels_file.exists():
        print("❌ Mixed batch files not found. Run create_mixed_batch.py first.")
        return
    
    print(f"📂 Loading flows from: {flows_file}")
    with np.load(flows_file) as data:
        flows_np = data['flows']
    flows_data = mx.array(flows_np, dtype=mx.float32)
    mx.eval(flows_data)
    
    print(f"📂 Loading labels from: {labels_file}")
    labels_np = np.load(labels_file)
    labels_data = mx.array(labels_np, dtype=mx.float32)
    mx.eval(labels_data)
    
    total_frames = flows_data.shape[0]
    print(f"✅ Mixed batch loaded: {total_frames} frames")
    print(f"   Flows shape: {flows_data.shape}")
    print(f"   Labels shape: {labels_data.shape}")
    print()
    
    # ========================================
    # BASELINE (NO FILTERING)
    # ========================================
    if distances_no_filter_file.exists():
        print("📊 BASELINE: Loading cached no-filter distances...")
        distances_no_filter_np = np.load(distances_no_filter_file)
        distances_no_filter = mx.array(distances_no_filter_np)
        time_no_filter = 0  # Not measured when loading
        print(f"✅ Baseline loaded from cache")
    else:
        print("📊 BASELINE: Computing no-filter distances...")
        start_time = time.time()
        
        predictions_no_filter = optimize_batch(flows_data, ground_truth_batch=None)
        mx.eval(predictions_no_filter)
        
        distances_no_filter = mx.sqrt(mx.sum(mx.square(predictions_no_filter - labels_data), axis=1))
        mx.eval(distances_no_filter)
        
        time_no_filter = time.time() - start_time
        
        distances_no_filter_np = np.array(distances_no_filter)
        np.save(distances_no_filter_file, distances_no_filter_np)
        print(f"✅ Baseline computed in {time_no_filter:.2f}s and cached")
    
    # Baseline statistics
    baseline_stats = {
        'avg': float(mx.mean(distances_no_filter)),
        'median': float(np.median(distances_no_filter_np)),
        'min': float(mx.min(distances_no_filter)),
        'max': float(mx.max(distances_no_filter)),
        'std': float(mx.std(distances_no_filter))
    }
    
    print(f"📈 Baseline Results:")
    print(f"   Average distance: {baseline_stats['avg']:.2f} pixels")
    print(f"   Median distance: {baseline_stats['median']:.2f} pixels")
    print(f"   Min distance: {baseline_stats['min']:.2f} pixels")
    print(f"   Max distance: {baseline_stats['max']:.2f} pixels")
    print(f"   Std distance: {baseline_stats['std']:.2f} pixels")
    print()
    
    # ========================================
    # COMBINED FILTERING TEST
    # ========================================
    print("🔄 COMBINED FILTERING: Testing specified parameters...")
    print(f"   Norm threshold: {norm_threshold}")
    print(f"   Colinearity threshold: {colinearity_threshold}")
    print(f"   Weight mode: '{weight_mode}'")
    
    start_time = time.time()
    
    # Test with combined filtering
    predictions_filtered, distances_filtered, filtered_flows, weights = optimize_batch(
        flows_data, 
        labels_data, 
        min_norm_threshold=norm_threshold,
        weight_mode=weight_mode,
        filter_type='combined',  # Combined filtering
        min_colinearity_threshold=colinearity_threshold
    )
    mx.eval(predictions_filtered)
    mx.eval(distances_filtered)
    
    time_filtered = time.time() - start_time
    
    # Convert to numpy for analysis
    distances_with_filter_np = np.array(distances_filtered)
    
    print(f"✅ Combined filtering completed in {time_filtered:.2f}s")
    print()
    
    # Filtered statistics
    filtered_stats = {
        'avg': float(mx.mean(distances_filtered)),
        'median': float(np.median(distances_with_filter_np)),
        'min': float(mx.min(distances_filtered)),
        'max': float(mx.max(distances_filtered)),
        'std': float(mx.std(distances_filtered))
    }
    
    print(f"📈 Filtered Results:")
    print(f"   Average distance: {filtered_stats['avg']:.2f} pixels")
    print(f"   Median distance: {filtered_stats['median']:.2f} pixels")
    print(f"   Min distance: {filtered_stats['min']:.2f} pixels")
    print(f"   Max distance: {filtered_stats['max']:.2f} pixels")
    print(f"   Std distance: {filtered_stats['std']:.2f} pixels")
    print()
    
    # ========================================
    # IMPROVEMENT ANALYSIS
    # ========================================
    print("📊 RAPPORT D'AMÉLIORATION")
    print("="*50)
    
    # Calculate improvements
    improvements = {}
    for metric in ['avg', 'median', 'min', 'max', 'std']:
        baseline_val = baseline_stats[metric]
        filtered_val = filtered_stats[metric]
        
        if baseline_val > 0:
            improvement = (baseline_val - filtered_val) / baseline_val * 100
        else:
            improvement = 0
        
        improvements[metric] = improvement
    
    # Display improvements table
    print(f"{'Métrique':<12} {'Baseline':<12} {'Filtré':<12} {'Amélioration':<12}")
    print("-" * 50)
    
    for metric in ['avg', 'median', 'min', 'max', 'std']:
        baseline_val = baseline_stats[metric]
        filtered_val = filtered_stats[metric]
        improvement = improvements[metric]
        
        print(f"{metric.capitalize():<12} {baseline_val:<12.2f} {filtered_val:<12.2f} {improvement:<12.1f}%")
    
    print()
    
    # Filtering effectiveness analysis
    print("📈 ANALYSE DU FILTRAGE")
    print("-" * 30)
    
    original_non_zero = mx.sum(flows_data != 0)
    filtered_non_zero = mx.sum(filtered_flows != 0)
    filtered_out = original_non_zero - filtered_non_zero
    filter_percentage = float(filtered_out) / float(original_non_zero) * 100
    
    print(f"Vecteurs de flow originaux (non-zéro): {original_non_zero}")
    print(f"Vecteurs après filtrage: {filtered_non_zero}")
    print(f"Vecteurs filtrés: {filtered_out} ({filter_percentage:.1f}%)")
    
    # Weight analysis
    print(f"\nAnalyse des poids:")
    print(f"   Poids minimum: {float(mx.min(weights)):.3f}")
    print(f"   Poids maximum: {float(mx.max(weights)):.3f}")
    print(f"   Poids moyen: {float(mx.mean(weights)):.3f}")
    print(f"   Poids non-zéro: {float(mx.sum(weights > 0))}")
    
    # Performance analysis
    if time_no_filter > 0:
        print(f"\n⏱️  ANALYSE DE PERFORMANCE")
        print(f"Temps sans filtrage: {time_no_filter:.2f}s")
        print(f"Temps avec filtrage combiné: {time_filtered:.2f}s")
        time_overhead = ((time_filtered - time_no_filter) / time_no_filter) * 100
        print(f"Surcoût temporel: {time_overhead:+.1f}%")
    else:
        print(f"\n⏱️  PERFORMANCE")
        print(f"Temps avec filtrage combiné: {time_filtered:.2f}s")
    
    # Percentile analysis
    print(f"\n📊 ANALYSE PAR PERCENTILES")
    print(f"{'Percentile':<12} {'Baseline':<12} {'Filtré':<12} {'Amélioration':<12}")
    print("-" * 50)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        baseline_val = np.percentile(distances_no_filter_np, p)
        filtered_val = np.percentile(distances_with_filter_np, p)
        
        if baseline_val > 0:
            improvement = (baseline_val - filtered_val) / baseline_val * 100
        else:
            improvement = 0
        
        print(f"{p}e{'':<9} {baseline_val:<12.2f} {filtered_val:<12.2f} {improvement:<12.1f}%")
    
    # Distance range distribution
    print(f"\n📊 DISTRIBUTION PAR PLAGES DE DISTANCE")
    ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, float('inf'))]
    
    print(f"{'Plage':<15} {'Baseline':<15} {'Filtré':<15} {'Changement':<10}")
    print("-" * 60)
    
    for low, high in ranges:
        if high == float('inf'):
            count_baseline = np.sum(distances_no_filter_np >= low)
            count_filtered = np.sum(distances_with_filter_np >= low)
            range_str = f"{low}+ pixels"
        else:
            count_baseline = np.sum((distances_no_filter_np >= low) & (distances_no_filter_np < high))
            count_filtered = np.sum((distances_with_filter_np >= low) & (distances_with_filter_np < high))
            range_str = f"{low}-{high} pixels"
        
        pct_baseline = count_baseline / total_frames * 100
        pct_filtered = count_filtered / total_frames * 100
        change = pct_filtered - pct_baseline
        
        print(f"{range_str:<15} {count_baseline:>3} ({pct_baseline:>5.1f}%) {count_filtered:>3} ({pct_filtered:>5.1f}%) {change:>+6.1f}%")
    
    # Overall summary
    overall_improvement = improvements['avg']
    median_improvement = improvements['median']
    
    print(f"\n🎯 RÉSUMÉ EXÉCUTIF")
    print("="*30)
    print(f"Paramètres testés:")
    print(f"   Seuil de colinéarité: {colinearity_threshold}")
    print(f"   Seuil de norme: {norm_threshold}")
    print(f"   Mode de pondération: '{weight_mode}'")
    print(f"   Type de filtrage: COMBINÉ")
    print()
    print(f"Résultats:")
    print(f"   Amélioration moyenne: {overall_improvement:+.1f}%")
    print(f"   Amélioration médiane: {median_improvement:+.1f}%")
    print(f"   Efficacité du filtrage: {filter_percentage:.1f}% des vecteurs filtrés")
    
    if overall_improvement > 5:
        print("✅ Le filtrage combiné AMÉLIORE SIGNIFICATIVEMENT les résultats!")
    elif overall_improvement > 0:
        print("✅ Le filtrage combiné améliore les résultats")
    elif overall_improvement < -5:
        print("❌ Le filtrage combiné DÉGRADE SIGNIFICATIVEMENT les résultats!")
    else:
        print("⚠️  Le filtrage combiné a un impact minimal")
    
    if time_no_filter > 0:
        time_overhead = ((time_filtered - time_no_filter) / time_no_filter) * 100
        print(f"Processing time overhead: {time_overhead:+.1f}%")
    
    print()
    print(f"📁 No-filter distances cached in: {distances_no_filter_file}")
    print("🔄 Filter results computed fresh each time for testing different parameters")
    
    return {
        'baseline': baseline_stats,
        'filtered': filtered_stats,
        'improvements': improvements,
        'filter_effectiveness': filter_percentage,
        'parameters': {
            'colinearity_threshold': colinearity_threshold,
            'norm_threshold': norm_threshold,
            'weight_mode': weight_mode,
            'filter_type': 'combined'
        },
        'performance': {
            'time_baseline': time_no_filter,
            'time_filtered': time_filtered,
            'time_overhead': time_overhead if time_no_filter > 0 else None
        }
    }

def process_mixed_batch_comparison(config=None):
    """
    Process the mixed batch created by create_mixed_batch.py with and without filtering.
    Compare the results and show improvement percentages.
    Save only no-filter distances to avoid recalculation when testing different filters.
    
    Args:
        config: FlowFilter configuration dictionary. If None, uses default norm filtering.
    """
    # Default configuration if none provided
    if config is None:
        config = {
            'filtering': {
                'norm': {'is_used': True, 'min_threshold': 1e-2}
            },
            'weighting': {
                'norm': {'is_used': True, 'type': 'linear'}
            }
        }
    
    print("=== MIXED BATCH COMPARISON: NO FILTERING vs WITH FILTERING ===")
    print("Loading mixed batch files created by create_mixed_batch.py")
    print("Will run both evaluations and compare results")
    print("No-filter distances will be cached for faster filter testing")
    print(f"Filter configuration: {config}")
    print()
    
    # Create distances directory
    distances_dir = get_outputs_dir() / "distances"
    distances_dir.mkdir(exist_ok=True)
    
    # Check if no-filter distances already exist
    distances_no_filter_file = distances_dir / "mixed_batch_no_filter_distances.npy"
    
    # Load mixed batch flows
    flows_file = get_flows_dir() / "mixed_batch_flows.npz"
    labels_file = get_flows_dir() / "mixed_batch_labels.npy"
    
    if not flows_file.exists():
        print(f"❌ Flows file not found: {flows_file}")
        print("Please run create_mixed_batch.py first to create the mixed batch")
        return
    
    if not labels_file.exists():
        print(f"❌ Labels file not found: {labels_file}")
        print("Please run create_mixed_batch.py first to create the mixed batch")
        return
    
    print(f"Loading flows from: {flows_file}")
    with np.load(flows_file) as data:
        flows_np = data['flows']
    flows_data = mx.array(flows_np, dtype=mx.float32)
    mx.eval(flows_data)
    
    print(f"Loading labels from: {labels_file}")
    labels_np = np.load(labels_file)
    labels_data = mx.array(labels_np, dtype=mx.float32)
    mx.eval(labels_data)
    
    total_frames = flows_data.shape[0]
    print(f"Mixed batch loaded: {total_frames} frames")
    print(f"Flows shape: {flows_data.shape}")
    print(f"Labels shape: {labels_data.shape}")
    print()
    
    # ========================================
    # EVALUATION 1: WITHOUT FILTERING (CACHED)
    # ========================================
    if distances_no_filter_file.exists():
        print("🔄 EVALUATION 1: WITHOUT FILTERING (Loading cached distances)")
        distances_no_filter_np = np.load(distances_no_filter_file)
        distances_no_filter = mx.array(distances_no_filter_np)
        time_no_filter = 0  # Not measured when loading
        print(f"✅ No filtering distances loaded from cache: {distances_no_filter_file}")
    else:
        print("🔄 EVALUATION 1: WITHOUT FILTERING (Computing and caching)")
        print("Processing mixed batch WITHOUT filtering...")
        print("Using optimal plateau detection: threshold=1e-4, patience=3")
        
        start_time = time.time()
        
        # Direct optimization without filtering
        predictions_no_filter = optimize_batch(flows_data, ground_truth_batch=None)
        mx.eval(predictions_no_filter)
        
        # Calculate distances to ground truth
        distances_no_filter = mx.sqrt(mx.sum(mx.square(predictions_no_filter - labels_data), axis=1))
        mx.eval(distances_no_filter)
        
        time_no_filter = time.time() - start_time
        
        # Save distances for future use
        distances_no_filter_np = np.array(distances_no_filter)
        np.save(distances_no_filter_file, distances_no_filter_np)
        print(f"✅ No filtering completed in {time_no_filter:.2f}s")
        print(f"✅ Distances cached to: {distances_no_filter_file}")
    
    print()
    
    # ========================================
    # EVALUATION 2: WITH FILTERING (ALWAYS FRESH)
    # ========================================
    print("🔄 EVALUATION 2: WITH FILTERING (Fresh computation)")
    print("Processing mixed batch WITH filtering...")
    print(f"Using FlowFilter with configuration: {config}")
    
    start_time = time.time()
    
    # Optimization with filtering (always compute fresh) - use new config
    predictions_with_filter, distances_with_filter, filtered_flows, weights = optimize_batch(
        flows_data, labels_data, config=config
    )
    mx.eval(predictions_with_filter)
    mx.eval(distances_with_filter)
    
    time_with_filter = time.time() - start_time
    
    # Convert to numpy for analysis
    distances_with_filter_np = np.array(distances_with_filter)
    
    print(f"✅ With filtering completed in {time_with_filter:.2f}s")
    print()
    
    # ========================================
    # COMPARISON AND ANALYSIS
    # ========================================
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    # Basic statistics
    stats_no_filter = {
        'avg': float(mx.mean(distances_no_filter)),
        'median': float(np.median(distances_no_filter_np)),
        'min': float(mx.min(distances_no_filter)),
        'max': float(mx.max(distances_no_filter)),
        'std': float(mx.std(distances_no_filter))
    }
    
    stats_with_filter = {
        'avg': float(mx.mean(distances_with_filter)),
        'median': float(np.median(distances_with_filter_np)),
        'min': float(mx.min(distances_with_filter)),
        'max': float(mx.max(distances_with_filter)),
        'std': float(mx.std(distances_with_filter))
    }
    
    print("Distance Statistics Comparison:")
    print(f"{'Metric':<12} {'No Filter':<12} {'With Filter':<12} {'Improvement':<12}")
    print("-" * 50)
    
    for metric in ['avg', 'median', 'min', 'max', 'std']:
        no_filter_val = stats_no_filter[metric]
        with_filter_val = stats_with_filter[metric]
        
        if metric in ['min']:  # For min, lower is better, but improvement calculation is tricky
            if no_filter_val > 0:
                improvement = (no_filter_val - with_filter_val) / no_filter_val * 100
            else:
                improvement = 0
        else:  # For avg, median, max, std - lower is better
            if no_filter_val > 0:
                improvement = (no_filter_val - with_filter_val) / no_filter_val * 100
            else:
                improvement = 0
        
        print(f"{metric.capitalize():<12} {no_filter_val:<12.2f} {with_filter_val:<12.2f} {improvement:<12.1f}%")
    
    print()
    
    # Percentile comparison
    print("Percentile Comparison:")
    print(f"{'Percentile':<12} {'No Filter':<12} {'With Filter':<12} {'Improvement':<12}")
    print("-" * 50)
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        no_filter_val = np.percentile(distances_no_filter_np, p)
        with_filter_val = np.percentile(distances_with_filter_np, p)
        
        if no_filter_val > 0:
            improvement = (no_filter_val - with_filter_val) / no_filter_val * 100
        else:
            improvement = 0
        
        print(f"{p}th{'':<9} {no_filter_val:<12.2f} {with_filter_val:<12.2f} {improvement:<12.1f}%")
    
    print()
    
    # Distance range distribution comparison
    print("Distance Range Distribution:")
    ranges = [(0, 10), (10, 20), (20, 50), (50, 100), (100, float('inf'))]
    
    print(f"{'Range':<15} {'No Filter':<15} {'With Filter':<15} {'Change':<10}")
    print("-" * 60)
    
    for low, high in ranges:
        if high == float('inf'):
            count_no_filter = np.sum(distances_no_filter_np >= low)
            count_with_filter = np.sum(distances_with_filter_np >= low)
            range_str = f"{low}+ pixels"
        else:
            count_no_filter = np.sum((distances_no_filter_np >= low) & (distances_no_filter_np < high))
            count_with_filter = np.sum((distances_with_filter_np >= low) & (distances_with_filter_np < high))
            range_str = f"{low}-{high} pixels"
        
        pct_no_filter = count_no_filter / total_frames * 100
        pct_with_filter = count_with_filter / total_frames * 100
        change = pct_with_filter - pct_no_filter
        
        print(f"{range_str:<15} {count_no_filter:>3} ({pct_no_filter:>5.1f}%) {count_with_filter:>3} ({pct_with_filter:>5.1f}%) {change:>+6.1f}%")
    
    print()
    
    # Performance comparison (only if times were measured)
    if time_no_filter > 0:
        print("Performance Comparison:")
        print(f"Time without filtering: {time_no_filter:.2f}s ({time_no_filter/total_frames:.4f}s per frame)")
        print(f"Time with filtering:    {time_with_filter:.2f}s ({time_with_filter/total_frames:.4f}s per frame)")
        time_overhead = ((time_with_filter - time_no_filter) / time_no_filter) * 100
        print(f"Filtering overhead:     {time_overhead:+.1f}%")
    else:
        print("Performance Comparison:")
        print(f"Time without filtering: (cached)")
        print(f"Time with filtering:    {time_with_filter:.2f}s ({time_with_filter/total_frames:.4f}s per frame)")
    print()
    
    # Overall improvement summary
    overall_improvement = (stats_no_filter['avg'] - stats_with_filter['avg']) / stats_no_filter['avg'] * 100
    median_improvement = (stats_no_filter['median'] - stats_with_filter['median']) / stats_no_filter['median'] * 100
    
    print("🎯 SUMMARY")
    print("="*30)
    print(f"Filter configuration: {config}")
    print(f"Overall average distance improvement: {overall_improvement:+.1f}%")
    print(f"Median distance improvement:          {median_improvement:+.1f}%")
    
    if overall_improvement > 0:
        print("✅ Filtering IMPROVES results!")
    elif overall_improvement < -5:
        print("❌ Filtering WORSENS results significantly!")
    else:
        print("⚠️  Filtering has minimal impact")
    
    if time_no_filter > 0:
        time_overhead = ((time_with_filter - time_no_filter) / time_no_filter) * 100
        print(f"Processing time overhead: {time_overhead:+.1f}%")
    
    print()
    print(f"📁 No-filter distances cached in: {distances_no_filter_file}")
    print("🔄 Filter results computed fresh each time for testing different parameters")
    
    return {
        'no_filter': {
            'distances': distances_no_filter_np,
            'stats': stats_no_filter,
            'time': time_no_filter
        },
        'with_filter': {
            'distances': distances_with_filter_np,
            'stats': stats_with_filter,
            'time': time_with_filter
        },
        'improvements': {
            'overall_avg': overall_improvement,
            'median': median_improvement,
            'time_overhead': time_overhead if time_no_filter > 0 else None
        },
        'filter_config': config
    }

if __name__ == "__main__":
    print("🧪 FILTER BENCHMARK MODULE")
    print("="*50)
    print("Available functions:")
    print("  - test_multiple_thresholds()")
    print("  - compare_weight_modes()")  
    print("  - test_colinearity_filtering()")
    print("  - test_parameter_grid_3d()")
    print("  - test_specific_parameters()")
    print("  - process_mixed_batch_comparison()")
    print()
    print("Example usage:")
    print("  from src.experiments.filter_benchmark import test_multiple_thresholds, process_mixed_batch_comparison")
    print("  test_multiple_thresholds(num_thresholds=15)")
    print("  process_mixed_batch_comparison(config=my_config)")
    print()
    print("💡 Each function uses the mixed batch data for consistent comparison")
    print("📊 Results include visualizations and detailed performance analysis")