import numpy as np
import mlx.core as mx
import time
from src.core.collinearity_scorer_batch import BatchCollinearityScorer
from src.core.collinearity_scorer_sample import CollinearityScorer
from src.core.flow_filter import FlowFilterSample, FlowFilterBatch
from src.utilities.load_flows import load_flows

def test_collinearity_maps(flows):
    """Test 1: Compare collinearity maps between implementations"""
    print("\n=== Test 1: Collinearity Maps Comparison ===")
    
    # Use provided flows
    test_flow = flows[0]
    ref_point = (test_flow.shape[1]//2, test_flow.shape[0]//2)
    
    print(f"Testing with flow shape: {test_flow.shape}")
    print(f"Reference point: {ref_point}")
    
    # Convert to MLX arrays
    test_flow_mx = mx.array(test_flow, dtype=mx.float32)
    ref_point_mx = mx.array(ref_point, dtype=mx.float32)
    
    # Initialize scorers
    sample_scorer = CollinearityScorer()
    batch_scorer = BatchCollinearityScorer()
    
    # Compute maps
    sample_map = sample_scorer.compute_colinearity_map(test_flow, ref_point)
    batch_map = batch_scorer.compute_colinearity_map(test_flow_mx, ref_point_mx)
    
    # Compare maps
    map_diff = sample_map - batch_map
    print("\nMap differences:")
    print(f"Max difference: {np.max(map_diff):.6f}")
    print(f"Mean difference: {np.mean(map_diff):.6f}")
    print(f"Std difference: {np.std(map_diff):.6f}")

def test_single_sample_scores(flows):
    """Test 2: Compare scoring methods for a single sample"""
    print("\n=== Test 2: Single Sample Score Comparison ===")
    
    # Use provided flows
    test_flow = flows[0]
    ref_point = (test_flow.shape[1]//2, test_flow.shape[0]//2)
    
    print(f"Testing with flow shape: {test_flow.shape}")
    print(f"Reference point: {ref_point}")
    
    # Convert to MLX arrays
    test_flow_mx = mx.array(test_flow, dtype=mx.float32)
    ref_point_mx = mx.array(ref_point, dtype=mx.float32)
    
    # Initialize scorers
    sample_scorer = CollinearityScorer()
    batch_scorer = BatchCollinearityScorer()
    
    # Compute scores
    sample_score = sample_scorer.colin_score(test_flow, ref_point)
    batch_score = batch_scorer.colin_score(test_flow_mx, ref_point_mx)
    batch_score_batch = batch_scorer.colin_score_batch(
        test_flow_mx[None, ...],  # Add batch dimension
        ref_point_mx[None, ...],  # Add batch dimension
    )[0]  # Remove batch dimension
    
    print("\nScore comparison:")
    print(f"Sample Scorer (colin_score): {sample_score:.6f}")
    print(f"Batch Scorer (colin_score): {batch_score:.6f}")
    print(f"Batch Scorer (colin_score_batch): {batch_score_batch:.6f}")
    
    # Compare scores
    print("\nScore differences:")
    print(f"Sample Scorer (colin_score) vs Batch Scorer (colin_score): {abs(sample_score - batch_score):.6f}")
    print(f"Sample Scorer (colin_score) vs Batch Scorer (colin_score_batch): {abs(sample_score - batch_score_batch):.6f}")
    print(f"Batch Scorer (colin_score) vs Batch Scorer (colin_score_batch): {abs(batch_score - batch_score_batch):.6f}")

def test_batch_scores(flows):
    """Test 3: Compare batch vs sequential processing for 50 samples"""
    print("\n=== Test 3: Batch vs Sequential Processing (50 samples) ===")
    
    # Use provided flows
    test_flows = flows[:50]
    ref_points = [(flow.shape[1]//2, flow.shape[0]//2) for flow in test_flows]
    
    print(f"Testing with {len(test_flows)} flows of shape: {test_flows[0].shape}")
    
    # Convert to MLX arrays
    test_flows_mx = mx.array(test_flows, dtype=mx.float32)
    ref_points_mx = mx.array(ref_points, dtype=mx.float32)
    
    # Initialize scorer
    batch_scorer = BatchCollinearityScorer()
    
    # Test batch processing
    print("\nTesting batch processing...")
    start_time = time.time()
    batch_scores = batch_scorer.colin_score_batch(test_flows_mx, ref_points_mx)
    batch_time = time.time() - start_time
    
    # Test sequential processing
    print("Testing sequential processing...")
    start_time = time.time()
    seq_scores = mx.array([batch_scorer.colin_score(flow, pt) for flow, pt in zip(test_flows_mx, ref_points_mx)])
    seq_time = time.time() - start_time
    
    # Compare performance
    print("\nPerformance comparison:")
    print(f"Batch processing time: {batch_time:.3f} seconds")
    print(f"Sequential processing time: {seq_time:.3f} seconds")
    print(f"Speedup factor: {seq_time/batch_time:.2f}x")
    
    # Compare scores
    score_diff = batch_scores - seq_scores
    print("\nScore differences:")
    print(f"Max difference: {mx.max(mx.abs(score_diff)):.6f}")
    print(f"Mean difference: {mx.mean(mx.abs(score_diff)):.6f}")
    print(f"Std difference: {mx.std(score_diff):.6f}")

def test_gradients(flows):
    """Test 4: Check gradient computations for all scoring methods"""
    print("\n=== Test 4: Gradient Computation Check ===")
    
    # Use provided flows
    test_flow = flows[0]
    ref_point = (test_flow.shape[1]//2, test_flow.shape[0]//2)
    
    print(f"Testing with flow shape: {test_flow.shape}")
    print(f"Reference point: {ref_point}")
    
    # Convert to MLX arrays
    test_flow_mx = mx.array(test_flow, dtype=mx.float32)
    ref_point_mx = mx.array(ref_point, dtype=mx.float32)
    
    # Initialize scorer
    batch_scorer = BatchCollinearityScorer()
    
    # Test gradient for colin_score
    print("\nTesting gradient for colin_score...")
    def score_fn(pt):
        return batch_scorer.colin_score(test_flow_mx, pt)
    
    grad_fn = mx.grad(score_fn)
    grad = grad_fn(ref_point_mx)
    print(f"Gradient (colin_score): {grad}")
    print(f"Contains NaN: {mx.isnan(grad).any()}")
    
    # Test gradient for colin_score_batch
    print("\nTesting gradient for colin_score_batch...")
    def score_fn_batch(pt):
        return batch_scorer.colin_score_batch(test_flow_mx[None, ...], pt[None, ...])[0]
    
    grad_fn_batch = mx.grad(score_fn_batch)
    grad_batch = grad_fn_batch(ref_point_mx)
    print(f"Gradient (colin_score_batch): {grad_batch}")
    print(f"Contains NaN: {mx.isnan(grad_batch).any()}")
    
    # Compare gradients
    grad_diff = grad - grad_batch
    print("\nGradient differences:")
    print(f"Max difference: {mx.max(mx.abs(grad_diff)):.6f}")
    print(f"Mean difference: {mx.mean(mx.abs(grad_diff)):.6f}")
    print(f"Std difference: {mx.std(grad_diff):.6f}")

def test_weights_impact(flows):
    """Test 5: Impact of weights on CollinearityScorer"""
    print("\n=== Test 5: Weights Impact Analysis ===")
    
    # Use provided flows
    test_flow = flows[0]
    ref_point = (test_flow.shape[1]//2, test_flow.shape[0]//2)
    
    print(f"Testing with flow shape: {test_flow.shape}")
    print(f"Reference point: {ref_point}")
    
    # Convert to MLX arrays
    test_flow_mx = mx.array(test_flow, dtype=mx.float32)
    ref_point_mx = mx.array(ref_point, dtype=mx.float32)
    
    # Initialize scorers
    sample_scorer = CollinearityScorer()
    batch_scorer = BatchCollinearityScorer()
    
    # Configuration for linear weights based on flow magnitude
    weight_config = {
        'weighting': {
            'norm': {'is_used': True, 'type': 'linear'}
        }
    }
    
    # Initialize flow filters
    flow_filter_sample = FlowFilterSample(weight_config)
    flow_filter_batch = FlowFilterBatch(weight_config)
    
    print("\nCreating weights based on flow magnitude...")
    
    # Create weights
    weights_sample = flow_filter_sample.weight(test_flow)
    weights_batch = flow_filter_batch.weight(test_flow_mx[None, ...])[0]
    
    print(f"Weights consistency check - Max difference: {np.max(np.abs(weights_sample - np.array(weights_batch))):.8f}")
    
    # Test scores without weights
    print("\nTesting scores without weights...")
    score_sample_no_weights = sample_scorer.colin_score(test_flow, ref_point)
    score_batch_no_weights = batch_scorer.colin_score(test_flow_mx, ref_point_mx)
    
    # Test scores with weights  
    print("Testing scores with weights...")
    score_sample_with_weights = sample_scorer.colin_score(test_flow, ref_point, weights=weights_sample)
    score_batch_with_weights = batch_scorer.colin_score(test_flow_mx, ref_point_mx, weights=mx.array(weights_sample, dtype=mx.float32))
    
    print("\nScore comparison:")
    print(f"Sample scorer (no weights): {score_sample_no_weights:.6f}")
    print(f"Sample scorer (with weights): {score_sample_with_weights:.6f}")
    print(f"Batch scorer (no weights): {score_batch_no_weights:.6f}")
    print(f"Batch scorer (with weights): {score_batch_with_weights:.6f}")
    
    # Calculate impact
    impact_sample = score_sample_with_weights - score_sample_no_weights
    impact_batch = score_batch_with_weights - score_batch_no_weights
    
    print(f"\nWeights impact:")
    print(f"Sample scorer: {impact_sample:.6f} ({impact_sample/score_sample_no_weights*100:.1f}%)")
    print(f"Batch scorer: {impact_batch:.6f} ({impact_batch/score_batch_no_weights*100:.1f}%)")
    
    # Test consistency
    print(f"\nConsistency check:")
    print(f"Sample vs Batch (no weights): {abs(score_sample_no_weights - score_batch_no_weights):.8f}")
    print(f"Sample vs Batch (with weights): {abs(score_sample_with_weights - score_batch_with_weights):.8f}")

def test_different_weight_types(flows):
    """Test 6: Different weight types impact"""
    print("\n=== Test 6: Different Weight Types Impact ===")
    
    # Use provided flows
    test_flow = flows[0]
    ref_point = (test_flow.shape[1]//2, test_flow.shape[0]//2)
    
    # Convert to MLX arrays
    test_flow_mx = mx.array(test_flow, dtype=mx.float32)
    ref_point_mx = mx.array(ref_point, dtype=mx.float32)
    
    # Initialize scorer
    batch_scorer = BatchCollinearityScorer()
    
    # Reference score without weights
    score_no_weights = batch_scorer.colin_score(test_flow_mx, ref_point_mx)
    
    # Test different weight types
    weight_types = ['linear', 'inverse', 'power', 'exp', 'log', 'constant']
    results = {}
    
    print(f"Reference score (no weights): {score_no_weights:.6f}")
    print("\nTesting different weight types:")
    
    for weight_type in weight_types:
        # Configuration for this weight type
        weight_config = {
            'weighting': {
                'norm': {'is_used': True, 'type': weight_type}
            }
        }
        
        # Create flow filter and weights
        flow_filter = FlowFilterBatch(weight_config)
        weights = flow_filter.weight(test_flow_mx[None, ...])[0]
        
        # Calculate score with weights
        score_with_weights = batch_scorer.colin_score(test_flow_mx, ref_point_mx, weights=weights)
        
        # Calculate impact
        impact = score_with_weights - score_no_weights
        impact_percent = impact / score_no_weights * 100
        
        results[weight_type] = {
            'score': score_with_weights,
            'impact': impact,
            'impact_percent': impact_percent
        }
        
        print(f"  {weight_type:8s}: {score_with_weights:.6f} (impact: {impact:+.6f}, {impact_percent:+.1f}%)")
    
    return results

def test_batch_vs_sequential_with_weights(flows):
    """Test 7: Batch vs Sequential with weights (10 samples)"""
    print("\n=== Test 7: Batch vs Sequential with Weights (10 samples) ===")
    
    # Use provided flows
    test_flows = flows[:10]
    ref_points = [(flow.shape[1]//2, flow.shape[0]//2) for flow in test_flows]
    
    print(f"Testing with {len(test_flows)} flows")
    
    # Convert to MLX arrays
    test_flows_mx = mx.array(test_flows, dtype=mx.float32)
    ref_points_mx = mx.array(ref_points, dtype=mx.float32)
    
    # Initialize scorer and flow filter
    batch_scorer = BatchCollinearityScorer()
    weight_config = {
        'weighting': {
            'norm': {'is_used': True, 'type': 'linear'}
        }
    }
    flow_filter = FlowFilterBatch(weight_config)
    
    print("\nTesting WITHOUT weights...")
    
    # Test without weights
    sequential_scores_no_weights = mx.array([batch_scorer.colin_score(test_flows_mx[i], ref_points_mx[i]) 
                                           for i in range(len(test_flows))])
    batch_scores_no_weights = batch_scorer.colin_score_batch(test_flows_mx, ref_points_mx)
    
    # Compare without weights
    diff_no_weights = mx.abs(sequential_scores_no_weights - batch_scores_no_weights)
    print(f"Max difference (no weights): {mx.max(diff_no_weights):.8f}")
    print(f"Mean difference (no weights): {mx.mean(diff_no_weights):.8f}")
    
    print("\nTesting WITH weights...")
    
    # Create weights and test with weights
    weights_batch = flow_filter.weight(test_flows_mx)
    
    sequential_scores_with_weights = mx.array([batch_scorer.colin_score(test_flows_mx[i], ref_points_mx[i], weights=weights_batch[i]) 
                                             for i in range(len(test_flows))])
    batch_scores_with_weights = batch_scorer.colin_score_batch(test_flows_mx, ref_points_mx, weights=weights_batch)
    
    # Compare with weights
    diff_with_weights = mx.abs(sequential_scores_with_weights - batch_scores_with_weights)
    print(f"Max difference (with weights): {mx.max(diff_with_weights):.8f}")
    print(f"Mean difference (with weights): {mx.mean(diff_with_weights):.8f}")
    
    # Analyze impact
    impact_sequential = sequential_scores_with_weights - sequential_scores_no_weights
    impact_batch = batch_scores_with_weights - batch_scores_no_weights
    
    print(f"\nWeights impact analysis:")
    print(f"Sequential method - Mean impact: {mx.mean(impact_sequential):.4f} ± {mx.std(impact_sequential):.4f}")
    print(f"Batch method - Mean impact: {mx.mean(impact_batch):.4f} ± {mx.std(impact_batch):.4f}")
    print(f"Impact consistency: {mx.max(mx.abs(impact_sequential - impact_batch)):.8f}")

if __name__ == "__main__":
    print("🔬 Starting comprehensive collinearity scoring tests...")
    
    # Load flows once at the beginning
    print("📂 Loading flows data...")
    flows = load_flows(0, start_frame=0, end_frame=50)
    print(f"✅ Flows loaded: {flows.shape} ({flows.dtype})")
    
    # Run all tests with the loaded flows
    test_collinearity_maps(flows)
    test_single_sample_scores(flows)
    test_batch_scores(flows)
    test_gradients(flows)
    test_weights_impact(flows)
    test_different_weight_types(flows)
    test_batch_vs_sequential_with_weights(flows)
    print("\n✅ All comprehensive tests completed!") 