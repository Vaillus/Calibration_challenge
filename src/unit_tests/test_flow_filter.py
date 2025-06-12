#!/usr/bin/env python3
"""
Test script to verify that the vectorized batch version of FlowFilter produces the same weighting results
as the sequential version for a single frame and for batches. Also tests filtering consistency.
"""

import numpy as np
import mlx.core as mx
from src.core.flow_filter import FlowFilterSample, FlowFilterBatch
from src.utilities.load_flows import load_flows

def compare_weights(sequential_weights, batch_weights, tolerance=1e-6):
    """Compare two weight arrays and print statistics."""
    # Convert to numpy for easier comparison
    seq_np = np.array(sequential_weights)
    batch_np = np.array(batch_weights)
    
    # Check shapes
    print(f"Sequential weights shape: {seq_np.shape}")
    print(f"Batch weights shape: {batch_np.shape}")
    
    # Check if arrays are equal within tolerance
    is_equal = np.allclose(seq_np, batch_np, rtol=tolerance, atol=tolerance)
    print(f"Arrays are equal within tolerance {tolerance}: {is_equal}")
    
    if not is_equal:
        # Print some statistics about the differences
        diff = np.abs(seq_np - batch_np)
        print(f"Max difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")
        print(f"Number of different elements: {np.sum(diff > tolerance)}")
        
        # Print some example values where they differ
        print("\nExample differences (first 5):")
        diff_indices = np.where(diff > tolerance)
        for i in range(min(5, len(diff_indices[0]))):
            if len(diff_indices) == 2:  # 2D array
                h, w = diff_indices[0][i], diff_indices[1][i]
                print(f"Position ({h},{w}):")
                print(f"  Sequential: {seq_np[h,w]:.6f}")
                print(f"  Batch: {batch_np[h,w]:.6f}")
                print(f"  Difference: {diff[h,w]:.6f}")
            else:  # 3D array (batch)
                b, h, w = diff_indices[0][i], diff_indices[1][i], diff_indices[2][i]
                print(f"Position ({b},{h},{w}):")
                print(f"  Sequential: {seq_np[b,h,w]:.6f}")
                print(f"  Batch: {batch_np[b,h,w]:.6f}")
                print(f"  Difference: {diff[b,h,w]:.6f}")

def compare_flows(sequential_flow, batch_flow, tolerance=1e-6):
    """Compare two flow arrays and print statistics."""
    # Convert to numpy for easier comparison
    seq_np = np.array(sequential_flow)
    batch_np = np.array(batch_flow)
    
    # Check shapes
    print(f"Sequential flow shape: {seq_np.shape}")
    print(f"Batch flow shape: {batch_np.shape}")
    
    # Check if arrays are equal within tolerance
    is_equal = np.allclose(seq_np, batch_np, rtol=tolerance, atol=tolerance)
    print(f"Flows are equal within tolerance {tolerance}: {is_equal}")
    
    if not is_equal:
        # Print some statistics about the differences
        diff = np.abs(seq_np - batch_np)
        print(f"Max difference: {np.max(diff)}")
        print(f"Mean difference: {np.mean(diff)}")
        print(f"Number of different elements: {np.sum(diff > tolerance)}")
    
    # Print filtering statistics
    original_nonzero = np.sum(seq_np != 0)
    print(f"Non-zero elements after filtering: {original_nonzero}")

def test_batch_50_frames_filtering():
    """Test filtering for 50 frames: sequential processing vs batch processing."""
    print("\n" + "="*70)
    print("BATCH OF 50 FRAMES FILTERING TEST")
    print("="*70)
    
    # Load 50 frames from video 4 (frames 100-149)
    print("Loading 50 frames from video 4...")
    flows = load_flows(video_index=4, start_frame=100, end_frame=149, return_mlx=True, verbose=True)
    if flows is None:
        print("❌ Failed to load flows")
        return
    
    print(f"Loaded flows shape: {flows.shape}")
    print(f"Original non-zero elements: {np.sum(np.array(flows) != 0)}")
    
    # Test filtering with minimum norm threshold
    min_threshold = 13
    print(f"\nTesting filtering with minimum norm threshold: {min_threshold}")
    print(f"{'-'*50}")
    
    # Create filter configs - ONLY filtering, no weighting
    filter_config = {
        'filtering': {
            'norm': {
                'is_used': True,
                'min_threshold': min_threshold
            }
        }
    }
    
    # Create both filter versions
    seq_filter = FlowFilterSample(filter_config)
    batch_filter = FlowFilterBatch(filter_config)
    
    # Sequential processing: process each frame individually
    print(f"--- Sequential Processing (50 frames individually) ---")
    sequential_filtered_flows = []
    for i in range(flows.shape[0]):
        single_flow = flows[i]
        filtered_flow = seq_filter.filter(single_flow)
        # Ensure filtered flows are MLX arrays
        if not isinstance(filtered_flow, mx.array):
            filtered_flow = mx.array(filtered_flow)
        sequential_filtered_flows.append(filtered_flow)
    
    # Stack all sequential results
    sequential_filtered_flows = mx.stack(sequential_filtered_flows)
    print(f"Sequential filtered flows shape: {sequential_filtered_flows.shape}")
    print(f"Sequential filtered flows non-zero elements: {np.sum(np.array(sequential_filtered_flows) != 0)}")
    
    # Batch processing: process all frames at once
    print(f"--- Batch Processing (50 frames at once) ---")
    batch_filtered_flows = batch_filter.filter(flows)
    print(f"Batch filtered flows shape: {batch_filtered_flows.shape}")
    print(f"Batch filtered flows non-zero elements: {np.sum(np.array(batch_filtered_flows) != 0)}")
    
    # Compare results
    print(f"--- Filtered Flows Comparison ---")
    compare_flows(sequential_filtered_flows, batch_filtered_flows)

def test_single_frame_filtering():
    """Test filtering for a single frame using both approaches."""
    print("\n" + "="*70)
    print("SINGLE FRAME FILTERING TEST")
    print("="*70)
    
    # Load just one frame from video 4 (frame 100 for example)
    print("Loading single frame from video 4...")
    flow = load_flows(video_index=4, start_frame=100, end_frame=100, return_mlx=True, verbose=True)
    if flow is None:
        print("❌ Failed to load flow")
        return
    
    # Extract the single frame (remove batch dimension)
    single_flow = flow[0]  # Shape: (h, w, 2)
    print(f"Single frame shape: {single_flow.shape}")
    print(f"Original non-zero elements: {np.sum(np.array(single_flow) != 0)}")
    
    # Create batch version (add batch dimension back)
    batch_flow = mx.expand_dims(single_flow, axis=0)  # Shape: (1, h, w, 2)
    print(f"Batch flow shape: {batch_flow.shape}")
    
    # Test filtering with minimum norm threshold
    min_threshold = 13
    print(f"\nTesting filtering with minimum norm threshold: {min_threshold}")
    print(f"{'-'*50}")
    
    # Create filter configs - ONLY filtering, no weighting
    filter_config = {
        'filtering': {
            'norm': {
                'is_used': True,
                'min_threshold': min_threshold
            }
        }
    }
    
    # Create both filter versions
    seq_filter = FlowFilterSample(filter_config)
    batch_filter = FlowFilterBatch(filter_config)
    
    # Test ONLY filtering (not weighting)
    print(f"--- Sequential Filtering ---")
    seq_filtered_flow = seq_filter.filter(single_flow)
    print(f"Sequential filtered flow non-zero elements: {np.sum(np.array(seq_filtered_flow) != 0)}")
    
    print(f"--- Batch Filtering ---")
    batch_filtered_flow = batch_filter.filter(batch_flow)
    print(f"Batch filtered flow non-zero elements: {np.sum(np.array(batch_filtered_flow) != 0)}")
    
    # Extract single result from batch (remove batch dimension)
    batch_filtered_single = batch_filtered_flow[0]
    
    print(f"--- Filtered Flows Comparison ---")
    compare_flows(seq_filtered_flow, batch_filtered_single)

def test_single_frame_weighting():
    """Test weighting for a single frame using both approaches."""
    print("\n" + "="*70)
    print("SINGLE FRAME WEIGHTING TEST")
    print("="*70)
    
    # Load just one frame from video 4 (frame 100 for example)
    print("Loading single frame from video 4...")
    flow = load_flows(video_index=4, start_frame=100, end_frame=100, return_mlx=True, verbose=True)
    if flow is None:
        print("❌ Failed to load flow")
        return
    
    # Extract the single frame (remove batch dimension)
    single_flow = flow[0]  # Shape: (h, w, 2)
    print(f"Single frame shape: {single_flow.shape}")
    
    # Create batch version (add batch dimension back)
    batch_flow = mx.expand_dims(single_flow, axis=0)  # Shape: (1, h, w, 2)
    print(f"Batch flow shape: {batch_flow.shape}")
    
    # Test each weight type
    weight_types = ['linear', 'inverse', 'power', 'exp', 'log', 'constant']
    
    for weight_type in weight_types:
        print(f"\n{'-'*50}")
        print(f"Testing weight type: {weight_type}")
        print(f"{'-'*50}")
        
        # Create filter configs - ONLY weighting, no filtering
        filter_config = {
            'weighting': {
                'norm': {
                    'is_used': True,
                    'type': weight_type
                }
            }
        }
        
        # Create both filter versions
        seq_filter = FlowFilterSample(filter_config)
        batch_filter = FlowFilterBatch(filter_config)
        
        # Test ONLY weighting (not filtering)
        print(f"--- Sequential Weighting ---")
        seq_weights = seq_filter.weight(single_flow)
        print(f"Sequential weights range: [{np.min(np.array(seq_weights)):.6f}, {np.max(np.array(seq_weights)):.6f}]")
        
        print(f"--- Batch Weighting ---")
        batch_weights = batch_filter.weight(batch_flow)
        print(f"Batch weights range: [{np.min(np.array(batch_weights)):.6f}, {np.max(np.array(batch_weights)):.6f}]")
        
        # Extract single result from batch (remove batch dimension)
        batch_weights_single = batch_weights[0]
        
        print(f"--- Weights Comparison ---")
        compare_weights(seq_weights, batch_weights_single)

def test_batch_50_frames_weighting():
    """Test weighting for 50 frames: sequential processing vs batch processing."""
    print("\n" + "="*70)
    print("BATCH OF 50 FRAMES WEIGHTING TEST")
    print("="*70)
    
    # Load 50 frames from video 4 (frames 100-149)
    print("Loading 50 frames from video 4...")
    flows = load_flows(video_index=4, start_frame=100, end_frame=149, return_mlx=True, verbose=True)
    if flows is None:
        print("❌ Failed to load flows")
        return
    
    print(f"Loaded flows shape: {flows.shape}")
    
    # Test each weight type
    weight_types = ['linear', 'inverse', 'power', 'exp', 'log', 'constant']
    
    for weight_type in weight_types:
        print(f"\n{'-'*50}")
        print(f"Testing weight type: {weight_type}")
        print(f"{'-'*50}")
        
        # Create filter configs - ONLY weighting, no filtering
        filter_config = {
            'weighting': {
                'norm': {
                    'is_used': True,
                    'type': weight_type
                }
            }
        }
        
        # Create both filter versions
        seq_filter = FlowFilterSample(filter_config)
        batch_filter = FlowFilterBatch(filter_config)
        
        # Sequential processing: process each frame individually
        print(f"--- Sequential Processing (50 frames individually) ---")
        sequential_weights = []
        for i in range(flows.shape[0]):
            single_flow = flows[i]
            weights = seq_filter.weight(single_flow)
            # Ensure weights are MLX arrays
            if not isinstance(weights, mx.array):
                weights = mx.array(weights)
            sequential_weights.append(weights)
        
        # Stack all sequential results
        sequential_weights = mx.stack(sequential_weights)
        print(f"Sequential weights shape: {sequential_weights.shape}")
        print(f"Sequential weights range: [{np.min(np.array(sequential_weights)):.6f}, {np.max(np.array(sequential_weights)):.6f}]")
        
        # Batch processing: process all frames at once
        print(f"--- Batch Processing (50 frames at once) ---")
        batch_weights = batch_filter.weight(flows)
        print(f"Batch weights shape: {batch_weights.shape}")
        print(f"Batch weights range: [{np.min(np.array(batch_weights)):.6f}, {np.max(np.array(batch_weights)):.6f}]")
        
        # Compare results
        print(f"--- Weights Comparison ---")
        compare_weights(sequential_weights, batch_weights)

if __name__ == "__main__":
    print("Testing FlowFilter: Filtering and Weighting Comparison")
    print("Sequential (FlowFilterSample) vs Batch (FlowFilterBatch)")
    print("=" * 70)
    
    # Run only the batch filtering test (others are commented out since already validated)
    test_batch_50_frames_filtering()
    
    # Previously validated tests (commented out):
    # test_single_frame_filtering()
    # test_single_frame()
    # test_batch_50_frames() 