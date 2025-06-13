#!/usr/bin/env python3
"""
Benchmark script to compare different methods of loading flow samples.
"""

import numpy as np
import time
from pathlib import Path
from src.utilities.paths import get_flows_dir

def benchmark_full_load_then_slice(video_index=4, n_samples=50):
    """Method 1: Load full file, then slice"""
    flows_dir = get_flows_dir()
    flow_file = flows_dir / f"{video_index}.npy"
    
    print(f"Method 1: Full load then slice ({n_samples} samples)")
    start_time = time.time()
    
    # Load entire file
    flows = np.load(flow_file)
    load_time = time.time()
    
    # Sample frames (every nth frame to get n_samples)
    total_frames = flows.shape[0]
    step = max(1, total_frames // n_samples)
    sample_indices = list(range(0, total_frames, step))[:n_samples]
    sampled_flows = flows[sample_indices]
    
    end_time = time.time()
    
    print(f"  - Full file shape: {flows.shape}")
    print(f"  - Sampled shape: {sampled_flows.shape}")
    print(f"  - Load time: {load_time - start_time:.2f}s")
    print(f"  - Total time: {end_time - start_time:.2f}s")
    print(f"  - Memory usage: ~{flows.nbytes / 1e9:.1f} GB")
    
    return sampled_flows, end_time - start_time

def benchmark_mmap_sampling(video_index=4, n_samples=50):
    """Method 2: Memory mapping with direct sampling"""
    flows_dir = get_flows_dir()
    flow_file = flows_dir / f"{video_index}.npy"
    
    print(f"\nMethod 2: Memory mapping with sampling ({n_samples} samples)")
    start_time = time.time()
    
    # Load with memory mapping
    flows_mmap = np.load(flow_file, mmap_mode='r')
    mmap_time = time.time()
    
    # Sample frames directly
    total_frames = flows_mmap.shape[0]
    step = max(1, total_frames // n_samples)
    sample_indices = list(range(0, total_frames, step))[:n_samples]
    
    # This should only load the sampled frames into memory
    sampled_flows = flows_mmap[sample_indices].copy()
    
    end_time = time.time()
    
    print(f"  - Full file shape: {flows_mmap.shape}")
    print(f"  - Sampled shape: {sampled_flows.shape}")
    print(f"  - Mmap setup time: {mmap_time - start_time:.2f}s")
    print(f"  - Total time: {end_time - start_time:.2f}s")
    print(f"  - Memory usage: ~{sampled_flows.nbytes / 1e9:.1f} GB (only sampled data)")
    
    return sampled_flows, end_time - start_time

def verify_results_identical(flows1, flows2):
    """Verify that both methods produce identical results"""
    print(f"\nVerification:")
    print(f"  - Shapes match: {flows1.shape == flows2.shape}")
    if flows1.shape == flows2.shape:
        are_equal = np.allclose(flows1, flows2)
        print(f"  - Arrays are identical: {are_equal}")
        if not are_equal:
            diff = np.abs(flows1 - flows2)
            print(f"  - Max difference: {np.max(diff)}")
    return flows1.shape == flows2.shape and np.allclose(flows1, flows2)

def main():
    print("🚀 BENCHMARK: Flow Loading Methods")
    print("=" * 60)
    
    # Check if file exists
    flows_dir = get_flows_dir()
    flow_file = flows_dir / "4.npy"
    
    if not flow_file.exists():
        print(f"❌ File not found: {flow_file}")
        return
    
    file_size_gb = flow_file.stat().st_size / 1e9
    print(f"📁 File: {flow_file}")
    print(f"📊 File size: {file_size_gb:.1f} GB")
    
    # Run benchmarks
    try:
        flows1, time1 = benchmark_full_load_then_slice(n_samples=50)
        flows2, time2 = benchmark_mmap_sampling(n_samples=50)
        
        # Verify results
        identical = verify_results_identical(flows1, flows2)
        
        # Summary
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"=" * 40)
        print(f"Method 1 (full load): {time1:.2f}s")
        print(f"Method 2 (mmap):      {time2:.2f}s")
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"Speedup: {speedup:.1f}x faster" if speedup > 1 else f"Slowdown: {1/speedup:.1f}x slower")
        print(f"Results identical: {identical}")
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 