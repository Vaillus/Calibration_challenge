import numpy as np
import mlx.core as mx
import os
import time
import shutil
from pathlib import Path
import psutil

def get_free_space(path):
    """Get free space in GB for the given path."""
    try:
        # Get the disk usage for the specific path
        usage = psutil.disk_usage(path)
        free_gb = usage.free / (1024**3)  # Convert to GB
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        
        print(f"Disk space details for {path}:")
        print(f"  Total: {total_gb:.2f} GB")
        print(f"  Used: {used_gb:.2f} GB")
        print(f"  Free: {free_gb:.2f} GB")
        
        return free_gb
    except Exception as e:
        print(f"Warning: Could not get disk space information: {e}")
        return None

def convert_npy_to_mlx(npy_path, mlx_path):
    """Convert a single .npy file to .mlx format."""
    print(f"\nConverting {npy_path} to {mlx_path}")
    
    # Check available disk space
    free_space = get_free_space(os.path.dirname(mlx_path))
    if free_space is not None and free_space < 5:  # Require at least 5GB free
        print(f"Warning: Low disk space ({free_space:.2f} GB). This might cause issues.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled due to low disk space.")
            return
    
    # Load the numpy array using memmap
    start_time = time.time()
    print("Loading numpy array...")
    try:
        npy_data = np.load(npy_path, mmap_mode='r')
        print(f"Loaded numpy array of shape {npy_data.shape}")
    except Exception as e:
        print(f"Error loading numpy file: {e}")
        return
    
    # Process in chunks to avoid memory issues
    chunk_size = 100  # Process 100 flows at a time
    total_flows = npy_data.shape[0]
    
    # Create empty MLX array for the full dataset
    try:
        mlx_data = mx.zeros(npy_data.shape, dtype=mx.float32)
    except Exception as e:
        print(f"Error creating MLX array: {e}")
        return
    
    for start_idx in range(0, total_flows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_flows)
        print(f"Processing chunk {start_idx} to {end_idx-1}")
        
        try:
            # Convert chunk to MLX and assign directly to the target array
            chunk = npy_data[start_idx:end_idx]
            mlx_data[start_idx:end_idx] = mx.array(chunk)
            
            # Clear memory
            del chunk
            mx.clear_cache()
        except Exception as e:
            print(f"Error processing chunk {start_idx} to {end_idx-1}: {e}")
            return
    
    # Save as MLX file using native format
    print("Saving MLX file...")
    try:
        # Save in MLX's native format
        mx.savez(str(mlx_path), data=mlx_data)
    except Exception as e:
        print(f"Error saving MLX file: {e}")
        return
    
    # Clean up
    del npy_data
    del mlx_data
    mx.clear_cache()
    
    end_time = time.time()
    print(f"Conversion completed in {end_time - start_time:.2f} seconds")

def main():
    # Create MLX directory if it doesn't exist
    mlx_dir = Path("calib_challenge/flows_mlx")
    mlx_dir.mkdir(exist_ok=True)
    
    # Get all .npy files
    npy_dir = Path("calib_challenge/flows")
    npy_files = sorted(npy_dir.glob("*.npy"))
    
    print(f"Found {len(npy_files)} .npy files to convert")
    
    # Convert each file
    for npy_file in npy_files:
        # Create MLX file path with correct extension
        mlx_file = mlx_dir / f"{npy_file.stem}.npz"  # MLX uses .npz extension for its native format
        convert_npy_to_mlx(str(npy_file), str(mlx_file))

if __name__ == "__main__":
    main() 