"""
This script provides tools to visualize colinearity scores between optical flow vectors and their
direction relative to a reference point (typically the ground truth vanishing point).

The colinearity score measures how well-aligned each flow vector is with the direction pointing
from the reference point to the flow vector's position. A score of 1 means perfect alignment
(the flow vector points exactly towards/away from the reference point), while a score of 0
means the flow vector is perpendicular to the direction to the reference point.

Key features:
- Computes colinearity scores for optical flow vectors relative to a reference point
- Visualizes these scores overlaid on the original video frame
- Filters out small/noisy flow vectors using a magnitude threshold
- Shows ground truth, reference point, and image center for comparison
- Provides statistics about the colinearity distribution

Usage example:
    python visualize_frame_colin.py
    # This will load frame 250 from video 4 and display colinearity scores
    # using the ground truth vanishing point as reference

The visualization uses a color map where:
- Higher colinearity (better alignment) = yellow/green
- Lower colinearity (worse alignment) = purple/blue
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Union, Optional
import numpy.typing as npt

from src.utilities.ground_truth import get_frame_pixel
from src.utilities.paths import get_labeled_dir, get_flows_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.core.flow_filter import FlowFilterSample


def compute_colinearity_for_points(
    pt: Tuple[float, float],
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
    u_flow: npt.NDArray[np.float64],
    v_flow: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Compute colinearity between a reference point and multiple flow vectors.
    
    Args:
        pt: Reference point (x, y)
        x_coords: X coordinates of flow vectors
        y_coords: Y coordinates of flow vectors
        u_flow: X components of flow vectors
        v_flow: Y components of flow vectors
        
    Returns:
        Colinearity values for each point
    """
    colinearity_values = np.zeros(len(x_coords))
    
    for i in range(len(x_coords)):
        # Vector from reference point to current point
        vec_to_point = np.array([x_coords[i] - pt[0], y_coords[i] - pt[1]])
        
        # Flow vector at current point
        flow_vec = np.array([u_flow[i], v_flow[i]])
        
        # Normalize vectors
        vec_to_point_norm = np.linalg.norm(vec_to_point)
        flow_vec_norm = np.linalg.norm(flow_vec)
        
        # Avoid division by zero
        if vec_to_point_norm > 0 and flow_vec_norm > 0:
            vec_to_point = vec_to_point / vec_to_point_norm
            flow_vec = flow_vec / flow_vec_norm
            
            # Compute dot product (colinearity)
            colinearity = np.abs(np.dot(vec_to_point, flow_vec))
            colinearity_values[i] = colinearity
    
    return colinearity_values


def plot_spatial_colinearity(
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
    colinearity: npt.NDArray[np.float64],
    title: str = "Spatial Distribution of Colinearity",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    Plot the spatial distribution of colinearity scores.
    
    Args:
        x_coords: X coordinates of flow vectors
        y_coords: Y coordinates of flow vectors
        colinearity: Colinearity values for each point
        title: Title of the plot
        figsize: Size of the figure (width, height)
        save_path: If provided, save the plot to this path
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(x_coords, y_coords, c=colinearity, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Colinearity')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_colinearity_on_frame(
    flow: npt.NDArray[np.float64],
    frame_idx: int,
    video_idx: int,
    ref_point: Tuple[float, float],
    stride: int = 2,
    min_flow_magnitude: float = 13.0,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
    display_img: bool = True
) -> None:
    """
    Visualize colinearity scores overlaid on a video frame.
    
    Args:
        flow: Flow field array for the specific frame (H, W, 2)
        frame_idx: Index of the frame to visualize
        video_idx: Index of the video
        ref_point: Reference point (x, y) for colinearity computation
        stride: Spacing between points (higher = fewer points)
        min_flow_magnitude: Minimum flow magnitude to consider (used for flow filtering)
        figsize: Size of the figure (width, height)
        save_path: If provided, save the plot to this path
        display_img: Whether to display the original frame as background
    """
    # Load the original frame for background
    video_path = get_labeled_dir() / f'{video_idx}.hevc'
    ret, frame_rgb = read_frame_rgb(video_path, frame_idx)

    if not ret:
        print(f"Could not read frame {frame_idx}")
        return

    # Get ground truth and center points
    gt_pixels = get_frame_pixel(video_idx, frame_idx)
    center_pixel = (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2)
    
    # Configure and apply flow filtering using FlowFilterSample
    filter_config = {
        'filtering': {
            'norm': {
                'is_used': True,
                'min_threshold': min_flow_magnitude
            }
        }
    }
    
    flow_filter = FlowFilterSample(filter_config)
    filtered_flow = flow_filter.filter(flow)
    
    # Create grid for points
    h, w = filtered_flow.shape[:2]
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Get flow vectors at grid points
    fx = filtered_flow[y, x, 0]
    fy = filtered_flow[y, x, 1]
    
    # Filter out zero flow values (these are the ones that were filtered out)
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > 0  # Only non-zero flow vectors remain after filtering
    
    # Compute colinearity for valid flow vectors
    colinearity = compute_colinearity_for_points(
        ref_point,
        x[mask],
        y[mask],
        fx[mask],
        fy[mask]
    )
    
    # Plot
    plt.figure(figsize=figsize)
    
    # Plot original frame with full opacity
    if display_img:
        plt.imshow(frame_rgb, alpha=1.0)
    else:
        # Set limits first, then invert y-axis to match image coordinate system
        plt.xlim(0, frame_rgb.shape[1])  # Use actual frame width
        plt.ylim(0, frame_rgb.shape[0])  # Use actual frame height
        ax = plt.gca()
        ax.invert_yaxis()  # Invert y-axis to match image coordinate system
        
    # Plot colinearity scores with smaller, more transparent points
    scatter = plt.scatter(
        x[mask],
        y[mask],
        c=colinearity,
        cmap='viridis',
        alpha=0.4,
        s=20  # Reduced point size
    )
    plt.colorbar(scatter, label='Colinearity Score')
    
    # Plot reference points
    plt.scatter(
        gt_pixels[0],
        gt_pixels[1],
        color='blue',
        s=150,
        marker='*',
        label='Ground Truth',
        edgecolor='white',
        linewidth=1
    )
    plt.scatter(
        ref_point[0],
        ref_point[1],
        color='red',
        s=150,
        marker='*',
        label='Reference Point',
        edgecolor='white',
        linewidth=1
    )
    plt.scatter(
        center_pixel[0],
        center_pixel[1],
        color='green',
        s=150,
        marker='*',
        label='Center',
        edgecolor='white',
        linewidth=1
    )
    
    plt.title(f'Colinearity Scores - Video {video_idx}, Frame {frame_idx}')
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def print_colinearity_stats(colinearity: npt.NDArray[np.float64], prefix: str = "") -> None:
    """
    Print statistics about the colinearity distribution.
    
    Args:
        colinearity: Array of colinearity values
        prefix: Optional prefix for the output (e.g., "GT" or "Pred")
    """
    if prefix:
        print(f"{prefix} colinearity stats:")
    else:
        print("Colinearity stats:")
    print(f"  Mean: {np.mean(colinearity):.4f}")
    print(f"  Median: {np.median(colinearity):.4f}")
    print(f"  Min: {np.min(colinearity):.4f}")
    print(f"  Max: {np.max(colinearity):.4f}")
    print(f"  Std: {np.std(colinearity):.4f}")


if __name__ == "__main__":
    # Example usage with a video frame:
    video_idx = 1
    frame_idx = 201
    
    # Load flows
    flow_file = get_flows_dir() / f'{video_idx}.npy'
    if not flow_file.exists():
        print(f"Flow file not found: {flow_file}")
        exit(1)
    
    flows = np.load(flow_file)
    flow = flows[frame_idx]
    
    # Get ground truth point as reference
    gt_pixels = get_frame_pixel(video_idx, frame_idx)
    ref_point = (gt_pixels[0], gt_pixels[1])  # Using ground truth as reference
    
    # Visualize colinearity scores
    plot_colinearity_on_frame(flow, frame_idx, video_idx, ref_point, min_flow_magnitude=13, display_img= False)