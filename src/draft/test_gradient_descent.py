import time
import numpy as np
import mlx.core as mx
from pathlib import Path
import sys
import os
from scipy.optimize import minimize

# Add parent directory to path to import ground_truth
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ground_truth import get_frame_pixel, read_ground_truth_pixels

from colinearity_optimization_parallel import ParallelVanishingPointEstimator
from colinearity_optimization import VanishingPointEstimator

def visualize_score_surface(flow, label):
    """
    Visualize the colinearity score surface, as well as the ground truth point and the trajectory 
    of the optimization process using the Adam optimizer.
    """
    assert flow.ndim == 3, "flow must be 2D array (h, w, 2)"
    pve = ParallelVanishingPointEstimator(flow.shape[1], flow.shape[0], use_max_distance=False, use_reoptimization=False)
    
    # Convertir flow et label au format batch
    flow_batch = flow.reshape(1, flow.shape[0], flow.shape[1], 2)
    flow_batch = mx.array(flow_batch, dtype=mx.float32)
    label_batch = mx.array([label], dtype=mx.float32)
    
    # Create a grid of points
    x_points = np.linspace(0, flow.shape[1]-1, 20)
    y_points = np.linspace(0, flow.shape[0]-1, 20)
    X, Y = np.meshgrid(x_points, y_points)

    # Initialize array to store scores
    scores = np.zeros((len(y_points), len(x_points)))

    # Evaluate colin_score_batch for each point
    for i, y in enumerate(y_points):
        for j, x in enumerate(x_points):
            point_batch = mx.array([[x, y]], dtype=mx.float32)
            scores[i, j] = float(pve.colin_score_batch(flow_batch, point_batch)[0])
    
    print("scores grid computed")
    
    # Create 2D topographic visualization
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create contour plot (topographic map)
    contour = ax.contourf(X, Y, scores, levels=20, cmap='viridis', alpha=0.8)
    contour_lines = ax.contour(X, Y, scores, levels=20, colors='black', alpha=0.3, linewidths=0.5)
    
    # Add contour labels
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.3f')

    # Plot the ground truth point
    label_score = float(pve.colin_score_batch(flow_batch, label_batch)[0])
    ax.scatter(label[0], label[1], color='red', s=200, label='Ground Truth', 
            edgecolor='black', linewidth=2, zorder=5)

    # Plot trajectory
    trajectory, trajectory_scores = optimize_new(flow_batch)
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            color='blue', label='Adam Trajectory', linewidth=3, marker='o', 
            markersize=6, markeredgecolor='white', markeredgewidth=1, zorder=4)
    
    # Mark starting point
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=150, 
            label='Starting Point', marker='s', edgecolor='black', linewidth=2, zorder=5)
    
    # Mark final point
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='orange', s=150, 
            label='Final Point', marker='*', edgecolor='black', linewidth=2, zorder=5)

    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Colinearity Score', rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title('Carte Topographique du Score de Colinéarité (Vue de dessus)')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set aspect ratio to equal for proper visualization
    ax.set_aspect('equal', adjustable='box')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def optimize_old(flow, label):
    """This function is here only for comparison purposes."""
    vpe = VanishingPointEstimator(flow.shape[1], flow.shape[0], use_max_distance=False, use_reoptimization=False)

    starting_point = (flow.shape[0]//2, flow.shape[1]//2)
    all_points = [starting_point]
    all_scores = [vpe.colin_score(flow,starting_point)]
    # Callback function to record the trajectory
    def callback(point):
        all_points.append(point.copy())
        all_scores.append(vpe.colin_score(flow, point))

    start_time = time.time()
    # Optimization with L-BFGS-B without strict bounds
    result = minimize(
        lambda point: vpe.objective_function(point, flow),
        starting_point,
        method='L-BFGS-B',
        callback=callback,
        options={'disp': False}
    )
    end_time = time.time()
    print(f"Time taken to optimize: {end_time - start_time:.2f} seconds")
    return all_points, all_scores

def optimize_new(flow_batch):
    vpe = ParallelVanishingPointEstimator(flow_batch.shape[2], flow_batch.shape[1], use_max_distance=False, use_reoptimization=False)
    starting_point = (flow_batch.shape[2]//2, flow_batch.shape[1]//2)  # Centre de l'image
    starting_point = mx.array([starting_point], dtype=mx.float32)
    
    final_point, trajectory, scores = adam_optimize_with_tracking(lambda point: - vpe.colin_score_batch(flow_batch, point)[0], starting_point)
    return trajectory, scores

def adam_optimize(loss_fn, x0, lr=10.0, beta1=0.6, beta2=0.98, eps=1e-8, max_iter=50):
    """
    Adam optimizer WITHOUT trajectory tracking - optimized for batch processing.
    Memory efficient version for parallel processing.
    """
    x = x0
    m = mx.zeros_like(x0)  # momentum
    v = mx.zeros_like(x0)  # variance
    
    # Create a vectorized gradient function using vmap
    def single_sample_grad(single_point, sample_idx):
        def single_loss(point):
            point_batch = mx.expand_dims(point, 0)
            full_scores = loss_fn(point_batch)
            return full_scores[sample_idx]
        return mx.grad(single_loss)(single_point)
    
    # Vectorize the gradient computation across the batch
    batch_grad_fn = mx.vmap(single_sample_grad, in_axes=(0, 0))
    
    for t in range(1, max_iter + 1):
        # Create indices for each sample in the batch
        batch_size = x.shape[0]
        sample_indices = mx.arange(batch_size)
        
        # Compute gradients for all samples in parallel
        grad = batch_grad_fn(x, sample_indices)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = x - lr * m_hat / (mx.sqrt(v_hat) + eps)
    
    return x

def adam_optimize_with_tracking(loss_fn, x0, lr=10.0, beta1=0.6, beta2=0.98, eps=1e-8, max_iter=50):
    """
    Adam optimizer WITH trajectory tracking - for visualization and analysis.
    Use this only for single samples or small batches due to memory overhead.
    """
    x = x0
    m = mx.zeros_like(x0)  # momentum
    v = mx.zeros_like(x0)  # variance
    
    # Track trajectory
    trajectory = [np.array(x[0])]  # Convert first point to numpy and extract from batch dimension
    scores = [float(-loss_fn(x))]  # Store the actual score (not the negative loss)
    
    print(f"Initial loss: {loss_fn(x)}")
    
    for t in range(1, max_iter + 1):
        grad = mx.grad(loss_fn)(x)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = x - lr * m_hat / (mx.sqrt(v_hat) + eps)
        
        # Track the trajectory
        trajectory.append(np.array(x[0]))  # Convert to numpy and extract from batch dimension
        scores.append(float(-loss_fn(x)))  # Store the actual score (not the negative loss)
        
        if t % 10 == 0:  # Print every 10 iterations instead of every iteration
            print(f"Iteration {t}: x = {np.array(x[0])}, score = {scores[-1]:.6f}")
    
    return x, trajectory, scores

def adam_optimize_single(single_flow, starting_point, lr=10.0, beta1=0.6, beta2=0.98, eps=1e-8, max_iter=50):
    """
    Adam optimizer for a SINGLE sample.
    Now uses colin_score directly instead of converting to batch format.
    """
    vpe = ParallelVanishingPointEstimator(single_flow.shape[1], single_flow.shape[0], use_max_distance=False, use_reoptimization=False)
    
    x = starting_point
    m = mx.zeros_like(x)  # momentum
    v = mx.zeros_like(x)  # variance
    
    for t in range(1, max_iter + 1):
        # Loss function for single sample - returns scalar
        def single_loss(point):
            # Use colin_score directly - no batch conversion needed!
            score = vpe.colin_score(single_flow, point)
            return -score  # Negative because we want to maximize score
        
        # Compute gradient for this single sample
        grad = mx.grad(single_loss)(x)
        
        # Adam updates
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        x = x - lr * m_hat / (mx.sqrt(v_hat) + eps)
    
    return x

def adam_optimize_parallel(flow_batch, starting_points, lr=10.0, beta1=0.6, beta2=0.98, eps=1e-8, max_iter=50):
    """
    Parallel Adam optimization using vmap to vectorize single-sample optimization.
    Each sample optimizes independently - no aggregation!
    """
    # Vectorize the single-sample optimizer across the batch
    vectorized_optimizer = mx.vmap(
        lambda flow, start_point: adam_optimize_single(flow, start_point, lr, beta1, beta2, eps, max_iter),
        in_axes=(0, 0)  # Both flow_batch and starting_points vary along axis 0
    )
    
    # Apply to the entire batch - each sample optimized independently
    final_points = vectorized_optimizer(flow_batch, starting_points)
    
    return final_points

def optimize_batch_efficient(flow_batch):
    """
    Simple batch optimization - no nested functions, no for loops.
    """
    # Initialize starting points
    batch_size = flow_batch.shape[0]
    center_x = flow_batch.shape[2] // 2
    center_y = flow_batch.shape[1] // 2
    
    # Create starting points - center of image for each sample
    center_point = mx.array([center_x, center_y], dtype=mx.float32)
    starting_points = mx.tile(center_point, (batch_size, 1))
    
    # Optimize - that's it!
    final_points = adam_optimize_parallel(flow_batch, starting_points)
    
    return final_points

if __name__ == "__main__":
    
    npy_path = Path("calib_challenge/flows/0.npy")
    flow_batch = np.load(npy_path, mmap_mode='r')[0:10]  # Load only first sample using memory mapping
    flow_batch = mx.array(flow_batch)

    labels = mx.array(read_ground_truth_pixels(0)[1:11])
    print(f"\nGround truth label: {labels}")

    preds = optimize_batch_efficient(flow_batch)
    print(f"Predictions: {preds}")
    # compute difference between preds and labels
    distances = mx.sqrt(mx.sum(mx.square(preds - labels), axis=1))
    print(f"Difference between preds and labels: {distances}")


    # print("\n=== Testing Adam optimization ===")
    # trajectory_new, scores_new = optimize_new(np.array(flow), label)
    # print(f"Final result: {trajectory_new[-1]}")
    # print(f"Final score: {scores_new[-1]:.6f}")

    # Activate visualization
    # visualize_score_surface(np.array(flow), label)

