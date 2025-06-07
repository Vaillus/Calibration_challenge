import numpy as np
import matplotlib.pyplot as plt
import cv2

import mlx.core as mx

from src.utilities.pixel_angle_converter import pixels_to_angles
from src.core.flow_filter import FlowFilterSample

class VanishingPointEstimator:
    # ===== Core initialization and configuration =====
    def __init__(
            self, 
            frame_width, 
            frame_height, 
            focal_length=910, 
            max_distance=100, 
            use_max_distance=True,
            use_reoptimization=True
        ):
        """
        Initialize the VanishingPointEstimator.
        
        Note: Cette classe utilise maintenant les optimiseurs centralisés du module 
        core.optimizers pour L-BFGS-B. Pour Adam, utilisez AdamOptimizer directement.
        
        Args:
            frame_width (int): Width of the video frames
            frame_height (int): Height of the video frames
            focal_length (int): Camera focal length in pixels
            max_distance (int): Maximum allowed distance from center for vanishing point
            use_max_distance (bool): Whether to enforce the maximum distance constraint
            use_reoptimization (bool): Whether to use previous vanishing point as starting point
                                       for optimization (True) or always start from default point (False)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focal_length = focal_length
        self.max_distance = max_distance
        self.use_max_distance = use_max_distance
        self.use_reoptimization = use_reoptimization
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Default vanishing point is the center of the image
        self.default_vanishing_point = (self.center_x, self.center_y)
        # Previous vanishing point starts at the default
        self.previous_vanishing_point = self.default_vanishing_point
        
        # Initialiser le filtre de flow avec une configuration par défaut
        filter_config = {
            'filtering': {
                'norm': {'is_used': True, 'min_threshold': 1e-2}
            },
            'weighting': {
                'norm': {'is_used': True, 'type': 'linear'}
            }
        }
        self.flow_filter = FlowFilterSample(filter_config)

    # ===== Vector and colinearity calculations =====

    def colin_score(self, flow, pt, step=1, weights=None):
        """
        Computes the colinearity between the flow vector and the vector from pt to each pixel.
        Skips computation for zero vectors (filtered vectors).
        
        """
        colinearity_map = self.compute_colinearity_map(flow, pt, step)

        # Calculate the average score (ignore pixels that don't have flow)
        valid_pixels = (flow[:,:,0]**2 + flow[:,:,1]**2) > 0
        
        if np.sum(valid_pixels) > 0:
            if weights is not None:
                # Use weights to weight the colinearity scores
                weighted_sum = np.sum(colinearity_map * valid_pixels * weights)
                total_weight = np.sum(valid_pixels * weights)
                score = weighted_sum / total_weight if total_weight > 0 else 0
            else:
                # Original behavior without weights
                score = np.sum(colinearity_map * valid_pixels) / np.sum(valid_pixels)
        else:
            score = 0
        return score

    def compute_colinearity_map(self, flow:np.ndarray, pt:tuple, step:int=1):
        """
        Computes the colinearity between the flow vector and the vector from pt to each pixel.
        Skips computation for zero vectors (filtered vectors).
        
        Args:
            flow: Optical flow field of shape (h, w, 2)
            pt: Tuple (x, y) representing the reference point
            step: Integer, compute colinearity every 'step' pixels for faster processing (default=1)
            
        Returns:
            numpy.ndarray: A 2D array of colinearity values between -1 and 1
        """
        h, w = flow.shape[:2]
        colinearity_map = np.zeros((h, w))
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get the flow vector at this pixel
                flow_vector = (flow[y, x, 0], flow[y, x, 1])
                
                # Skip if flow vector is zero (filtered)
                if flow_vector[0] == 0 and flow_vector[1] == 0:
                    continue
                
                # Get the vector from pt to this pixel
                pixel_vector = self._get_vector_to_pixel(pt, (x, y))
                
                # Compute colinearity
                colinearity_value = self._compute_colinearity(flow_vector, pixel_vector)
                
                # Fill in the step x step block with the same value
                y_end = min(y + step, h)
                x_end = min(x + step, w)
                colinearity_map[y:y_end, x:x_end] = colinearity_value
        
        return colinearity_map
    
    def _get_vector_to_pixel(self, ref_point, pixel_point):
        """
        Returns the vector from a reference point to a specified pixel.
        
        Args:
            ref_point: Tuple (x, y) representing the reference point coordinates
            pixel_point: Tuple (x, y) representing the target pixel coordinates
            
        Returns:
            Tuple (dx, dy) representing the vector components from ref_point to pixel_point
        """
        dx = pixel_point[0] - ref_point[0]
        dy = pixel_point[1] - ref_point[1]
        return (dx, dy)

    def _compute_colinearity(self, vector1, vector2):
        """
        Computes the colinearity between two vectors.
        
        Args:
            vector1: Tuple (dx1, dy1) representing the first vector
            vector2: Tuple (dx2, dy2) representing the second vector
            
        Returns:
            float: A value between -1 and 1 where:
                1 means vectors are parallel and pointing in the same direction
                0 means vectors are perpendicular
               -1 means vectors are parallel but pointing in opposite directions
        """
        # Extract vector components
        dx1, dy1 = vector1
        dx2, dy2 = vector2
        
        # Calculate magnitudes of vectors
        mag1 = np.sqrt(dx1**2 + dy1**2)
        mag2 = np.sqrt(dx2**2 + dy2**2)
        
        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0
        
        # Calculate dot product and normalize by magnitudes to get cosine of angle
        dot_product = dx1 * dx2 + dy1 * dy2
        colinearity = dot_product / (mag1 * mag2)
        
        return colinearity

    # ===== Optimization and estimation =====
    def estimate_vanishing_point(self, flow, visualize=False, ground_truth_point=None):
        """
        Estimate the vanishing point using the colinearity optimization.
        
        Args:
            flow (numpy.ndarray): Optical flow field
            visualize (bool): Whether to visualize the optimization process
            ground_truth_point (tuple): Optional ground truth point for visualization
            
        Returns:
            tuple: (x, y) coordinates of the estimated vanishing point
        """
        # Filtrer le flow avant l'estimation
        filtered_flow, weights = self.flow_filter.filter_and_weight(flow)
        
        # Find vanishing point using L-BFGS-B optimization
        vanishing_point = self.find_vanishing_point_lbfgsb(
            filtered_flow, 
            weights,
            visualize=visualize,
            ground_truth_point=ground_truth_point
        )
        
        # Correct the vanishing point if needed
        if self.use_max_distance:
            vanishing_point = self.correct_vanishing_point(vanishing_point)
        
        # Update previous point for next frame
        self.previous_vanishing_point = vanishing_point
        
        return vanishing_point
    
    def find_vanishing_point_lbfgsb(self, flow, weights=None, visualize=True, ground_truth_point=None):
        """
        Find the vanishing point using the L-BFGS-B algorithm
        
        Note: Cette méthode utilise maintenant LBFGSOptimizer du module core.optimizers
        pour une architecture plus cohérente et modulaire.
        
        Args:
            flow: Optical flow vector field of shape (h, w, 2)
            weights: Optional weight matrix of shape (h, w) to weight the importance of each flow vector
            visualize: Boolean indicating whether to visualize the results
            ground_truth_point: Optional ground truth point for visualization
            
        Returns:
            vanishing_point: Tuple (x, y) representing the estimated vanishing point
        """
        from src.core.optimizers import LBFGSOptimizer
        
        valid_pixels = (flow[:,:,0]**2 + flow[:,:,1]**2) > 0
        if np.sum(valid_pixels) == 0:
            #print("No valid pixels found")
            #print(f"returning previous vanishing point: {self.previous_vanishing_point}")
            return self.previous_vanishing_point
        
        # Choose starting point based on use_ray_optimization
        starting_point = self.previous_vanishing_point if self.use_reoptimization else self.default_vanishing_point
        
        # Create LBFGSOptimizer instance
        optimizer = LBFGSOptimizer(display_warnings=False)
        
        if visualize:
            # Points and scores for visualization
            all_points = [starting_point]
            all_scores = [self.objective_function(starting_point, flow, weights)]
            
            # Callback function to record the trajectory
            def callback(point):
                all_points.append(point.copy())
                all_scores.append(self.objective_function(point, flow, weights))
            
            # Use LBFGSOptimizer with callback for visualization
            vanishing_point = optimizer.optimize_single(
                self, flow, starting_point, weights, callback_fn=callback
            )
            
            self.visualize_gradient_descent(flow, all_points, all_scores, vanishing_point, ground_truth_point)
        else:
            # Use LBFGSOptimizer without visualization overhead
            vanishing_point = optimizer.optimize_single(
                self, flow, starting_point, weights
            )
        
        return vanishing_point
    
    def objective_function(self, point, flow, weights=None, step=10):
        """
        Calculate a global colinearity score for a candidate point
        
        Args:
            point: Tuple (x, y) representing the candidate point
            flow: Optical flow vector field of shape (h, w, 2)
            weights: Optional weight matrix of shape (h, w) to weight the importance of each flow vector
            step: Integer, compute colinearity every 'step' pixels for faster processing
            
        Returns:
            score: Numerical value, higher means better global colinearity
        """
        # Negative because we want to minimize in gradient descent
        return -self.colin_score(flow, point, step, weights)

    def compute_gradient(self, point, flow, epsilon=1.0, step=10):
        """
        Calculate the gradient of the objective function with respect to point coordinates
        
        Args:
            point: Tuple (x, y) representing the candidate point
            flow: Optical flow vector field
            epsilon: Small displacement for numerical approximation
            
        Returns:
            gradient: Tuple (dx, dy) representing the gradient
        """
        x, y = point
        
        # Evaluate the objective function at the current point
        f_current = self.objective_function(point, flow, step)
        
        # Evaluate the function after a small displacement in x
        f_x_plus = self.objective_function((x + epsilon, y), flow, step)
        
        # Evaluate the function after a small displacement in y
        f_y_plus = self.objective_function((x, y + epsilon), flow, step)
        
        # Calculate partial derivatives
        df_dx = (f_x_plus - f_current) / epsilon
        df_dy = (f_y_plus - f_current) / epsilon
        
        return (df_dx, df_dy)

    # ===== Visualization and drawing =====
    def visualize_gradient_descent(self, flow, trajectory, scores, final_point, ground_truth_point):
        """
        Visualize the gradient descent results
        
        Args:
            flow: Optical flow vector field
            trajectory: List of points visited during descent
            scores: List of scores at each iteration
            final_point: Final point (x, y)
            ground_truth_point: Optional ground truth point for visualization
        """
        h, w = flow.shape[:2]
        
        plt.figure(figsize=(15, 10))
        
        # Plot score evolution
        plt.subplot(2, 2, 1)
        plt.plot([-s for s in scores])
        plt.title('Score Evolution')
        plt.xlabel('Iterations')
        plt.ylabel('Score (higher = better)')
        plt.grid(True)
        
        # Visualize optical flow field
        plt.subplot(2, 2, 2)
        
        # Create a grid for flow visualization
        Y, X = np.mgrid[0:h:20, 0:w:20]
        U = flow[::20, ::20, 0]
        V = flow[::20, ::20, 1]
        
        plt.imshow(np.zeros((h, w, 3)), cmap='gray')
        plt.quiver(X, Y, U, -V, color='w', scale=50)
        if ground_truth_point:
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='green', s=100, marker='o', label='Ground truth')
        plt.title('Optical Flow')
        
        # Visualize colinearity map for final point
        plt.subplot(2, 2, 3)
        colinearity_map = self.compute_colinearity_map(flow, final_point)
        plt.imshow(colinearity_map, cmap='hot')
        plt.colorbar(label='Colinearity')
        if ground_truth_point:
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='green', s=100, marker='o', label='Ground truth')
        plt.title('Colinearity Map for Final Point')
        
        # Visualize descent trajectory
        plt.subplot(2, 2, 4)
        plt.imshow(np.zeros((h, w, 3)))
        
        # Convert list of tuples to numpy array for easier plotting
        trajectory_array = np.array(trajectory)
        plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'r.-', linewidth=2)
        plt.scatter(final_point[0], final_point[1], c='g', s=100, marker='*')
        if ground_truth_point:
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='green', s=100, marker='o', label='Ground truth')
        
        # Add annotations for first points
        for i, (x, y) in enumerate(trajectory[:min(5, len(trajectory))]):
            plt.annotate(f"{i}", (x, y), fontsize=12, color='white')
        
        plt.title('Gradient Descent Trajectory')
        plt.xlim(0, w)
        plt.ylim(h, 0)  # Invert y-axis to match image coordinates
        if ground_truth_point:
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='green', s=100, marker='o', label='Ground truth')
        
        plt.tight_layout()
        plt.show()

    def draw_vanishing_point_zone(self, frame):
        """
        Draw a red circle representing the legal zone for vanishing points.
        
        Args:
            frame (numpy.ndarray): The frame to draw on
        """
        if self.use_max_distance:
            cv2.circle(frame, (self.center_x, self.center_y), self.max_distance, (0, 0, 255), 2)

    # ===== Utility methods =====
    def correct_vanishing_point(self, vanishing_point):
        """
        Correct the vanishing point position if it's too far from the center.
        
        Args:
            vanishing_point (tuple): (x, y) coordinates of the vanishing point
            
        Returns:
            tuple: Corrected (x, y) coordinates of the vanishing point
        """
        vp_x, vp_y = vanishing_point
        
        # Calculate the distance from the center
        dx = vp_x - self.center_x
        dy = vp_y - self.center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # If the distance is greater than max_distance pixels, use the center point
        if distance > self.max_distance:
            return (self.center_x, self.center_y)
        
        return vanishing_point

    def get_angles(self, vanishing_point):
        """
        Convert vanishing point coordinates to angles.
        
        Args:
            vanishing_point (tuple): (x, y) coordinates of the vanishing point
            
        Returns:
            tuple: (yaw, pitch) angles in radians
        """
        return pixels_to_angles(
            vanishing_point[0], 
            vanishing_point[1], 
            self.focal_length, 
            self.frame_width, 
            self.frame_height
        )