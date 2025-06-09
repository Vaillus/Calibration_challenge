import numpy as np
import matplotlib.pyplot as plt
import cv2

import mlx.core as mx

from src.utilities.pixel_angle_converter import pixels_to_angles
from src.core.flow_filter import FlowFilterSample
from src.utilities.project_constants import get_project_constants

class VanishingPointEstimator:
    # ===== Core initialization and configuration =====
    def __init__(self):
        """
        Initialize the VanishingPointEstimator.
        
        Note: Cette classe utilise maintenant les optimiseurs centralisés du module 
        core.optimizers pour L-BFGS-B. Pour Adam, utilisez AdamOptimizer directement.
        """
        project_constants = get_project_constants()
        self.frame_width = project_constants["frame_width"]
        self.frame_height = project_constants["frame_height"]
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Default vanishing point is the center of the image
        self.default_vanishing_point = (self.center_x, self.center_y)
        
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

    # ==== Optimization and estimation =====
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
        
        return vanishing_point

