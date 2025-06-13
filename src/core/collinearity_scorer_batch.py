import numpy as np
import mlx.core as mx
import psutil
from pathlib import Path
import gc
import time
from src.core.flow_filter import FlowFilterBatch
from src.utilities.project_constants import get_project_constants

class BatchCollinearityScorer:
    """
    Batch version of CollinearityScorer using MLX for efficient batch processing.

    This class provides optimized methods for computing collinearity scores and maps
    for multiple frames simultaneously using MLX's parallel processing capabilities.
    It is particularly efficient on Apple Silicon chips.

    Key features:
    - Batch processing of multiple frames
    - MLX-optimized vectorized operations
    - Pre-computed coordinate grids for efficiency
    - Memory-efficient processing with configurable batch sizes

    Example:
        >>> scorer = BatchCollinearityScorer()
        >>> flow_batch = mx.array(np.random.rand(10, 1080, 1920, 2))  # Batch of 10 flows
        >>> points = mx.array([[960, 540] for _ in range(10)])  # Batch of 10 points
        >>> scores = scorer.colin_score_batch(flow_batch, points)
    """
    
    def __init__(self):
        """
        Initialize the BatchCollinearityScorer.
        """
        project_constants = get_project_constants()
        self.frame_width = project_constants["frame_width"]
        self.frame_height = project_constants["frame_height"]
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        
        # Default reference point is the center of the image
        self.default_reference_point = (self.center_x, self.center_y)
        
        # Pre-compute coordinate grids for parallel computation
        self._init_coordinate_grids()

    def _init_coordinate_grids(self):
        """Initialize coordinate grids for parallel computation."""
        # Create base coordinate grids
        self.y_coords_base = mx.arange(0, self.frame_height)[:, None]  # Shape: (h, 1)
        self.x_coords_base = mx.arange(0, self.frame_width)[None, :]   # Shape: (1, w)

    def _validate_input_types(self, flows=None, pts=None, weights=None):
        """
        Simple type validation for inputs.
        
        Args:
            flow: Single flow array (should be MLX array float32)
            pt: Single point array (should be MLX array float32)
            weights: Weight array (should be MLX array float32)
            flows: Batch of flows (should be MLX array)
            points: Batch of points (should be MLX array)
        """
        if flows is not None:
            assert isinstance(flows, mx.array) and flows.dtype == mx.float32, \
                "flows must be MLX array of type float32"
        if pts is not None:
            assert isinstance(pts, mx.array) and pts.dtype == mx.float32, \
                "pts must be MLX array of type float32"
        if weights is not None:
            assert isinstance(weights, mx.array) and weights.dtype == mx.float32, \
                "weights must be MLX array of type float32"

    def _validate_input_dimensions(self, flows=None, pts=None, weights=None, batch_mode=False):
        """
        Validate input dimensions for both single and batch modes.
        
        Args:
            flows: Flow array(s) - single (h, w, 2) or batch (batch_size, h, w, 2)
            pts: Point array(s) - single (2,) or batch (batch_size, 2)
            weights: Weight array(s) - single (h, w) or batch (h, w) or (batch_size, h, w)
            batch_mode: Whether to validate for batch processing
        """
        if batch_mode:
            # Batch mode validation
            if flows is not None:
                assert flows.ndim == 4, "flows must be 4D array (batch_size, h, w, 2)"
                assert flows.shape[1:3] == (self.frame_height, self.frame_width), \
                    f"flow shape {flows.shape[1:3]} doesn't match frame dimensions {(self.frame_height, self.frame_width)}"
            
            if pts is not None:
                assert pts.ndim == 2 and pts.shape[1] == 2, "pts must be 2D array (batch_size, 2)"
            
            if weights is not None:
                if weights.ndim == 2:
                    assert weights.shape == (self.frame_height, self.frame_width), \
                        f"weights shape {weights.shape} doesn't match frame dimensions"
                elif weights.ndim == 3:
                    assert weights.shape[1:] == (self.frame_height, self.frame_width), \
                        f"weights shape {weights.shape[1:]} doesn't match frame dimensions"
                else:
                    raise ValueError(f"weights must be 2D or 3D, got {weights.ndim}D")
        else:
            # Single mode validation
            if flows is not None:
                assert flows.shape[:2] == (self.frame_height, self.frame_width), \
                    f"flow shape {flows.shape[:2]} doesn't match frame dimensions {(self.frame_height, self.frame_width)}"
            
            if pts is not None:
                assert pts.shape == (2,), f"pt must have shape (2,), got {pts.shape}"
            
            if weights is not None:
                assert weights.shape == (self.frame_height, self.frame_width), \
                    f"weights shape {weights.shape} doesn't match frame dimensions"

    # ==== SCORE FUNCTIONS ====
    def colin_score(self, flow, pt, weights=None):
        """
        Gradient-compatible version of colin_score using MLX.
        This version is optimized for gradient computation and mirrors the logic of colin_score_batch but for a single sample.
        
        Args:
            flow: Optical flow field of shape (h, w, 2) - MLX array (float32)
            pt: Point coordinates of shape (2,) - MLX array with (x, y) (float32)
            weights: Optional weight matrix of shape (h, w) - MLX array (float32)
            
        Returns:
            mlx.core.array: Scalar collinearity score
        """
        # Validate inputs
        self._validate_input_types(flows=flow, pts=pt, weights=weights)
        self._validate_input_dimensions(flows=flow, pts=pt, weights=weights, batch_mode=False)
            
        # Get colinearity map using the dedicated method
        colinearity = self.compute_colinearity_map(flow, pt)
        
        # Get flow vectors for valid pixels mask
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        valid_pixels = (flow_x**2 + flow_y**2) > 0
        
        # Compute final score
        eps = 1e-8
        if weights is not None:
            # Use weights to weight the colinearity scores
            weighted_sum = mx.sum(colinearity * valid_pixels * weights)
            total_weight = mx.sum(valid_pixels * weights)
            score = weighted_sum / (total_weight + eps)
        else:
            # Original behavior without weights
            valid_sum = mx.sum(colinearity * valid_pixels)
            total_valid = mx.sum(valid_pixels)
            score = valid_sum / (total_valid + eps)
        
        return score
    
    def compute_colinearity_map(self, flow, pt):
        """
        Parallel version of compute_colinearity_map using MLX for Apple Silicon optimization.
        This version is compatible with automatic differentiation.
        
        Args:
            flow: Optical flow field of shape (h, w, 2) - MLX array (float32)
            pt: Point coordinates of shape (2,) - MLX array with (x, y) (float32)
            
        Returns:
            mlx.core.array: A 2D array of collinearity values between -1 and 1
        """
        # Validate inputs
        self._validate_input_types(flows=flow, pts=pt)
        self._validate_input_dimensions(flows=flow, pts=pt, batch_mode=False)
        
        # Get flow vectors
        flow_x = flow[:, :, 0]  # Shape: (h, w)
        flow_y = flow[:, :, 1]  # Shape: (h, w)
        
        # Create pixel vectors using pre-computed grids
        pixel_x = self.x_coords_base - pt[0]
        pixel_y = self.y_coords_base - pt[1]
        
        # Compute magnitudes and dot product with epsilon for gradient stability
        eps = 1e-8
        flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
        pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
        dot_product = flow_x * pixel_x + flow_y * pixel_y
        
        # Compute colinearity with same division by zero handling as sequential
        mask = (flow_mag > 0) & (pixel_mag > 0)
        colinearity_map = mx.where(
            mask,
            dot_product / (flow_mag * pixel_mag + eps),
            mx.zeros_like(dot_product)
        )
        
        return colinearity_map  # Return MLX array directly

    def colin_score_batch(self, flows:mx.array, points:mx.array, weights:mx.array=None, chunk_size:int=30):
        """
        Parallel version of colin_score that processes multiple flows and points in batch.
        Uses optimized chunking for better memory management and performance.
        
        Args:
            flows: Batch of optical flow fields of shape (batch_size, h, w, 2)
                  Can be either numpy array or MLX array
            points: Batch of points of shape (batch_size, 2) where each point is (x, y)
                   Can be either numpy array or MLX array
            weights: Optional weight matrix of shape (h, w) or (batch_size, h, w)
                    Can be either numpy array or MLX array.
                    These weights are used to give more importance to certain pixels
                    when computing the collinearity score. Higher weights mean the pixel's
                    collinearity will have more impact on the final score.
            chunk_size: Size of chunks to process in parallel (default: 30)
            
        Returns:
            mlx.core.array: Array of scores of shape (batch_size,)
        """
        # Validate inputs
        self._validate_input_types(flows=flows, pts=points, weights=weights)
        self._validate_input_dimensions(flows=flows, pts=points, weights=weights, batch_mode=True)
        
        batch_size = flows.shape[0]
        all_scores = mx.zeros(batch_size)
        
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            
            flows_chunk, points_chunk, weights_chunk = self._get_chunk(flows, points, weights, start_idx, end_idx)
            
            # Compute colinearity map for the chunk using compute_colinearity_map_batch
            colinearity_map = self.compute_colinearity_map_batch(flows_chunk, points_chunk)
            
            # Create valid pixels mask (non-zero flow vectors)
            flow_x = flows_chunk[:, :, :, 0]
            flow_y = flows_chunk[:, :, :, 1]
            valid_pixels = (flow_x**2 + flow_y**2) > 0
            
            eps = 1e-8
            if weights_chunk is not None:
                # Use weights to weight the colinearity scores
                weighted_sum = mx.sum(colinearity_map * valid_pixels * weights_chunk, axis=(1, 2))
                total_weight = mx.sum(valid_pixels * weights_chunk, axis=(1, 2))
                scores = weighted_sum / (total_weight + eps)
            else:
                # Original behavior without weights
                valid_sum = mx.sum(colinearity_map * valid_pixels, axis=(1, 2))
                total_valid = mx.sum(valid_pixels, axis=(1, 2))
                scores = valid_sum / (total_valid + eps)
            
            # Store results
            all_scores[start_idx:end_idx] = scores
            
            # Force garbage collection
            gc.collect()
        
        return all_scores
    
    def _get_chunk(self, flows, points, weights, start_idx, end_idx):
        """
        Get a chunk of flows, points, and weights for batch processing.
        
        Args:
            flows: Full batch of flows
            points: Full batch of points
            weights: Full batch of weights
            start_idx: Start index for chunk
            end_idx: End index for chunk
            
        Returns:
            tuple: (flows_chunk, points_chunk, weights_chunk)
        """
        if weights is not None:
            if weights.ndim == 2:
                weights_chunk = weights[None, :, :]
            else:
                weights_chunk = weights[start_idx:end_idx]
        else:
            weights_chunk = None
        return flows[start_idx:end_idx], points[start_idx:end_idx], weights_chunk

    def compute_colinearity_map_batch(self, flows, points):
        """
        Batch version of compute_colinearity_map.
        Args:
            flows: MLX array of shape (batch_size, H, W, 2)
            points: MLX array of shape (batch_size, 2)
        Returns:
            MLX array of shape (batch_size, H, W) with colinearity values between -1 and 1
        """
        # Validate inputs
        self._validate_input_types(flows=flows, pts=points)
        self._validate_input_dimensions(flows=flows, pts=points, batch_mode=True)
        
        batch_size, h, w, _ = flows.shape
        # Get flow vectors
        flow_x = flows[:, :, :, 0]  # (batch, h, w)
        flow_y = flows[:, :, :, 1]  # (batch, h, w)
        # Broadcast pixel vectors for each sample
        # x_coords_base: (1, w), y_coords_base: (h, 1)
        pixel_x = self.x_coords_base[None, :, :] - points[:, 0, None, None]  # (batch, h, w)
        pixel_y = self.y_coords_base[None, :, :] - points[:, 1, None, None]  # (batch, h, w)
        # Compute magnitudes and dot product
        eps = 1e-8
        flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
        pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
        dot_product = flow_x * pixel_x + flow_y * pixel_y
        # Compute colinearity with mask
        mask = (flow_mag > 0) & (pixel_mag > 0)
        colinearity_map = mx.where(
            mask,
            dot_product / (flow_mag * pixel_mag + eps),
            mx.zeros_like(dot_product)
        )
        return colinearity_map


