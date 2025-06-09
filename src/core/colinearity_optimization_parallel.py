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


    # ==== SCORE FUNCTIONS ====
    def colin_score(self, flow, pt, step=1, weights=None):
        """
        Gradient-compatible version of colin_score using MLX.
        This version is optimized for gradient computation and mirrors the logic
        of colin_score_batch but for a single sample.
        
        Args:
            flow: Optical flow field of shape (h, w, 2) - MLX array (float32)
            pt: Point coordinates of shape (2,) - MLX array with (x, y) (float32)
            weights: Optional weight matrix of shape (h, w) - MLX array (float32)
            
        Returns:
            mlx.core.array: Scalar collinearity score
        """
        # Ensure inputs are MLX arrays (assume float32)
        if not isinstance(flow, mx.array):
            flow = mx.array(flow, dtype=mx.float32)
        if not isinstance(pt, mx.array):
            pt = mx.array(pt, dtype=mx.float32)
        if weights is not None and not isinstance(weights, mx.array):
            weights = mx.array(weights, dtype=mx.float32)
            
        # Create coordinate grids (all float32)
        pixel_x = self.x_coords_base - pt[0]  # Shape: (h, w)
        pixel_y = self.y_coords_base - pt[1]  # Shape: (h, w)
        
        # Get flow vectors
        flow_x = flow[:, :, 0]  # Shape: (h, w)
        flow_y = flow[:, :, 1]  # Shape: (h, w)
        
        # Standard epsilon for float32
        eps = 1e-8
        
        # Compute magnitudes and dot product
        flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
        pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
        dot_product = flow_x * pixel_x + flow_y * pixel_y
        
        # Compute colinearity
        colinearity = dot_product / (flow_mag * pixel_mag)
        
        # Compute score using the same logic as colin_score_batch
        if weights is not None:
            # Pondération naturelle par la magnitude du flux + weights externes
            flow_importance = flow_mag / (flow_mag + eps)
            weighted_colinearity = colinearity * flow_importance * weights
            total_weight = mx.sum(flow_importance * weights)
            weighted_sum = mx.sum(weighted_colinearity)
            score = weighted_sum / (total_weight + eps)
        else:
            # Pas de weights externes, mais pondération naturelle par la magnitude du flux
            flow_importance = flow_mag / (flow_mag + eps)  # Remplace valid_pixels
            total_valid = mx.sum(flow_importance)
            weighted_sum = mx.sum(colinearity * flow_importance)
            score = weighted_sum / (total_valid + eps)
        
        return score
    
    def compute_colinearity_map(self, flow:np.ndarray, pt:tuple, step:int=1):
        """
        Parallel version of compute_colinearity_map using MLX for Apple Silicon optimization.
        
        Args:
            flow: Optical flow field of shape (h, w, 2)
            pt: Tuple (x, y) representing the reference point
            step: Integer, compute collinearity every 'step' pixels (ignored in parallel version)
            
        Returns:
            mlx.core.array: A 2D array of collinearity values between -1 and 1
        """
        # Check and convert inputs
        flow_mx, _ = self._check_and_convert_inputs(flow)
        pt_x, pt_y = pt
        
        # Get flow vectors
        flow_x = flow_mx[:, :, 0]  # Shape: (h, w)
        flow_y = flow_mx[:, :, 1]  # Shape: (h, w)
        
        # Create pixel vectors using pre-computed grids
        pixel_x = self.x_coords_base - pt_x
        pixel_y = self.y_coords_base - pt_y
        
        # Compute magnitudes and dot product
        flow_mag = mx.sqrt(flow_x**2 + flow_y**2)
        pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2)
        dot_product = flow_x * pixel_x + flow_y * pixel_y
        
        # Compute colinearity (avoiding division by zero)
        mask = (flow_mag > 0) & (pixel_mag > 0)
        colinearity = mx.where(
            mask,
            dot_product / (flow_mag * pixel_mag),
            mx.zeros_like(dot_product)
        )
        
        return colinearity  # Return MLX array directly

    def _check_and_convert_inputs(self, flow, weights=None):
        """
        Check input types and convert to MLX arrays if necessary.
        
        Args:
            flow: Optical flow field (numpy array or MLX array)
            weights: Optional weight matrix (numpy array or MLX array)
            
        Returns:
            tuple: (flow_mx, weights_mx) where weights_mx is None if weights is None
        """
        # Check flow type and shape
        assert isinstance(flow, (np.ndarray, mx.array)), "flow must be numpy array or MLX array"
        if isinstance(flow, np.ndarray):
            flow_mx = mx.array(flow)
        else:
            flow_mx = flow

        assert flow_mx.shape[:2] == (self.frame_height, self.frame_width), \
            f"flow shape {flow_mx.shape[:2]} doesn't match frame dimensions {(self.frame_height, self.frame_width)}"

        # Check weights if provided
        weights_mx = None
        if weights is not None:
            assert isinstance(weights, (np.ndarray, mx.array)), "weights must be numpy array or MLX array"
            if isinstance(weights, np.ndarray):
                weights_mx = mx.array(weights)
            else:
                weights_mx = weights
            assert weights_mx.shape == (self.frame_height, self.frame_width), \
                f"weights shape {weights_mx.shape} doesn't match frame dimensions"

        return flow_mx, weights_mx

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
        # Check input types
        if not isinstance(flows, mx.array):
            raise TypeError(f"flows must be an MLX array, got {type(flows)}")
        if not isinstance(points, mx.array):
            raise TypeError(f"points must be an MLX array, got {type(points)}")
        if weights is not None and not isinstance(weights, mx.array):
            raise TypeError(f"weights must be an MLX array or None, got {type(weights)}")
            
        # Check shapes
        assert flows.ndim == 4, "flows must be 4D array (batch_size, h, w, 2)"
        assert points.ndim == 2 and points.shape[1] == 2, "points must be 2D array (batch_size, 2)"
        assert flows.shape[1:3] == (self.frame_height, self.frame_width), \
            f"flow shape {flows.shape[1:3]} doesn't match frame dimensions {(self.frame_height, self.frame_width)}"
        
        batch_size = flows.shape[0]
        all_scores = mx.zeros(batch_size)
        
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            
            flows_chunk, points_chunk, weights_chunk = self._get_chunk(flows, points, weights, start_idx, end_idx)
            
            # Create coordinate grids for current chunk
            pixel_x = self.x_coords_base[None, :, :] - points_chunk[:, 0, None, None]
            pixel_y = self.y_coords_base[None, :, :] - points_chunk[:, 1, None, None]
            
            # Get flow vectors
            flow_x = flows_chunk[:, :, :, 0]
            flow_y = flows_chunk[:, :, :, 1]
            
            eps = 1e-8
            # Compute magnitudes and dot product
            flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
            pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
            dot_product = flow_x * pixel_x + flow_y * pixel_y
            
            # Compute colinearity
            
            colinearity = dot_product / (flow_mag * pixel_mag)
            
            # Calculate valid pixels mask using flow_mag directly
            # valid_pixels = (flow_mag > 0) & (pixel_mag > 0)
            
            # Compute scores for current chunk
            if weights_chunk is not None:
                # Pondération naturelle par la magnitude du flux
                flow_importance = flow_mag / (flow_mag + eps)
                weighted_colinearity = colinearity * flow_importance * weights_chunk
                total_weight = mx.sum(flow_importance * weights_chunk, axis=(1, 2))
                weighted_sum = mx.sum(weighted_colinearity, axis=(1, 2))
                scores = weighted_sum / (total_weight + eps)
            else:
                # Cas sans weights - pondération naturelle par la magnitude du flux
                # flow_importance = flow_mag / (flow_mag + eps)
                # weighted_colinearity = colinearity * flow_importance
                # total_weight = mx.sum(flow_importance, axis=(1, 2))
                # weighted_sum = mx.sum(weighted_colinearity, axis=(1, 2))
                # scores = weighted_sum / (total_weight + eps)
                # Pas de weights externes, mais on doit quand même exclure les pixels sans mouvement
                flow_importance = flow_mag / (flow_mag + eps)  # Remplace valid_pixels
                total_valid = mx.sum(flow_importance, axis=(1, 2))
                weighted_sum = mx.sum(colinearity * flow_importance, axis=(1, 2))
                scores = weighted_sum / (total_valid + eps)
            
            # Store results
            all_scores[start_idx:end_idx] = scores
            
            # Clear MLX memory
            # del flows_chunk, points_chunk, pixel_x, pixel_y, flow_x, flow_y, flow_mag, pixel_mag
            # del dot_product, mask, colinearity, valid_pixels, scores
            # if weights_chunk is not None:
                # del weights_chunk
            
            # Force garbage collection
            gc.collect()
        
        return all_scores
    
    def colin_score_batch_simple(self, flows, points, weights=None, chunk_size=30): 
        """
        Simplified version of colin_score_batch that processes all samples at once.
        Faster but uses more memory.
        
        Args:
            flows: Batch of optical flow fields of shape (batch_size, h, w, 2)
            points: Batch of points of shape (batch_size, 2)
            weights: Optional weight matrix of shape (h, w) or (batch_size, h, w)
            chunk_size: Ignored in this version
            
        Returns:
            mlx.core.array: Array of scores of shape (batch_size,)
        """
        batch_size = flows.shape[0]
        
        # Coordonnées
        pixel_x = self.x_coords_base[None, :, :] - points[:, 0, None, None]
        pixel_y = self.y_coords_base[None, :, :] - points[:, 1, None, None]
        
        # Flux
        flow_x = flows[:, :, :, 0]
        flow_y = flows[:, :, :, 1]
        
        # Ajouter les magnitudes
        eps = 1e-8
        flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
        pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
        
        # Dot product
        dot_product = flow_x * pixel_x + flow_y * pixel_y
        
        # Test avec division simple
        colinearity = dot_product / (flow_mag * pixel_mag)
        scores = mx.sum(colinearity, axis=(1, 2))
        
        return scores
    
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

    # =============================================================================
    # L-BFGS-B + MLX IMPLEMENTATION ABANDONED
    # =============================================================================
    # 
    # POURQUOI L-BFGS-B + MLX N'EST PAS IMPLÉMENTÉ :
    #
    # 1. INCOMPATIBILITÉ FONDAMENTALE :
    #    - scipy.optimize.minimize requiert des functions/gradients NumPy
    #    - MLX utilise ses propres arrays et gradients automatiques
    #    - Aucune passerelle native entre scipy et MLX
    #
    # 2. PROBLÈME DE PERFORMANCE :
    #    - Les conversions MLX ↔ NumPy annulent les gains de performance
    #    - Créer des estimateurs temporaires à chaque évaluation est très lent
    #    - Le benchmark montre 47s/frame vs 0.47s pour L-BFGS-B original
    #
    # 3. COMPLEXITÉ D'IMPLÉMENTATION :
    #    - Implémenter L-BFGS-B from scratch en MLX serait très complexe
    #    - Gestion manuelle de la matrice hessienne, line search, etc.
    #    - Maintenance et debugging difficiles
    #
    # 4. ALTERNATIVES MEILLEURES :
    #    - Adam MLX est 9x plus rapide et utilise pleinement MLX
    #    - L-BFGS-B original reste le plus précis pour une frame unique
    #    - Trade-off vitesse/précision déjà optimal avec ces deux options
    #
    # CONCLUSION : Nous gardons deux optimiseurs spécialisés :
    # - L-BFGS-B original (scipy) : précision maximale, 1 frame à la fois
    # - Adam MLX : vitesse maximale, support batch naturel
    # =============================================================================

if __name__ == "__main__":
    from ground_truth import read_ground_truth_pixels
    from core.flow_filter import FlowFilterBatch
    import time
    import psutil
    import numpy as np
    import gc
    
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Load ground truth pixels directly as MLX array
    gt_pixels = mx.array(read_ground_truth_pixels(0)[1:])
    mx.eval(gt_pixels)  # Force evaluation of ground truth pixels
    print(f"GT pixels shape: {gt_pixels.shape}")
    print(f"Initial memory usage: {get_memory_usage():.1f} MB")
    
    # Initialize estimator
    pve = BatchCollinearityScorer()
    
    # Initialize filter
    filter_config = {
        'filtering': {
            'norm': {'is_used': True, 'min_threshold': 1e-2}
        },
        'weighting': {
            'norm': {'is_used': True, 'type': 'linear'}
        }
    }
    flow_filter = FlowFilterBatch(filter_config)
    
    # Process flows in batches
    batch_size = 100  # Reduced batch size to prevent memory accumulation
    total_flows = 1199  # Total number of flows
    all_scores = []
    
    
    
    # Load NPZ file
    npz_path = Path("calib_challenge/flows/0_float16.npz")
    print(f"\nLoading NPZ file: {npz_path}")
    with np.load(npz_path) as data:
        flows_data = mx.array(data['flow'])
    print(f"Loaded MLX array of shape {flows_data.shape}")
    start_time = time.time()
    for start_idx in range(0, total_flows, batch_size):
        end_idx = min(start_idx + batch_size, total_flows)
        print(f"\nProcessing flows {start_idx} to {end_idx-1}")
        
        # Get batch directly from MLX array
        flows_batch = flows_data[start_idx:end_idx]
        # mx.eval(flows_batch)  # Force evaluation of loaded batch
        print(f"Loaded batch shape: {flows_batch.shape}")
        print(f"Memory after loading batch: {get_memory_usage():.1f} MB")
        
        # Filter flows
        filtered_flows, weights = flow_filter.filter_and_weight(flows_batch)
        # mx.eval(filtered_flows)  # Force evaluation of filtered flows
        # mx.eval(weights)  # Force evaluation of weights
        print(f"Memory after filtering: {get_memory_usage():.1f} MB")
        
        # Get corresponding ground truth pixels
        gt_pixels_batch = gt_pixels[start_idx:end_idx]
        # mx.eval(gt_pixels_batch)  # Force evaluation of ground truth batch
        
        # Compute scores for this batch
        batch_scores = pve.colin_score_batch(filtered_flows, gt_pixels_batch, weights=weights, chunk_size=30)
        # mx.eval(batch_scores)  # Force evaluation of scores
        all_scores.extend(np.array(batch_scores))
        
        # Clear memory
        del flows_batch, filtered_flows, weights, gt_pixels_batch, batch_scores
        gc.collect()
        # mx.clear_cache()  # Clear GPU memory cache
        print(f"Memory after clearing batch: {get_memory_usage():.1f} MB")
    
    # Clean up
    del flows_data
    gc.collect()
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per flow: {total_time/total_flows:.3f} seconds")
    print(f"Final mean score: {np.mean(all_scores):.4f}")
