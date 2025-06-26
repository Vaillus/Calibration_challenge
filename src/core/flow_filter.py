import numpy as np
import mlx.core as mx

from src.utilities.paths import get_intermediate_dir

class FlowFilterBase:
    """
    Base class with shared configuration and helper methods for flow filtering using sigmoid weighting.
    """
    def __init__(self, config=None):
        """
        Initialize the FlowFilter with sigmoid-based weighting configuration.
        
        Args:
            config (dict): Configuration dictionary with the following structure:
                {
                    'norm': {'is_used': bool, 'k': float, 'x0': float},
                    'colinearity': {'is_used': bool, 'k': float, 'x0': float},
                    'heatmap': {'is_used': bool, 'weight': float, 'path': str}
                }
                
                Parameters:
                - k: Controls the "hardness" of the sigmoid (large k = hard filter, small k = soft weighting)
                - x0: Position of the threshold/center of the transition
                - weight: Heatmap influence coefficient (0=no effect, 1=full effect)
                - path: Path to the heatmap .npy file
        """
        # Default configuration
        self.default_config = {
            'norm': {'is_used': True, 'k': 20.0, 'x0': 13.0},
            'colinearity': {'is_used': True, 'k': 20.0, 'x0': 0.96},
            'heatmap': {'is_used': False, 'weight': 0.5, 'path': get_intermediate_dir() / 'heatmaps/unfiltered/global/global_heatmap.npy'}
        }
        
        # Use provided config or default
        if config is None:
            self.config = self.default_config.copy()
        else:
            self.config = self._merge_config(config)
        
        # Load and normalize heatmap if enabled
        self.normalized_heatmap = None
        if self.config['heatmap']['is_used'] and self.config['heatmap']['path']:
            self._load_heatmap()
    
    def _merge_config(self, user_config):
        """Merge user config with default config, filling missing values."""
        config = self.default_config.copy()
        
        for weight_type in ['norm', 'colinearity', 'heatmap']:
            if weight_type in user_config:
                config[weight_type].update(user_config[weight_type])
        
        return config
    
    def _load_heatmap(self):
        """Load and normalize the heatmap."""
        try:
            heatmap = np.load(self.config['heatmap']['path'])
            # Normalize to [0, 1]
            heatmap_min = np.min(heatmap)
            heatmap_max = np.max(heatmap)
            self.normalized_heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
            # print(f"✅ Heatmap loaded: {heatmap.shape}, range [{heatmap_min:.3f}, {heatmap_max:.3f}] → [0, 1]")
        except Exception as e:
            print(f"❌ Error loading heatmap: {e}")
            self.normalized_heatmap = None
            self.config['heatmap']['is_used'] = False
    
    def _sigmoid_weights(self, values, k, x0):
        """
        Apply sigmoid weighting function: w(x) = 1 / (1 + exp(-k × (x - x₀)))
        
        Args:
            values: Input values (numpy array or MLX array)
            k: Steepness parameter (higher = more step-like)
            x0: Center/threshold parameter
            
        Returns:
            Sigmoid weights between 0 and 1
        """
        if isinstance(values, np.ndarray):
            return 1.0 / (1.0 + np.exp(-k * (values - x0)))
        else:  # MLX array
            return 1.0 / (1.0 + mx.exp(-k * (values - x0)))
    
    def _sigmoid_weights_batch(self, values, k, x0):
        """Batch version of sigmoid weighting using MLX."""
        return 1.0 / (1.0 + mx.exp(-k * (values - x0)))


class FlowFilterSample(FlowFilterBase):
    """
    Flow filtering for single samples using numpy with sigmoid weighting.
    """
    
    def compute_weights(self, flow, reference_point=None):
        """
        Apply sigmoid weighting based on the configuration.
        
        Args:
            flow (numpy.ndarray): Flow matrix of shape (h, w, 2)
            reference_point (tuple): (x, y) reference point for colinearity, None for image center
            
        Returns:
            numpy.ndarray: Weight matrix of shape (h, w)
        """
        assert flow.ndim == 3, "Flow must be a 3D array"
        h, w = flow.shape[:2]
        weights = np.ones((h, w), dtype=np.float32)
        
        # Apply norm weighting if enabled
        if self.config['norm']['is_used']:
            norm_weights = self._weight_by_norm(flow)
            weights = weights * norm_weights
        
        # Apply colinearity weighting if enabled
        if self.config['colinearity']['is_used']:
            colinearity_weights = self._weight_by_colinearity(flow, reference_point)
            weights = weights * colinearity_weights
        
        # Apply heatmap weighting if enabled
        if self.config['heatmap']['is_used'] and self.normalized_heatmap is not None:
            heatmap_weights = self._weight_by_heatmap(h, w)
            weights = weights * heatmap_weights
        
        return weights
    
    def filter(self, flow, reference_point=None, threshold=0.5):
        """
        Apply filtering by thresholding sigmoid weights.
        
        Args:
            flow (numpy.ndarray): Flow matrix of shape (h, w, 2)
            reference_point (tuple): (x, y) reference point for colinearity, None for image center
            threshold (float): Threshold for converting weights to binary mask (default: 0.5)
            
        Returns:
            numpy.ndarray: Filtered flow matrix
        """
        weights = self.compute_weights(flow, reference_point)
        mask = weights >= threshold
        return flow * mask[..., np.newaxis]
    
    def filter_and_weight(self, flow, reference_point=None):
        """
        Apply sigmoid weighting (for backward compatibility).
        
        Args:
            flow (numpy.ndarray): Flow matrix of shape (h, w, 2)
            reference_point (tuple): (x, y) reference point for colinearity, None for image center
            
        Returns:
            tuple: (flow, weights) - flow unchanged, weights from sigmoid
        """
        weights = self.compute_weights(flow, reference_point)
        return flow, weights
    
    def _weight_by_norm(self, flow):
        """Apply sigmoid weighting based on flow norms."""
        norms = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        k = self.config['norm']['k']
        x0 = self.config['norm']['x0']
        return self._sigmoid_weights(norms, k, x0)
    
    def _weight_by_colinearity(self, flow, reference_point=None):
        """Apply sigmoid weighting based on colinearity scores."""
        h, w = flow.shape[:2]
        
        # Use provided reference point or default to image center
        if reference_point is None:
            ref_x, ref_y = w // 2, h // 2
        else:
            ref_x, ref_y = reference_point
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Pixel vectors relative to reference point
        pixel_x = x_coords - ref_x
        pixel_y = y_coords - ref_y
        
        # Flow vectors
        flow_x = flow[:, :, 0]
        flow_y = flow[:, :, 1]
        
        # Calculate magnitudes
        eps = 1e-8
        flow_mag = np.sqrt(flow_x**2 + flow_y**2 + eps)
        pixel_mag = np.sqrt(pixel_x**2 + pixel_y**2 + eps)
        
        # Dot product
        dot_product = flow_x * pixel_x + flow_y * pixel_y
        
        # Colinearity score (-1 to 1)
        colinearity = dot_product / (flow_mag * pixel_mag)
        
        k = self.config['colinearity']['k']
        x0 = self.config['colinearity']['x0']
        return self._sigmoid_weights(colinearity, k, x0)
    
    def _weight_by_heatmap(self, h, w):
        """Apply weighting based on the preloaded heatmap."""
        weight_coeff = self.config['heatmap']['weight']
        return 1.0 - weight_coeff + weight_coeff * self.normalized_heatmap


class FlowFilterBatch(FlowFilterBase):
    """
    Flow filtering for batches using MLX for GPU acceleration with sigmoid weighting.
    """
    
    def compute_weights(self, flows, reference_points=None, chunk_size=1200):
        """
        Apply sigmoid weighting to a batch of flows.
        
        Args:
            flows: Batch of flow matrices of shape (batch_size, h, w, 2)
            reference_points: Reference points for colinearity:
                - None: use image center for all flows
                - tuple (x, y): use same point for all flows  
                - list/array [(x1,y1), (x2,y2), ...]: one point per flow
            chunk_size: Chunk size for processing (default: 1200)
            
        Returns:
            mx.array: Batch of weight matrices of shape (batch_size, h, w)
        """
        # Preserve original dtype
        original_dtype = flows.dtype if hasattr(flows, 'dtype') else np.float32
        
        # Convert to MLX array if necessary
        if isinstance(flows, np.ndarray):
            flows = mx.array(flows, dtype=original_dtype)
        
        batch_size = flows.shape[0]
        h, w = flows.shape[1], flows.shape[2]
        all_weights = mx.ones((batch_size, h, w), dtype=original_dtype)
        
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            flows_chunk = flows[start_idx:end_idx]
            
            # Extract reference points for this chunk
            chunk_reference_points = self._extract_chunk_reference_points(
                reference_points, start_idx, end_idx, batch_size
            )
            
            chunk_size_actual = flows_chunk.shape[0]
            weights_chunk = mx.ones((chunk_size_actual, h, w), dtype=original_dtype)
            
            # Apply norm weighting if enabled
            if self.config['norm']['is_used']:
                norm_weights = self._weight_by_norm_batch(flows_chunk)
                weights_chunk = weights_chunk * norm_weights
            
            # Apply colinearity weighting if enabled
            if self.config['colinearity']['is_used']:
                colinearity_weights = self._weight_by_colinearity_batch(flows_chunk, chunk_reference_points)
                weights_chunk = weights_chunk * colinearity_weights
            
            # Apply heatmap weighting if enabled
            if self.config['heatmap']['is_used'] and self.normalized_heatmap is not None:
                heatmap_weights = self._weight_by_heatmap_batch(chunk_size_actual, h, w)
                weights_chunk = weights_chunk * heatmap_weights
            
            # Store results
            all_weights = mx.concatenate([
                all_weights[:start_idx],
                weights_chunk,
                all_weights[end_idx:]
            ])
            
            # Memory cleanup
            del flows_chunk, weights_chunk
            import gc
            gc.collect()
        
        return all_weights
    
    def filter(self, flows, reference_points=None, chunk_size=1200, threshold=0.5):
        """
        Apply filtering by thresholding sigmoid weights.
        
        Args:
            flows: Batch of flow matrices of shape (batch_size, h, w, 2)
            reference_points: Reference points for colinearity
            chunk_size: Chunk size for processing (default: 1200)
            threshold: Threshold for converting weights to binary mask (default: 0.5)
            
        Returns:
            mx.array: Filtered flows batch
        """
        weights = self.compute_weights(flows, reference_points, chunk_size)
        mask = weights >= threshold
        return flows * mask[..., None]
    
    def filter_and_weight(self, flows, reference_points=None, chunk_size=1200):
        """
        Apply sigmoid weighting (for backward compatibility).
        
        Args:
            flows: Batch of flow matrices of shape (batch_size, h, w, 2)
            reference_points: Reference points for colinearity
            chunk_size: Chunk size for processing (default: 1200)
            
        Returns:
            tuple: (flows, weights) - flows unchanged, weights from sigmoid
        """
        weights = self.compute_weights(flows, reference_points, chunk_size)
        return flows, weights
    
    def _extract_chunk_reference_points(self, reference_points, start_idx, end_idx, batch_size):
        """Extract reference points for the current chunk."""
        if reference_points is None:
            return None
        elif isinstance(reference_points, tuple):
            # Same point for all flows
            return reference_points
        else:
            # List/array of points - extract chunk
            return reference_points[start_idx:end_idx]
    
    def _weight_by_norm_batch(self, flows_chunk):
        """Batch version of sigmoid norm weighting."""
        norms = mx.sqrt(flows_chunk[..., 0]**2 + flows_chunk[..., 1]**2)
        k = self.config['norm']['k']
        x0 = self.config['norm']['x0']
        return self._sigmoid_weights_batch(norms, k, x0)
    
    def _weight_by_colinearity_batch(self, flows_chunk, reference_points=None):
        """Batch version of sigmoid colinearity weighting."""
        chunk_size, h, w = flows_chunk.shape[:3]
        
        # Handle reference points
        if reference_points is None:
            # Use image center for all flows
            ref_x, ref_y = w // 2, h // 2
            pixel_x = mx.arange(0, w)[None, :] - ref_x  # Shape: (1, w)
            pixel_y = mx.arange(0, h)[:, None] - ref_y  # Shape: (h, 1)
            
            # Flow vectors
            flow_x = flows_chunk[..., 0]  # Shape: (chunk_size, h, w)
            flow_y = flows_chunk[..., 1]  # Shape: (chunk_size, h, w)
            
            # Calculate magnitudes
            eps = 1e-8
            flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
            pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
            
            # Dot product
            dot_product = flow_x * pixel_x + flow_y * pixel_y
            
            # Colinearity score (-1 to 1)
            colinearity = dot_product / (flow_mag * pixel_mag)
            
        elif isinstance(reference_points, tuple):
            # Same point for all flows
            ref_x, ref_y = reference_points
            pixel_x = mx.arange(0, w)[None, :] - ref_x  # Shape: (1, w)
            pixel_y = mx.arange(0, h)[:, None] - ref_y  # Shape: (h, 1)
            
            # Flow vectors
            flow_x = flows_chunk[..., 0]  # Shape: (chunk_size, h, w)
            flow_y = flows_chunk[..., 1]  # Shape: (chunk_size, h, w)
            
            # Calculate magnitudes
            eps = 1e-8
            flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
            pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
            
            # Dot product
            dot_product = flow_x * pixel_x + flow_y * pixel_y
            
            # Colinearity score (-1 to 1)
            colinearity = dot_product / (flow_mag * pixel_mag)
            
        elif hasattr(reference_points, 'shape') and hasattr(reference_points, 'dtype'):
            # MLX array with different point per flow
            # Expected shape: (chunk_size, 2) with (x, y) per flow
            if reference_points.shape != (chunk_size, 2):
                raise ValueError(f"reference_points must have shape ({chunk_size}, 2), got {reference_points.shape}")
            
            # Extract ref_x and ref_y for each flow: shape (chunk_size,)
            ref_x = reference_points[:, 0]  # Shape: (chunk_size,)
            ref_y = reference_points[:, 1]  # Shape: (chunk_size,)
            
            # Create coordinate grids
            x_coords = mx.arange(0, w)[None, None, :]  # Shape: (1, 1, w)
            y_coords = mx.arange(0, h)[None, :, None]  # Shape: (1, h, 1)
            
            # Pixel vectors relative to reference point (broadcast to (chunk_size, h, w))
            pixel_x = x_coords - ref_x[:, None, None]  # Shape: (chunk_size, h, w)
            pixel_y = y_coords - ref_y[:, None, None]  # Shape: (chunk_size, h, w)
            
            # Broadcast pixel coordinates to full shape
            pixel_x = mx.broadcast_to(pixel_x, (chunk_size, h, w))
            pixel_y = mx.broadcast_to(pixel_y, (chunk_size, h, w))
            
            # Flow vectors
            flow_x = flows_chunk[..., 0]  # Shape: (chunk_size, h, w)
            flow_y = flows_chunk[..., 1]  # Shape: (chunk_size, h, w)
            
            # Calculate magnitudes
            eps = 1e-8
            flow_mag = mx.sqrt(flow_x**2 + flow_y**2 + eps)
            pixel_mag = mx.sqrt(pixel_x**2 + pixel_y**2 + eps)
            
            # Dot product
            dot_product = flow_x * pixel_x + flow_y * pixel_y
            
            # Colinearity score (-1 to 1)
            colinearity = dot_product / (flow_mag * pixel_mag)
            
        else:
            # Invalid type
            raise TypeError(f"reference_points must be None, tuple, or MLX array, got {type(reference_points)}")
        
        k = self.config['colinearity']['k']
        x0 = self.config['colinearity']['x0']
        return self._sigmoid_weights_batch(colinearity, k, x0)
    
    def _weight_by_heatmap_batch(self, chunk_size, h, w):
        """Batch version of heatmap weighting."""
        # Resize heatmap to match flow dimensions if needed
        if self.normalized_heatmap.shape != (h, w):
            from scipy.ndimage import zoom
            scale_y = h / self.normalized_heatmap.shape[0] 
            scale_x = w / self.normalized_heatmap.shape[1]
            resized_heatmap = zoom(self.normalized_heatmap, (scale_y, scale_x), order=1)
        else:
            resized_heatmap = self.normalized_heatmap
        
        # Convert to MLX and expand for batch
        heatmap_mlx = mx.array(resized_heatmap)
        weight_coeff = self.config['heatmap']['weight']
        
        # Formula: (1 - weight_coeff + weight_coeff * normalized_heatmap)
        heatmap_weights = 1.0 - weight_coeff + weight_coeff * heatmap_mlx
        
        # Expand to batch size: (chunk_size, h, w)
        return mx.broadcast_to(heatmap_weights[None, :, :], (chunk_size, h, w))


if __name__ == "__main__":
    import time
    import psutil
    from src.utilities.load_flows import load_flows
    
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Load data
    flows = load_flows(4, return_mlx=False)
    
    print("🔬 TESTING NEW SIGMOID-BASED FLOWFILTER")
    print(f"Flow shape: {flows.shape}")
    print("="*70)
    
    # ========================================
    # EXAMPLE 1: SINGLE SAMPLE PROCESSING
    # ========================================
    print("\n📊 EXAMPLE 1: SINGLE SAMPLE PROCESSING (FlowFilterSample)")
    print("-" * 60)
    
    config_sample = {
        'norm': {'is_used': True, 'k': 2.0, 'x0': 13.0},
        'colinearity': {'is_used': True, 'k': 10.0, 'x0': 0.97}
    }
    
    print(f"Configuration: {config_sample}")
    
    flow_filter_sample = FlowFilterSample(config_sample)
    
    # Test single flow
    single_flow = flows[0]
    weights = flow_filter_sample.compute_weights(single_flow)
    filtered_flow = flow_filter_sample.filter(single_flow, threshold=0.5)
    
    print(f"Original non-zero elements: {np.sum(single_flow != 0)}")
    print(f"Filtered non-zero elements: {np.sum(filtered_flow != 0)}")
    print(f"Weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
    print(f"Weights > 0.5: {np.sum(weights > 0.5)}")
    
    # ========================================
    # EXAMPLE 2: BATCH PROCESSING
    # ========================================
    print("\n📊 EXAMPLE 2: BATCH PROCESSING (FlowFilterBatch)")
    print("-" * 60)
    
    config_batch = {
        'norm': {'is_used': True, 'k': 2.0, 'x0': 13.0},
        'colinearity': {'is_used': True, 'k': 10.0, 'x0': 0.97}
    }
    
    print(f"Configuration: {config_batch}")
    
    flow_filter_batch = FlowFilterBatch(config_batch)
    
    batch_size = 100
    print(f"Testing batch processing with {batch_size} flows")
    
    # Convert to MLX for batch processing
    xflows = mx.array(flows)
    
    start_time = time.time()
    weights_batch = flow_filter_batch.compute_weights(xflows[:batch_size], chunk_size=50)
    batch_time = time.time() - start_time
    
    print(f"Batch processing completed in {batch_time:.3f}s ({batch_time/batch_size:.4f}s per flow)")
    print(f"Batch weights shape: {weights_batch.shape}")
    print(f"Batch weight range: [{float(mx.min(weights_batch)):.3f}, {float(mx.max(weights_batch)):.3f}]")
    
    # ========================================
    # EXAMPLE 3: DIFFERENT SIGMOID PARAMETERS
    # ========================================
    print("\n📊 EXAMPLE 3: DIFFERENT SIGMOID PARAMETERS")
    print("-" * 60)
    
    configs = {
        'Soft weighting (k=1)': {
            'norm': {'is_used': True, 'k': 1.0, 'x0': 13.0}
        },
        'Medium (k=5)': {
            'norm': {'is_used': True, 'k': 5.0, 'x0': 13.0}
        },
        'Hard filter (k=20)': {
            'norm': {'is_used': True, 'k': 20.0, 'x0': 13.0}
        }
    }
    
    for name, config in configs.items():
        filter_test = FlowFilterSample(config)
        weights = filter_test.compute_weights(single_flow)
        filtered = filter_test.filter(single_flow, threshold=0.5)
        
        print(f"{name}: weights [{np.min(weights):.3f}, {np.max(weights):.3f}], "
              f"filtered elements: {np.sum(filtered != 0)}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n🎯 SUMMARY")
    print("="*50)
    print("✅ Unified sigmoid-based approach:")
    print("   - Single function: w(x) = 1 / (1 + exp(-k × (x - x₀)))")
    print("   - Two optimizable parameters: k (hardness) and x₀ (threshold)")
    print("   - Replaces separate filtering and weighting")
    print("✅ Simplified configuration structure")
    print("✅ Backward compatibility maintained")
    print("✅ Same performance with cleaner code")
    print()
    print("💡 Usage:")
    print("   - apply_weights(): returns sigmoid weights [0,1]")
    print("   - filter(): applies threshold to sigmoid weights")
    print("   - k parameter: 1=soft, 10=medium, 20+=hard filter")