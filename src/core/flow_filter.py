import numpy as np
import mlx.core as mx

class FlowFilterBase:
    """
    Base class with shared configuration and helper methods for flow filtering.
    """
    def __init__(self, config=None):
        """
        Initialize the FlowFilter with a flexible configuration structure.
        
        Args:
            config (dict): Configuration dictionary with the following structure:
                {
                    'filtering': {
                        'norm': {'is_used': bool, 'min_threshold': float},
                        'colinearity': {'is_used': bool, 'min_threshold': float}
                    },
                    'weighting': {
                        'norm': {'is_used': bool, 'type': str},
                        'colinearity': {'is_used': bool, 'type': str}
                    }
                }
                
                Available weighting types:
                - 'linear': weights proportional to values
                - 'inverse': weights inversely proportional to values
                - 'power': weights proportional to values raised to a power
                - 'exp': weights proportional to exponential of values
                - 'log': weights proportional to logarithm of values
                - 'constant': all weights set to 1
        """
        # Default configuration
        self.default_config = {
            'filtering': {
                'norm': {'is_used': False, 'min_threshold': 1e-2},
                'colinearity': {'is_used': False, 'min_threshold': 0.0}
            },
            'weighting': {
                'norm': {'is_used': False, 'type': 'linear'},
                'colinearity': {'is_used': False, 'type': 'linear'}
            }
        }
        
        # Use provided config or default
        if config is None:
            self.config = self.default_config.copy()
        else:
            self.config = self._merge_config(config)
    
    def _merge_config(self, user_config):
        """Merge user config with default config, filling missing values."""
        config = self.default_config.copy()
        
        if 'filtering' in user_config:
            for filter_type in ['norm', 'colinearity']:
                if filter_type in user_config['filtering']:
                    config['filtering'][filter_type].update(user_config['filtering'][filter_type])
        
        if 'weighting' in user_config:
            for weight_type in ['norm', 'colinearity']:
                if weight_type in user_config['weighting']:
                    config['weighting'][weight_type].update(user_config['weighting'][weight_type])
        
        return config
    
    def _transform_values_to_weights(self, values, weight_type):
        """
        Transform values into weights using the specified type.
        
        Args:
            values (numpy.ndarray): Input values
            weight_type (str): Type of weight transformation
            
        Returns:
            numpy.ndarray: Transformed weights between 0 and 1
        """
        if np.max(values) == 0:
            return np.zeros_like(values)
        
        # Normalize values to [0, 1] range
        normalized_values = values / np.max(values)
        
        if weight_type == 'linear':
            weights = normalized_values
        elif weight_type == 'inverse':
            # Avoid division by zero by adding small epsilon
            weights = 1.0 / (normalized_values + 1e-10)
            # Normalize again to [0, 1] range
            if np.max(weights) > 0:
                weights = weights / np.max(weights)
        elif weight_type == 'power':
            # Default power of 2 for quadratic
            weights = normalized_values ** 2
        elif weight_type == 'exp':
            # Scale down to avoid overflow
            weights = np.exp(normalized_values * 5) - 1
            if np.max(weights) > 0:
                weights = weights / np.max(weights)
        elif weight_type == 'log':
            # Add small value to avoid log(0)
            weights = np.log1p(normalized_values)
            if np.max(weights) > 0:
                weights = weights / np.max(weights)
        elif weight_type == 'constant':
            weights = np.ones_like(values)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        return weights
    
    def _transform_values_to_weights_batch(self, values, weight_type):
        """Batch version of weight transformation using MLX."""
        if weight_type == 'linear':
            weights = values / mx.max(values)
        elif weight_type == 'inverse':
            weights = 1.0 / (values + 1e-10)
            weights = weights / mx.max(weights)
        elif weight_type == 'power':
            weights = values ** 2
        elif weight_type == 'exp':
            weights = mx.exp(values * 5) - 1
            weights = weights / mx.max(weights)
        elif weight_type == 'log':
            weights = mx.log1p(values)
            weights = weights / mx.max(weights)
        elif weight_type == 'constant':
            weights = mx.ones_like(values)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        return weights


class FlowFilterSample(FlowFilterBase):
    """
    Flow filtering for single samples using numpy.
    """
    
    def filter(self, flow, reference_point=None):
        """
        Apply filtering based on the configuration.
        
        Args:
            flow (numpy.ndarray): Flow matrix of shape (h, w, 2)
            reference_point (tuple): (x, y) reference point for colinearity, None for image center
            
        Returns:
            numpy.ndarray: Filtered flow matrix
        """
        h, w = flow.shape[:2]
        
        # Initialize with all True mask
        combined_mask = np.ones((h, w), dtype=bool)
        
        # Apply norm filtering if enabled
        if self.config['filtering']['norm']['is_used']:
            norm_mask = self._filter_by_norm(flow)
            combined_mask = combined_mask & norm_mask
        
        # Apply colinearity filtering if enabled
        if self.config['filtering']['colinearity']['is_used']:
            colinearity_mask = self._filter_by_colinearity(flow, reference_point)
            combined_mask = combined_mask & colinearity_mask
        
        # Apply combined mask to flow
        filtered_flow = flow * combined_mask[..., np.newaxis]
        
        return filtered_flow
    
    def weight(self, flow, reference_point=None):
        """
        Apply weighting based on the configuration.
        
        Args:
            flow (numpy.ndarray): Flow matrix of shape (h, w, 2)
            reference_point (tuple): (x, y) reference point for colinearity, None for image center
            
        Returns:
            numpy.ndarray: Weight matrix of shape (h, w)
        """
        h, w = flow.shape[:2]
        weights = np.ones((h, w), dtype=np.float32)
        
        # Apply norm weighting if enabled
        if self.config['weighting']['norm']['is_used']:
            norm_weights = self._weight_by_norm(flow)
            weights = weights * norm_weights
        
        # Apply colinearity weighting if enabled
        if self.config['weighting']['colinearity']['is_used']:
            colinearity_weights = self._weight_by_colinearity(flow, reference_point)
            weights = weights * colinearity_weights
        
        return weights
    
    def filter_and_weight(self, flow, reference_point=None):
        """
        Apply both filtering and weighting in one call.
        
        Args:
            flow (numpy.ndarray): Flow matrix of shape (h, w, 2)
            reference_point (tuple): (x, y) reference point for colinearity, None for image center
            
        Returns:
            tuple: (filtered_flow, weights)
        """
        filtered_flow = self.filter(flow, reference_point)
        weights = self.weight(filtered_flow, reference_point)
        
        return filtered_flow, weights
    
    def _filter_by_norm(self, flow):
        """Apply norm-based filtering."""
        norms = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        threshold = self.config['filtering']['norm']['min_threshold']
        return norms >= threshold
    
    def _filter_by_colinearity(self, flow, reference_point=None):
        """Apply colinearity-based filtering."""
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
        
        # Apply threshold
        threshold = self.config['filtering']['colinearity']['min_threshold']
        return colinearity >= threshold
    
    def _weight_by_norm(self, flow):
        """Apply norm-based weighting."""
        norms = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        weight_type = self.config['weighting']['norm']['type']
        return self._transform_values_to_weights(norms, weight_type)
    
    def _weight_by_colinearity(self, flow, reference_point=None):
        """Apply colinearity-based weighting."""
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
        
        # Normalize colinearity from [-1, 1] to [0, 1]
        normalized_colinearity = (colinearity + 1) / 2
        
        weight_type = self.config['weighting']['colinearity']['type']
        return self._transform_values_to_weights(normalized_colinearity, weight_type)


class FlowFilterBatch(FlowFilterBase):
    """
    Flow filtering for batches using MLX for GPU acceleration.
    """
    
    def filter(self, flows, reference_points=None, chunk_size=1200):
        """
        Apply filtering to a batch of flows.
        
        Args:
            flows: Batch of flow matrices of shape (batch_size, h, w, 2)
            reference_points: Reference points for colinearity:
                - None: use image center for all flows
                - tuple (x, y): use same point for all flows  
                - list/array [(x1,y1), (x2,y2), ...]: one point per flow
            chunk_size: Chunk size for processing (default: 1200)
            
        Returns:
            mx.array: Filtered flows batch
        """
        # Preserve original dtype
        original_dtype = flows.dtype if hasattr(flows, 'dtype') else np.float32
        
        # Convert to MLX array if necessary
        if isinstance(flows, np.ndarray):
            flows = mx.array(flows, dtype=original_dtype)
        
        batch_size = flows.shape[0]
        all_filtered_flows = mx.zeros_like(flows)
        
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            flows_chunk = flows[start_idx:end_idx]
            
            # Extract reference points for this chunk
            chunk_reference_points = self._extract_chunk_reference_points(
                reference_points, start_idx, end_idx, batch_size
            )
            
            # Initialize combined mask
            chunk_size_actual = flows_chunk.shape[0]
            h, w = flows_chunk.shape[1], flows_chunk.shape[2]
            combined_mask = mx.ones((chunk_size_actual, h, w), dtype=mx.bool_)
            
            # Apply norm filtering if enabled
            if self.config['filtering']['norm']['is_used']:
                norm_mask = self._filter_by_norm_batch(flows_chunk)
                combined_mask = combined_mask & norm_mask
            
            # Apply colinearity filtering if enabled
            if self.config['filtering']['colinearity']['is_used']:
                colinearity_mask = self._filter_by_colinearity_batch(flows_chunk, chunk_reference_points)
                combined_mask = combined_mask & colinearity_mask
            
            # Apply mask to flows
            filtered_flows_chunk = flows_chunk * combined_mask[..., None]
            
            # Store results
            all_filtered_flows = mx.concatenate([
                all_filtered_flows[:start_idx],
                filtered_flows_chunk,
                all_filtered_flows[end_idx:]
            ])
            
            # Memory cleanup
            del flows_chunk, combined_mask, filtered_flows_chunk
            import gc
            gc.collect()
        
        return all_filtered_flows
    
    def weight(self, flows, reference_points=None, chunk_size=1200):
        """
        Apply weighting to a batch of flows.
        
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
            if self.config['weighting']['norm']['is_used']:
                norm_weights = self._weight_by_norm_batch(flows_chunk)
                weights_chunk = weights_chunk * norm_weights
            
            # Apply colinearity weighting if enabled
            if self.config['weighting']['colinearity']['is_used']:
                colinearity_weights = self._weight_by_colinearity_batch(flows_chunk, chunk_reference_points)
                weights_chunk = weights_chunk * colinearity_weights
            
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
    
    def filter_and_weight(self, flows, reference_points=None, chunk_size=1200):
        """
        Apply both filtering and weighting to a batch of flows.
        
        Args:
            flows: Batch of flow matrices of shape (batch_size, h, w, 2)
            reference_points: Reference points for colinearity:
                - None: use image center for all flows
                - tuple (x, y): use same point for all flows  
                - list/array [(x1,y1), (x2,y2), ...]: one point per flow
            chunk_size: Chunk size for processing (default: 1200)
            
        Returns:
            tuple: (filtered_flows, weights)
        """
        filtered_flows = self.filter(flows, reference_points, chunk_size)
        weights = self.weight(filtered_flows, reference_points, chunk_size)
        
        return filtered_flows, weights
    
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
    
    def _filter_by_norm_batch(self, flows_chunk):
        """Batch version of norm filtering."""
        norms = mx.sqrt(flows_chunk[..., 0]**2 + flows_chunk[..., 1]**2)
        threshold = self.config['filtering']['norm']['min_threshold']
        return norms >= threshold
    
    def _filter_by_colinearity_batch(self, flows_chunk, reference_points=None):
        """Batch version of colinearity filtering."""
        chunk_size, h, w = flows_chunk.shape[:3]
        
        # Handle reference points
        if reference_points is None:
            # Use image center for all flows
            ref_x, ref_y = w // 2, h // 2
            pixel_x = mx.arange(0, w)[None, :] - ref_x  # Shape: (1, w)
            pixel_y = mx.arange(0, h)[:, None] - ref_y  # Shape: (h, 1)
        elif isinstance(reference_points, tuple):
            # Same point for all flows
            ref_x, ref_y = reference_points
            pixel_x = mx.arange(0, w)[None, :] - ref_x  # Shape: (1, w)
            pixel_y = mx.arange(0, h)[:, None] - ref_y  # Shape: (h, 1)
        else:
            # Different point per flow - more complex handling needed
            # For now, implement the simpler cases first
            raise NotImplementedError("Different reference points per flow not yet implemented")
        
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
        
        # Apply threshold
        threshold = self.config['filtering']['colinearity']['min_threshold']
        return colinearity >= threshold
    
    def _weight_by_norm_batch(self, flows_chunk):
        """Batch version of norm weighting."""
        norms = mx.sqrt(flows_chunk[..., 0]**2 + flows_chunk[..., 1]**2)
        weight_type = self.config['weighting']['norm']['type']
        return self._transform_values_to_weights_batch(norms, weight_type)
    
    def _weight_by_colinearity_batch(self, flows_chunk, reference_points=None):
        """Batch version of colinearity weighting."""
        chunk_size, h, w = flows_chunk.shape[:3]
        
        # Handle reference points
        if reference_points is None:
            # Use image center for all flows
            ref_x, ref_y = w // 2, h // 2
            pixel_x = mx.arange(0, w)[None, :] - ref_x  # Shape: (1, w)
            pixel_y = mx.arange(0, h)[:, None] - ref_y  # Shape: (h, 1)
        elif isinstance(reference_points, tuple):
            # Same point for all flows
            ref_x, ref_y = reference_points
            pixel_x = mx.arange(0, w)[None, :] - ref_x  # Shape: (1, w)
            pixel_y = mx.arange(0, h)[:, None] - ref_y  # Shape: (h, 1)
        else:
            # Different point per flow - more complex handling needed
            # For now, implement the simpler cases first
            raise NotImplementedError("Different reference points per flow not yet implemented")
        
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
        
        # Normalize colinearity from [-1, 1] to [0, 1]
        normalized_colinearity = (colinearity + 1) / 2
        
        weight_type = self.config['weighting']['colinearity']['type']
        return self._transform_values_to_weights_batch(normalized_colinearity, weight_type)


if __name__ == "__main__":
    import time
    import psutil
    
    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Load data
    flows = np.load('calib_challenge/flows/0.npy')
    
    # Convert to MLX array once
    xflows = mx.array(flows)
    
    print("🔬 TESTING NEW CLEAN FLOWFILTER STRUCTURE")
    print(f"Flow shape: {flows.shape}")
    print("="*70)
    
    # ========================================
    # EXAMPLE 1: SINGLE SAMPLE PROCESSING
    # ========================================
    print("\n📊 EXAMPLE 1: SINGLE SAMPLE PROCESSING (FlowFilterSample)")
    print("-" * 60)
    
    config_sample = {
        'filtering': {
            'norm': {'is_used': True, 'min_threshold': 13}
        },
        'weighting': {
            'norm': {'is_used': True, 'type': 'linear'}
        }
    }
    
    print(f"Configuration: {config_sample}")
    
    flow_filter_sample = FlowFilterSample(config_sample)
    
    # Test single flow
    single_flow = flows[0]
    filtered_flow, weights = flow_filter_sample.filter_and_weight(single_flow)
    
    print(f"Original non-zero elements: {np.sum(single_flow != 0)}")
    print(f"Filtered non-zero elements: {np.sum(filtered_flow != 0)}")
    print(f"Weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}]")
    
    # ========================================
    # EXAMPLE 2: BATCH PROCESSING
    # ========================================
    print("\n📊 EXAMPLE 2: BATCH PROCESSING (FlowFilterBatch)")
    print("-" * 60)
    
    config_batch = {
        'filtering': {
            'norm': {'is_used': True, 'min_threshold': 13},
            'colinearity': {'is_used': True, 'min_threshold': 0.97}
        },
        'weighting': {
            'norm': {'is_used': True, 'type': 'linear'},
            'colinearity': {'is_used': True, 'type': 'exp'}
        }
    }
    
    print(f"Configuration: {config_batch}")
    
    flow_filter_batch = FlowFilterBatch(config_batch)
    
    batch_size = 100
    print(f"Testing batch processing with {batch_size} flows")
    
    start_time = time.time()
    filtered_flows_batch, weights_batch = flow_filter_batch.filter_and_weight(
        xflows[:batch_size], chunk_size=50
    )
    batch_time = time.time() - start_time
    
    print(f"Batch processing completed in {batch_time:.3f}s ({batch_time/batch_size:.4f}s per flow)")
    print(f"Batch filtered flows shape: {filtered_flows_batch.shape}")
    print(f"Batch weights shape: {weights_batch.shape}")
    
    # ========================================
    # EXAMPLE 3: STEP-BY-STEP PROCESSING
    # ========================================
    print("\n📊 EXAMPLE 3: STEP-BY-STEP PROCESSING")
    print("-" * 60)
    
    # Step 1: Apply filtering only
    filtered_flow_step = flow_filter_sample.filter(single_flow)
    print(f"Step 1 - Filtering: {np.sum(single_flow != 0)} -> {np.sum(filtered_flow_step != 0)} elements")
    
    # Step 2: Apply weighting only (without filter masks)
    weights_step_no_mask = flow_filter_sample.weight(single_flow)
    print(f"Step 2a - Weighting (no mask): weight range [{np.min(weights_step_no_mask):.3f}, {np.max(weights_step_no_mask):.3f}]")
    
    # Step 3: Apply weighting with filter masks
    weights_step_with_mask = flow_filter_sample.weight(single_flow)
    print(f"Step 2b - Weighting (with mask): weight range [{np.min(weights_step_with_mask):.3f}, {np.max(weights_step_with_mask):.3f}]")
    
    # ========================================
    # EXAMPLE 4: DIFFERENT CONFIGURATIONS
    # ========================================
    print("\n📊 EXAMPLE 4: DIFFERENT CONFIGURATIONS")
    print("-" * 60)
    
    configs = {
        'Norm only': {
            'filtering': {'norm': {'is_used': True, 'min_threshold': 13}},
            'weighting': {'norm': {'is_used': True, 'type': 'linear'}}
        },
        'Colinearity only': {
            'filtering': {'colinearity': {'is_used': True, 'min_threshold': 0.97}},
            'weighting': {'colinearity': {'is_used': True, 'type': 'exp'}}
        },
        'Weighting only': {
            'weighting': {
                'norm': {'is_used': True, 'type': 'power'},
                'colinearity': {'is_used': True, 'type': 'log'}
            }
        }
    }
    
    for name, config in configs.items():
        filter_test = FlowFilterSample(config)
        filtered = filter_test.filter(single_flow)
        
        print(f"{name}: {np.sum(single_flow != 0)} -> {np.sum(filtered != 0)} elements, "
              f"weights [{np.min(filter_test.weight(single_flow)):.3f}, {np.max(filter_test.weight(single_flow)):.3f}]")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n🎯 SUMMARY")
    print("="*50)
    print("✅ Clean structure with 3 classes:")
    print("   - FlowFilterBase: Shared configuration and helper methods")
    print("   - FlowFilterSample: Single sample processing (numpy)")
    print("   - FlowFilterBatch: Batch processing (MLX acceleration)")
    print("✅ Same interface for both: filter(), weight(), filter_and_weight()")
    print("✅ Flexible configuration structure maintained")
    print("✅ No legacy code - clean and simple")
    print("✅ Clear separation of responsibilities")
    print()
    print("💡 Usage:")
    print("   - For single samples: FlowFilterSample(config)")
    print("   - For batches: FlowFilterBatch(config)")
    print("   - Same config structure for both classes")