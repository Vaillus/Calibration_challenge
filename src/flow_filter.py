import numpy as np
import mlx.core as mx

class FlowFilter:
    def __init__(self, min_norm_threshold=1e-2, weight_mode='linear', weight_power=1.0):
        """
        Initialize the FlowFilter with configurable weight transformation.
        
        Args:
            min_norm_threshold (float): Minimum norm threshold for filtering
            weight_mode (str): Mode of weight transformation:
                - 'linear': weights are proportional to norms (default)
                - 'inverse': weights are inversely proportional to norms
                - 'power': weights are proportional to norms raised to a power
                - 'exp': weights are proportional to exponential of norms
                - 'log': weights are proportional to logarithm of norms
                - 'constant': all weights are set to 1
            weight_power (float): Power to use for 'power' mode (default=1.0)
        """
        self.min_norm_threshold = min_norm_threshold
        self.weight_mode = weight_mode
        self.weight_power = weight_power

    def filter_by_distance(self, flow):
       pass 

    def filter_by_norm(self, flow):
        """
        Filtre les vecteurs de flow basé sur leur norme minimale et retourne une matrice de pondération.
        
        Args:
            flow (numpy.ndarray): Matrice de flow de forme (h, w, 2)
            
        Returns:
            tuple: (filtered_flow, weights) où:
                - filtered_flow est la matrice de flow filtrée de même forme, avec 0 pour les vecteurs filtrés
                - weights est une matrice de pondération de forme (h, w) avec des valeurs entre 0 et 1
        """
        norms = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mask = norms >= self.min_norm_threshold
        
        # Calcul des poids en utilisant la transformation configurée
        weights = self._transform_norms_to_weights(norms)
        
        # Appliquer le masque aux poids
        weights = weights * mask
        
        return flow * mask[..., np.newaxis], weights 
    
    def _transform_norms_to_weights(self, norms):
        """
        Transform flow vector norms into weights using the configured mode.
        
        Args:
            norms (numpy.ndarray): Array of flow vector norms
            
        Returns:
            numpy.ndarray: Transformed weights between 0 and 1
        """
        if np.max(norms) == 0:
            return np.zeros_like(norms)
            
        # Normalize norms to [0, 1] range
        normalized_norms = norms / np.max(norms)
        
        if self.weight_mode == 'linear':
            weights = normalized_norms
        elif self.weight_mode == 'inverse':
            # Avoid division by zero by adding small epsilon
            weights = 1.0 / (normalized_norms + 1e-10)
            # Normalize again to [0, 1] range
            weights = weights / np.max(weights)
        elif self.weight_mode == 'power':
            weights = normalized_norms ** self.weight_power
        elif self.weight_mode == 'exp':
            # Scale down to avoid overflow
            weights = np.exp(normalized_norms * 5) - 1
            weights = weights / np.max(weights)
        elif self.weight_mode == 'log':
            # Add 1 to avoid log(0)
            weights = np.log1p(normalized_norms)
            weights = weights / np.max(weights)
        elif self.weight_mode == 'constant':
            weights = np.ones_like(norms)
        else:
            raise ValueError(f"Unknown weight mode: {self.weight_mode}")
            
        return weights
    
    def filter_by_norm_batch(self, flows:np.ndarray, chunk_size:int=1200):
        """
        Version parallèle de filter_by_norm qui traite plusieurs flows en batch.
        Utilise MLX pour l'optimisation GPU et le traitement par chunks.
        Preserve le dtype original pour correspondre exactement à la méthode séquentielle.
        
        Args:
            flows: Batch de matrices de flow de forme (batch_size, h, w, 2)
                  Peut être soit un numpy array soit un MLX array
            chunk_size: Taille des chunks pour le traitement parallèle (default: 1200)
            
        Returns:
            tuple: (filtered_flows, weights) où:
                - filtered_flows est le batch de flows filtrés de même forme
                - weights est un batch de matrices de pondération de forme (batch_size, h, w)
        """
        # Préserver le dtype original
        original_dtype = flows.dtype if hasattr(flows, 'dtype') else np.float32
        
        # Convertir en MLX array si nécessaire, en préservant le dtype
        if isinstance(flows, np.ndarray):
            flows = mx.array(flows, dtype=original_dtype)
            
        batch_size = flows.shape[0]
        all_filtered_flows = mx.zeros_like(flows)
        all_weights = mx.zeros((batch_size, flows.shape[1], flows.shape[2]), dtype=original_dtype)
        
        for start_idx in range(0, batch_size, chunk_size):
            end_idx = min(start_idx + chunk_size, batch_size)
            
            # Get current chunk
            flows_chunk = flows[start_idx:end_idx]
            
            # Compute norms en préservant le dtype
            norms = mx.sqrt(flows_chunk[..., 0]**2 + flows_chunk[..., 1]**2)
            mask = norms >= self.min_norm_threshold
            
            # Compute weights using MLX operations avec le bon dtype
            if self.weight_mode == 'linear':
                weights = norms / mx.max(norms)
            elif self.weight_mode == 'inverse':
                weights = 1.0 / (norms + 1e-10)
                weights = weights / mx.max(weights)
            elif self.weight_mode == 'power':
                weights = norms ** self.weight_power
            elif self.weight_mode == 'exp':
                weights = mx.exp(norms * 5) - 1
                weights = weights / mx.max(weights)
            elif self.weight_mode == 'log':
                weights = mx.log1p(norms)
                weights = weights / mx.max(weights)
            elif self.weight_mode == 'constant':
                weights = mx.ones_like(norms)
            else:
                raise ValueError(f"Unknown weight mode: {self.weight_mode}")
            
            # Apply mask to weights
            weights = weights * mask
            
            # Apply mask to flows
            filtered_flows = flows_chunk * mask[..., None]
            
            # Store results
            all_filtered_flows = mx.concatenate([
                all_filtered_flows[:start_idx],
                filtered_flows,
                all_filtered_flows[end_idx:]
            ])
            all_weights = mx.concatenate([
                all_weights[:start_idx],
                weights,
                all_weights[end_idx:]
            ])
            
            # Clear MLX memory
            del flows_chunk, norms, mask, weights, filtered_flows
            import gc
            gc.collect()
        
        return all_filtered_flows, all_weights

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
    
    # Test different batch sizes
    batch_sizes = [100]
    
    print("Testing different batch sizes on GPU:")
    print(f"Flow shape: {flows.shape}")
    
    for batch_size in batch_sizes:
        print(f"\nTesting with batch size: {batch_size}")
        
        # Initialize filter
        flow_filter = FlowFilter(min_norm_threshold=1e-2, weight_mode='linear')
        
        # Initial memory state
        initial_mem = get_memory_usage()
        print(f"Initial memory usage: {initial_mem:.1f} MB")
        
        # Test on GPU
        start_time = time.time()
        filtered_flows, weights = flow_filter.filter_by_norm_batch(xflows, chunk_size=batch_size)
        gpu_time = time.time() - start_time
        
        print(f"GPU time: {gpu_time:.3f} seconds")
        
        # Verify computations
        print("\nVerifying computations:")
        print(f"Type of filtered_flows: {type(filtered_flows)}")
        print(f"Type of weights: {type(weights)}")
        
        # Check if any filtering was done
        original_non_zero = mx.sum(xflows != 0)
        filtered_non_zero = mx.sum(filtered_flows != 0)
        print(f"Original non-zero elements: {original_non_zero}")
        print(f"Filtered non-zero elements: {filtered_non_zero}")
        print(f"Difference: {original_non_zero - filtered_non_zero}")
        
        # Check weights statistics
        print("\nWeights statistics:")
        print(f"Min weight: {float(mx.min(weights))}")
        print(f"Max weight: {float(mx.max(weights))}")
        print(f"Mean weight: {float(mx.mean(weights))}")
        
        # Print final memory usage
        final_mem = get_memory_usage()
        print(f"\nFinal memory usage: {final_mem:.1f} MB")
        print(f"Memory delta: {final_mem - initial_mem:.1f} MB")