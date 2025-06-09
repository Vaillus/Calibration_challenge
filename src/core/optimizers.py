"""
Module d'optimisation pour l'estimation du point de fuite.

Ce module centralise toutes les méthodes d'optimisation utilisées dans le projet :
- Adam optimizer (implémentation MLX native avec early stopping)
- L-BFGS-B optimizer (interface vers scipy.optimize)

Chaque optimiseur est conçu pour fonctionner avec les estimateurs de points de fuite
définis dans colinearity_optimization.py et colinearity_optimization_parallel.py.
"""

import time
import numpy as np
import mlx.core as mx
from scipy.optimize import minimize
import gc
from typing import Tuple, Optional, Union, List, Dict, Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from src.core.colinearity_optimization import CollinearityScorer
from src.core.colinearity_optimization_parallel import BatchCollinearityScorer


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers with common visualization functionality.
    
    This class provides:
    - Common visualization interface
    - State tracking for optimization trajectory
    - Abstract optimization methods to be implemented by subclasses
    """
    
    def __init__(self):
        """Initialize the base optimizer with empty visualization state."""
        self.trajectory = []
        self.scores = []
        self.estimator = None
    
    @abstractmethod
    def optimize_single(self, flow, starting_point, **kwargs):
        """
        Abstract method that must be implemented by subclasses to optimize a single sample.
        
        Args:
            flow: The optical flow data
            starting_point: Initial point for optimization
            **kwargs: Additional arguments specific to the optimizer implementation
        """
        pass
    
    @abstractmethod
    def optimize_batch(self, flow_batch, starting_points=None, **kwargs):
        """
        Abstract method that must be implemented by subclasses to optimize a batch of samples.
        
        Args:
            flow_batch: Batch of optical flow data
            starting_points: Optional starting points for the batch
            **kwargs: Additional arguments specific to the optimizer implementation
        """
        pass
    
    def visualize_optimization(self, flow: np.ndarray, final_point: Tuple[float, float], 
                             ground_truth_point: Optional[Tuple[float, float]] = None):
        """
        Visualize the optimization results.
        
        Args:
            estimator: Instance of VanishingPointEstimator for computing colinearity maps
            flow: Optical flow vector field
            final_point: Final optimized point (x, y)
            ground_truth_point: Optional ground truth point for visualization
        """
        assert self.estimator is not None, "Estimator must be set before visualizing"
        h, w = flow.shape[:2]
        
        plt.figure(figsize=(15, 10))
        
        # Plot score evolution
        plt.subplot(2, 2, 1)
        plt.plot([-s for s in self.scores])
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
        colinearity_map = self.estimator.compute_colinearity_map(flow, final_point)
        plt.imshow(colinearity_map, cmap='hot')
        plt.colorbar(label='Colinearity')
        if ground_truth_point:
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='green', s=100, marker='o', label='Ground truth')
        plt.title('Colinearity Map for Final Point')
        
        # Visualize descent trajectory
        plt.subplot(2, 2, 4)
        plt.imshow(np.zeros((h, w, 3)))
        
        # Convert list of tuples to numpy array for easier plotting
        trajectory_array = np.array(self.trajectory)
        plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'r.-', linewidth=2)
        plt.scatter(final_point[0], final_point[1], c='g', s=100, marker='*')
        if ground_truth_point:
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='green', s=100, marker='o', label='Ground truth')
        
        # Add annotations for first points
        for i, (x, y) in enumerate(self.trajectory[:min(5, len(self.trajectory))]):
            plt.annotate(f"{i}", (x, y), fontsize=12, color='white')
        
        plt.title('Gradient Descent Trajectory')
        plt.xlim(0, w)
        plt.ylim(h, 0)  # Invert y-axis to match image coordinates
        
        plt.tight_layout()
        plt.show()


class LBFGSOptimizer(BaseOptimizer):
    """
    Interface vers l'optimiseur L-BFGS-B de scipy pour l'estimation de points de fuite.
    
    Caractéristiques :
    - Précision maximale (meilleur convergence que Adam dans la plupart des cas)
    - Support des poids pour les vecteurs de flux
    - Callback pour visualisation/debug
    - Compatible avec les estimateurs numpy/scipy
    """
    def __init__(
        self, 
        max_iter: int = 100,
        display_warnings: bool = False
    ):
        """
        Args:
            max_iter: Maximum iterations for L-BFGS-B
            display_warnings: Whether to display scipy optimization warnings
        """
        super().__init__()
        self.max_iter = max_iter
        self.display_warnings = display_warnings
        # We use the numpy version of the estimator because the function scipy.optimize.minimize does not support mx.array
        self.estimator = CollinearityScorer()
    
    def optimize_single(
            self, 
            flow: np.ndarray, 
            starting_point: Optional[np.ndarray] = None,
            weights: Optional[np.ndarray] = None,
            visualize: Optional[bool] = False,
            ground_truth_point: Optional[Tuple[float, float]] = None
        ) -> Tuple[float, float]:
        """
        Optimise un seul échantillon avec L-BFGS-B.
        
        Args:
            estimator: Instance de VanishingPointEstimator (version numpy)
            flow: Flux optique (h, w, 2)
            starting_point: Point de départ [x, y]
            weights: Poids optionnels (h, w)
            visualize: Whether to visualize the optimization process
            ground_truth_point: Optional ground truth point for visualization
            
        Returns:
            Point de fuite optimisé (x, y)
        """
        if starting_point is None:
            starting_point = np.array([flow.shape[1] // 2, flow.shape[0] // 2])
        
        # Check for valid pixels
        valid_pixels = (flow[:,:,0]**2 + flow[:,:,1]**2) > 0
        if np.sum(valid_pixels) == 0:
            return tuple(starting_point)
        
        # Reset visualization data
        self.trajectory = [starting_point]
        self.scores = [self.estimator.objective_function(starting_point, flow, weights)]
        
        # Create callback if visualization is enabled
        callback_fn = None
        if visualize:
            def callback(point):
                self.trajectory.append(point.copy())
                self.scores.append(self.estimator.objective_function(point, flow, weights))
            callback_fn = callback
        
        # Optimization with L-BFGS-B
        result = minimize(
            lambda point: self.estimator.objective_function(point, flow, weights),
            starting_point,
            method='L-BFGS-B',
            callback=callback_fn,
            options={
                'disp': self.display_warnings,
                'maxiter': self.max_iter
            }
        )
        
        final_point = tuple(result.x)
        
        # Visualize if requested
        if visualize:
            self.visualize_optimization(flow, final_point, ground_truth_point)
        
        return final_point
    
    def optimize_batch(self, flow_batch: np.ndarray, 
                      starting_points: np.ndarray,
                      weights_batch: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimisation séquentielle d'un batch avec L-BFGS-B.
        
        Args:
            estimator: Instance de VanishingPointEstimator
            flow_batch: Batch de flux optiques (batch_size, h, w, 2)
            starting_points: Points de départ (batch_size, 2)
            weights_batch: Poids optionnels (batch_size, h, w)
            
        Returns:
            Points de fuite optimisés (batch_size, 2)
        """
        batch_size = flow_batch.shape[0]
        final_points = []
        
        for i in range(batch_size):
            single_flow = flow_batch[i]
            single_start_point = starting_points[i]
            single_weights = weights_batch[i] if weights_batch is not None else None
            
            final_point = self.optimize_single(single_flow, single_start_point, single_weights
            )
            final_points.append(final_point)
        
        return np.array(final_points)


class AdamOptimizer(BaseOptimizer):
    """
    Optimiseur Adam implémenté nativement avec MLX pour l'estimation de points de fuite.
    
    Caractéristiques :
    - Support batch natif (traitement séquentiel optimisé)
    - Early stopping avec détection de plateau automatique
    - Gestion mémoire optimisée pour de gros volumes
    - Paramètres pré-optimisés pour la convergence rapide
    """
    
    def __init__(self, 
                 lr: float = 10.0,
                 beta1: float = 0.6, 
                 beta2: float = 0.98,
                 eps: float = 1e-8,
                 max_iter: int = 50,
                 plateau_threshold: float = 1e-4,
                 plateau_patience: int = 3):
        """
        Args:
            lr: Learning rate (optimisé pour les points de fuite)
            beta1: Adam momentum parameter (ajusté pour convergence rapide)
            beta2: Adam variance parameter (ajusté pour stabilité)
            eps: Numerical stability epsilon
            max_iter: Maximum iterations per optimization
            plateau_threshold: Minimum improvement to continue (1e-4 optimal)
            plateau_patience: Iterations to check for improvement (3 optimal)
        """
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iter = max_iter
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.estimator = BatchCollinearityScorer()
    
    def optimize_single(
        self, 
        single_flow: mx.array, 
        starting_point: mx.array,
        visualize: bool = False, 
        ground_truth_point: Optional[Tuple[float, float]] = None
    ) -> mx.array:
        """
        Optimise un seul échantillon avec Adam.
        
        Args:
            estimator: Instance de ParallelVanishingPointEstimator
            single_flow: Flux optique unique de forme (h, w, 2)
            starting_point: Point de départ [x, y]
            visualize: Whether to visualize the optimization process
            ground_truth_point: Optional ground truth point for visualization
            
        Returns:
            Point de fuite optimisé [x, y]
        """
        x = starting_point
        m = mx.zeros_like(x)  # momentum
        v = mx.zeros_like(x)  # variance
        
        # Force evaluation of initial state
        mx.eval(x, m, v)
        
        # Reset visualization data
        self.trajectory = [np.array(x)]
        self.scores = [float(-self.estimator.colin_score(single_flow, x))]
        
        # Plateau detection - store recent scores efficiently
        recent_scores = []
        
        for t in range(1, self.max_iter + 1):
            # Loss function for single sample - returns scalar
            def single_loss(point):
                score = self.estimator.colin_score(single_flow, point)
                return -score  # Negative because we want to maximize score
            
            # Compute gradient
            grad = mx.grad(single_loss)(x)
            mx.eval(grad)
            
            # Adam updates
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad**2
            mx.eval(m, v)
            
            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)
            
            x = x - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)
            mx.eval(x)
            
            # Store trajectory and score for visualization
            if visualize:
                self.trajectory.append(np.array(x))
                self.scores.append(float(-single_loss(x)))
            
            # Efficient plateau detection
            if self.plateau_threshold > 0 and t >= 5:
                current_score = float(-single_loss(x))
                recent_scores.append(current_score)
                
                # Keep only the last plateau_patience scores
                if len(recent_scores) > self.plateau_patience:
                    recent_scores.pop(0)
                
                # Check for plateau: improvement over plateau_patience iterations
                if len(recent_scores) == self.plateau_patience:
                    improvement = recent_scores[-1] - recent_scores[0]
                    if improvement < self.plateau_threshold:
                        break
        
        mx.eval(x)
        
        # Visualize if requested
        if visualize:
            self.visualize_optimization(
                np.array(single_flow), tuple(np.array(x)), ground_truth_point)
        
        return x
    
    def optimize_batch(self, flow_batch: mx.array, 
                      starting_points: Optional[mx.array] = None) -> mx.array:
        """
        Optimize a batch of optical flow samples to find vanishing points.
        Note: Samples are optimized sequentially.
        
        Args:
            estimator: Instance of ParallelVanishingPointEstimator
            flow_batch: Batch of optical flows (batch_size, h, w, 2)
            starting_points: Optional starting points (batch_size, 2)
            
        Returns:
            Optimized vanishing points (batch_size, 2)
        """
        batch_size = flow_batch.shape[0]
        
        # Initialize starting points at image center if not provided
        if starting_points is None:
            center_x = flow_batch.shape[2] // 2
            center_y = flow_batch.shape[1] // 2
            center_point = mx.array([center_x, center_y], dtype=mx.float32)
            starting_points = mx.tile(center_point, (batch_size, 1))
            mx.eval(starting_points)
        
        final_points = []
        
        for i in range(batch_size):
            single_flow = flow_batch[i]
            single_start_point = starting_points[i]
            # TODO: try reoptimizing the starting point
            mx.eval(single_flow, single_start_point)
            
            final_point = self.optimize_single(single_flow, single_start_point)
            mx.eval(final_point)
            
            final_points.append(final_point)
            
            # Memory cleanup every 10 samples
            if (i + 1) % 10 == 0:
                gc.collect()
        
        result = mx.stack(final_points, axis=0)
        mx.eval(result)
        return result
