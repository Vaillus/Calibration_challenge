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


class AdamOptimizer:
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
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_iter = max_iter
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
    
    def optimize_single(self, estimator, single_flow: mx.array, starting_point: mx.array) -> mx.array:
        """
        Optimise un seul échantillon avec Adam.
        
        Args:
            estimator: Instance de ParallelVanishingPointEstimator
            single_flow: Flux optique unique de forme (h, w, 2)
            starting_point: Point de départ [x, y]
            
        Returns:
            Point de fuite optimisé [x, y]
        """
        x = starting_point
        m = mx.zeros_like(x)  # momentum
        v = mx.zeros_like(x)  # variance
        
        # Force evaluation of initial state
        mx.eval(x, m, v)
        
        # Plateau detection - store recent scores efficiently
        recent_scores = []
        
        for t in range(1, self.max_iter + 1):
            # Loss function for single sample - returns scalar
            def single_loss(point):
                score = estimator.colin_score(single_flow, point)
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
        return x
    
    def optimize_batch(self, estimator, flow_batch: mx.array, 
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
            
            final_point = self.optimize_single(estimator, single_flow, single_start_point)
            mx.eval(final_point)
            
            final_points.append(final_point)
            
            # Memory cleanup every 10 samples
            if (i + 1) % 10 == 0:
                gc.collect()
        
        result = mx.stack(final_points, axis=0)
        mx.eval(result)
        return result


class LBFGSOptimizer:
    """
    Interface vers l'optimiseur L-BFGS-B de scipy pour l'estimation de points de fuite.
    
    Caractéristiques :
    - Précision maximale (meilleur convergence que Adam dans la plupart des cas)
    - Support des poids pour les vecteurs de flux
    - Callback pour visualisation/debug
    - Compatible avec les estimateurs numpy/scipy
    """
    
    def __init__(self, 
                 max_iter: int = 100,
                 display_warnings: bool = False):
        """
        Args:
            max_iter: Maximum iterations for L-BFGS-B
            display_warnings: Whether to display scipy optimization warnings
        """
        self.max_iter = max_iter
        self.display_warnings = display_warnings
    
    def optimize_single(self, estimator, flow: np.ndarray, starting_point: np.ndarray,
                       weights: Optional[np.ndarray] = None,
                       callback_fn: Optional[callable] = None) -> Tuple[float, float]:
        """
        Optimise un seul échantillon avec L-BFGS-B.
        
        Args:
            estimator: Instance de VanishingPointEstimator (version numpy)
            flow: Flux optique (h, w, 2)
            starting_point: Point de départ [x, y]
            weights: Poids optionnels (h, w)
            callback_fn: Fonction callback optionnelle pour traçage
            
        Returns:
            Point de fuite optimisé (x, y)
        """
        # Check for valid pixels
        valid_pixels = (flow[:,:,0]**2 + flow[:,:,1]**2) > 0
        if np.sum(valid_pixels) == 0:
            return tuple(starting_point)
        
        # Optimization with L-BFGS-B
        result = minimize(
            lambda point: estimator.objective_function(point, flow, weights),
            starting_point,
            method='L-BFGS-B',
            callback=callback_fn,
            options={
                'disp': self.display_warnings,
                'maxiter': self.max_iter
            }
        )
        
        return tuple(result.x)
    
    def optimize_batch(self, estimator, flow_batch: np.ndarray, 
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
            
            final_point = self.optimize_single(
                estimator, single_flow, single_start_point, single_weights
            )
            final_points.append(final_point)
        
        return np.array(final_points)


# === API Functions pour compatibilité ===


def optimize_batch(flow_batch, plateau_threshold=1e-4, plateau_patience=3):
    """
    API de compatibilité pour optimize_batch.
    Utilise maintenant la classe AdamOptimizer.
    """
    from src.core.colinearity_optimization_parallel import ParallelVanishingPointEstimator
    
    estimator = ParallelVanishingPointEstimator(
        flow_batch.shape[2], flow_batch.shape[1], 
        use_max_distance=False, use_reoptimization=False
    )
    
    optimizer = AdamOptimizer(plateau_threshold=plateau_threshold, plateau_patience=plateau_patience)
    return optimizer.optimize_batch(estimator, flow_batch) 