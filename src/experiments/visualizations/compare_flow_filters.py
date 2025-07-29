#!/usr/bin/env python3
"""
Comparaison visuelle de trois niveaux de filtrage de flux optique.
Affiche côte à côte l'effet de différents filtres sur la densité des vecteurs.

Filtres comparés:
1. Aucun filtrage (flow brut)
2. Filtrage léger (norme min 0.01)
3. Filtrage fort (norme min 13 + colinéarité 0.96)

Usage:
    python compare_flow_filters.py
"""

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx

from src.utilities.paths import get_labeled_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.utilities.load_flows import load_flows
from src.utilities.load_ground_truth import get_frame_pixel
from src.core.flow_filter import FlowFilterSample
from src.core.optimizers import AdamOptimizer


def compare_flow_filters(video_idx: int = 4, frame_idx: int = 380, stride: int = 20):
    """
    Compare trois niveaux de filtrage côte à côte.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
        stride: Espacement entre les flèches
    """
    print(f"📂 Chargement frame {frame_idx} - vidéo {video_idx}...")
    
    # Load RGB frame
    video_path = get_labeled_dir() / f'{video_idx}.hevc'
    _, frame_rgb = read_frame_rgb(video_path, frame_idx)
    
    # Load raw flow data
    flow_data = load_flows(video_idx, start_frame=frame_idx-1, end_frame=frame_idx-1)
    flow_data = flow_data[0]  # Remove batch dimension: (1, H, W, 2) -> (H, W, 2)
    
    # Get ground truth point for colinearity reference
    gt_pixels = get_frame_pixel(video_idx, frame_idx)
    ref_point = (gt_pixels[0], gt_pixels[1])
    
    print(f"✅ Données chargées - Flow shape: {flow_data.shape}")
    
    # Configuration des trois filtres
    filters_config = {
        'Aucun filtrage': None,  # Pas de filtrage
        'Filtrage léger\n(norme ≥ 0.01)': {
            'norm': {'is_used': True, 'k': 20.0, 'x0': 0.01},
            'colinearity': {'is_used': False}
        },
        'Filtrage fort\n(norme ≥ 13, colin ≥ 0.96)': {
            'norm': {'is_used': True, 'k': 20.0, 'x0': 13.0},
            'colinearity': {'is_used': True, 'k': 20.0, 'x0': 0.96}
        }
    }
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for arrows
    h, w = flow_data.shape[:2]
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Initialize Adam optimizer with threshold stopping (no pixel patience)
    adam_optimizer = AdamOptimizer(
        lr=10.0,
        plateau_threshold=0,  # Arrêt par seuil seulement
        pixel_patience=0,     # Pas de pixel patience
        max_iter=100
    )
    
    for idx, (filter_name, config) in enumerate(filters_config.items()):
        ax = axes[idx]
        
        # Apply filtering and predict vanishing point
        if config is None:
            # No filtering - use raw flow
            filtered_flow = flow_data.copy()
            fx = filtered_flow[y, x, 0]
            fy = filtered_flow[y, x, 1]
            
            # Count all vectors
            total_vectors = len(fx)
            visible_vectors = total_vectors
            
            # Predict vanishing point with raw flow
            mlx_flow = mx.array(filtered_flow, dtype=mx.float32)
            predicted_point = adam_optimizer.optimize_single(mlx_flow)
            predicted_point_np = np.array(predicted_point)
            
        else:
            # Apply the specified filter
            flow_filter = FlowFilterSample(config)
            filtered_flow = flow_filter.filter(flow_data, reference_point=ref_point)
            
            # Get flow vectors at grid points
            fx = filtered_flow[y, x, 0]
            fy = filtered_flow[y, x, 1]
            
            # Count only non-zero vectors (those that passed the filter)
            magnitude = np.sqrt(fx**2 + fy**2)
            mask = magnitude > 1e-6  # Small threshold to avoid numerical noise
            fx = fx[mask]
            fy = fy[mask]
            x_filtered = x[mask]
            y_filtered = y[mask]
            
            total_vectors = len(x)
            visible_vectors = len(fx)
            
            # Update coordinates for filtered vectors
            x = x_filtered
            y = y_filtered
            
            # Predict vanishing point with filtered flow
            mlx_flow = mx.array(filtered_flow, dtype=mx.float32)
            predicted_point = adam_optimizer.optimize_single(mlx_flow)
            predicted_point_np = np.array(predicted_point)
        
        # Display the frame as background
        ax.imshow(frame_rgb)
        
        # Plot flow vectors
        if len(fx) > 0:
            ax.quiver(x, y, fx, fy, 
                     color='red', angles='xy', scale_units='xy', scale=1, 
                     width=0.002, alpha=0.8)
        
        # Add ground truth point
        ax.scatter(gt_pixels[0], gt_pixels[1], 
                  color='blue', s=100, marker='*', 
                  label='Label', edgecolor='white', linewidth=1)
        
        # Add predicted point
        ax.scatter(predicted_point_np[0], predicted_point_np[1], 
                  color='orange', s=100, marker='*', 
                  label='Prédiction', edgecolor='white', linewidth=1)
        
        # Calculate prediction error
        error_pixels = np.sqrt((predicted_point_np[0] - gt_pixels[0])**2 + 
                              (predicted_point_np[1] - gt_pixels[1])**2)
        
        # Set title with vector count and prediction error
        percentage = (visible_vectors / total_vectors * 100) if total_vectors > 0 else 0
        ax.set_title(f'{filter_name}\n {percentage:.1f}% des vecteurs\nErreur: {error_pixels:.1f} px', 
                    fontsize=12, pad=10)
        
        ax.axis('off')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Reset coordinates for next iteration
        y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # plt.suptitle(f'Comparaison des filtres - Vidéo {video_idx}, Frame {frame_idx}', 
                # fontsize=16, y=0.95)
    plt.tight_layout()
    plt.show()
    
    print("✅ Comparaison terminée")


def main():
    """Fonction principale avec paramètres par défaut."""
    print("🎯 COMPARAISON DES FILTRES DE FLUX OPTIQUE")
    print("=" * 50)
    
    # Paramètres par défaut
    video_idx = 4
    frame_idx = 380
    stride = 20
    
    print(f"📊 Configuration: Vidéo {video_idx}, Frame {frame_idx}, Stride {stride}")
    
    compare_flow_filters(video_idx=video_idx, frame_idx=frame_idx, stride=stride)


if __name__ == "__main__":
    main() 