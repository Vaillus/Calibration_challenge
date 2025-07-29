#!/usr/bin/env python3
"""
Visualisation de la carte topographique des scores de collinéarité sur toute l'image.

Cette visualisation crée une carte de chaleur montrant comment le score de collinéarité
varie pour chaque position candidat d'épipole dans l'image.

Usage:
    python collinearity_topographic_map.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from src.utilities.load_ground_truth import get_frame_pixel
from src.utilities.paths import get_labeled_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.utilities.load_flows import load_flows
from src.core.flow_filter import FlowFilterSample
from src.core.collinearity_scorer_sample import CollinearityScorer


def load_frame_data(video_idx: int, frame_idx: int) -> Tuple:
    """
    Charge toutes les données nécessaires pour une frame.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
        
    Returns:
        tuple: (frame_rgb, flow_data, filtered_flow, unfiltered_flow, weights, gt_point)
    """
    print(f"📂 Chargement frame {frame_idx} - vidéo {video_idx}...")
    
    # Load RGB frame
    video_path = get_labeled_dir() / f'{video_idx}.hevc'
    _, frame_rgb = read_frame_rgb(video_path, frame_idx)
    
    # Load flow data
    flow_data = load_flows(video_idx, start_frame=frame_idx-1, end_frame=frame_idx-1)
    flow_data = flow_data[0]  # Remove batch dimension
    
    # Load ground truth
    gt_pixels = get_frame_pixel(video_idx, frame_idx)
    gt_point = (gt_pixels[0], gt_pixels[1])
    
    # Apply filtering with simple configuration
    filter_config = {
        'norm': {'is_used': True, 'k': 150, 'x0': 13},
        'colinearity': {'is_used': True, 'k': 150, 'x0': 0.96},
        'heatmap': {'is_used': False}
    }
    
    flow_filter = FlowFilterSample(filter_config)
    filtered_flow = flow_filter.filter(flow_data)
    unfiltered_flow, weights = flow_filter.filter_and_weight(flow_data)
    
    print(f"✅ Données chargées - GT: ({gt_point[0]:.1f}, {gt_point[1]:.1f})")
    
    return frame_rgb, flow_data, filtered_flow, unfiltered_flow, weights, gt_point


def create_collinearity_topographic_map(video_idx: int = 4, frame_idx: int = 380, 
                                       resolution: int = 30):
    """
    Crée une carte topographique des scores de collinéarité sur toute l'image.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
        resolution: Résolution de la grille (nombre de points par dimension)
    """
    print(f"🎯 CRÉATION CARTE TOPOGRAPHIQUE - Vidéo {video_idx}, Frame {frame_idx}")
    print("=" * 60)
    
    # Load frame data
    frame_rgb, flow_data, filtered_flow, unfiltered_flow, weights, gt_point = load_frame_data(video_idx, frame_idx)
    
    # Get image dimensions
    height, width = frame_rgb.shape[:2]
    
    # Create grid covering the entire image
    print(f"📏 Création grille {resolution}x{resolution} sur image {width}x{height}...")
    x = np.linspace(0, width, resolution)
    y = np.linspace(height, 0, resolution)  # Flip Y axis
    X, Y = np.meshgrid(x, y)
    
    # Calculate collinearity scores for the entire image
    print("🧮 Calcul des scores de collinéarité...")
    estimator = CollinearityScorer()
    Z = np.zeros_like(X)
    
    start_time = time.time()
    total_points = resolution * resolution
    
    for i in range(resolution):
        for j in range(resolution):
            # Calculate collinearity score for this candidate epipole position
            # Use negative value so that better scores (lower values) appear as peaks
            Z[j, i] = -estimator.colin_score(unfiltered_flow, (X[j, i], Y[j, i]), 
                                           step=5, weights=weights)
        
        # Progress indicator
        progress = ((i + 1) * resolution) / total_points * 100
        print(f"\r⏳ Progression: {progress:.1f}%", end="", flush=True)
    
    calc_time = time.time() - start_time
    print(f"\n✅ Calcul terminé en {calc_time:.2f}s")
    
    # Create the visualization
    print("🎨 Création de la visualisation...")
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    frame_rgb = frame_rgb
    
    # Display the frame image as background
    ax.imshow(frame_rgb, extent=[0, width, 0, height], alpha=0.7)
    
    # Plot topological map with transparency over the image
    # Use RdYlGn_r colormap: green for low values (bad scores), red for high values (good scores)
    contour = ax.contourf(X, Y, Z, levels=20, cmap='Blues_r', alpha=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Score de Collinéarité', fontsize=12)
    
    # Plot ground truth point (flip Y coordinate to match new coordinate system)
    gt_y_flipped = height - gt_point[1]
    ax.scatter(gt_point[0], gt_y_flipped, color='red', s=200, marker='*', 
               label='Label', edgecolor='white', linewidth=2, zorder=10)
    
    # Plot center point for reference (flip Y coordinate)
    center_x, center_y = width // 2, height // 2
    center_y_flipped = height - center_y
    ax.scatter(center_x, center_y_flipped, color='orange', s=150, marker='+', 
               label='Centre image', linewidth=3, zorder=10)
    
    # Formatting
    # ax.set_title(f'Carte Topographique des Scores de Collinéarité\nVidéo {video_idx}, Frame {frame_idx}', 
    #             fontsize=14)
    # ax.set_xlabel('Coordonnée X (pixels)', fontsize=12)
    # ax.set_ylabel('Coordonnée Y (pixels)', fontsize=12)
    ax.set_xticks([])  # Remove x-axis values
    ax.set_yticks([])  # Remove y-axis values
    ax.legend(fontsize=11)
    # ax.grid(True, alpha=0.3)
    
    # Add info text
    # info_text = f"Résolution: {resolution}x{resolution}\nTemps calcul: {calc_time:.2f}s\nPoints évalués: {total_points}"
    # ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
    #         fontsize=10, verticalalignment='top', fontfamily='monospace',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualisation terminée")
    
    # Find and report maximum (since we negated the scores, max = best original score)
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    max_x, max_y_flipped = X[max_idx], Y[max_idx]
    max_y_original = height - max_y_flipped  # Convert back to original coordinates
    max_score_neg = Z[max_idx]
    original_score = -max_score_neg  # Convert back to original positive score
    print(f"📍 Meilleur score trouvé: ({max_x:.1f}, {max_y_original:.1f}) avec score original {original_score:.6f}")
    print(f"📐 Distance au GT: {np.sqrt((max_x - gt_point[0])**2 + (max_y_original - gt_point[1])**2):.1f} pixels")


def main():
    """
    Fonction principale.
    """
    print("🗺️  CARTE TOPOGRAPHIQUE DES SCORES DE COLLINÉARITÉ")
    print("=" * 60)
    
    # Paramètres par défaut
    video_idx = 4
    frame_idx = 380
    resolution = 5
    
    # Permettre à l'utilisateur de modifier les paramètres
    print(f"Paramètres par défaut: Vidéo {video_idx}, Frame {frame_idx}, Résolution {resolution}")
    choice = input("Modifier les paramètres ? (o/n, Entrée pour non): ").strip().lower()
    
    if choice in ['o', 'oui', 'y', 'yes']:
        try:
            video_idx = int(input(f"Vidéo (défaut {video_idx}): ") or video_idx)
            frame_idx = int(input(f"Frame (défaut {frame_idx}): ") or frame_idx)
            resolution = int(input(f"Résolution (défaut {resolution}): ") or resolution)
        except ValueError:
            print("Valeurs invalides, utilisation des paramètres par défaut")
    
    create_collinearity_topographic_map(video_idx, frame_idx, resolution)


if __name__ == "__main__":
    main() 