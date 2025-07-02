#!/usr/bin/env python3
"""
Visualisation complète d'une frame avec 3 panneaux côte à côte :
1. Vecteurs de flux optique (adapté de visualize_frame_flow.py)
2. Scores de collinéarité (adapté de visualize_frame_colin.py)  
3. Trajectoire d'optimisation Adam (adapté de visualize_optimizer_single_frame.py)

Configuration par défaut :
- Optimizer Adam avec paramètres par défaut
- Filtre simple avec min_threshold=13 uniquement
- Point de référence pour collinéarité = Ground Truth

Usage:
    python visualize_frame_full.py
"""

import time
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from src.utilities.load_ground_truth import get_frame_pixel, read_ground_truth_pixels
from src.utilities.paths import get_labeled_dir, get_flows_dir, get_pred_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.utilities.load_flows import load_flows
from src.utilities.load_predictions import load_predictions
from src.utilities.worst_errors import get_worst_errors_global, select_random_frames_from_7th_decile
from src.core.flow_filter import FlowFilterBatch, FlowFilterSample
from src.core.collinearity_scorer_batch import BatchCollinearityScorer
from src.core.collinearity_scorer_sample import CollinearityScorer
from src.core.optimizers import AdamOptimizer
import random


def load_frame_data(run_name: str, video_idx: int, frame_idx: int) -> Tuple:
    """
    Charge toutes les données nécessaires pour une frame.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
        
    Returns:
        tuple: (frame_rgb, flow_data, filtered_flow, gt_point, pred_point, center_point)
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
    
    # Load prediction
    pred_point = load_predictions(video_idx, frame_idx, run_name)
    
    # Center point
    center_point = (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2)
    
    # Apply simple filtering (min_threshold=13 only)
    filter_config = {
        'norm': {'is_used': True, 'k': 150, 'x0': 13},
        'colinearity': {'is_used': True, 'k': 150, 'x0': 0.96},
        'heatmap': {'is_used': False}
    }
    
    flow_filter = FlowFilterSample(filter_config)
    filtered_flow = flow_filter.filter(flow_data)
    unfiltered_flow, weights = flow_filter.filter_and_weight(flow_data)
    
    print(f"✅ Données chargées - GT: ({gt_point[0]:.1f}, {gt_point[1]:.1f}), Pred: ({pred_point[0]:.1f}, {pred_point[1]:.1f})")
    
    return frame_rgb, flow_data, filtered_flow, unfiltered_flow, weights, gt_point, pred_point, center_point


def compute_colinearity_for_points(
    pt: Tuple[float, float],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    u_flow: np.ndarray,
    v_flow: np.ndarray
) -> np.ndarray:
    """
    Calcule la collinéarité entre un point de référence et des vecteurs de flux.
    """
    colinearity_values = np.zeros(len(x_coords))
    
    for i in range(len(x_coords)):
        # Vector from reference point to current point
        vec_to_point = np.array([x_coords[i] - pt[0], y_coords[i] - pt[1]])
        
        # Flow vector at current point
        flow_vec = np.array([u_flow[i], v_flow[i]])
        
        # Normalize vectors
        vec_to_point_norm = np.linalg.norm(vec_to_point)
        flow_vec_norm = np.linalg.norm(flow_vec)
        
        # Avoid division by zero
        if vec_to_point_norm > 0 and flow_vec_norm > 0:
            vec_to_point = vec_to_point / vec_to_point_norm
            flow_vec = flow_vec / flow_vec_norm
            
            # Compute dot product (colinearity)
            colinearity = np.abs(np.dot(vec_to_point, flow_vec))
            colinearity_values[i] = colinearity
    
    return colinearity_values


def plot_flow_vectors(ax, frame_rgb: np.ndarray, filtered_flow: np.ndarray, 
                     gt_point: Tuple[float, float], pred_point: Tuple[float, float],
                     center_point: Tuple[float, float], video_idx: int, frame_idx: int, 
                     stride: int = 20):
    """
    Affiche les vecteurs de flux optique avec des flèches sur l'image de fond.
    """
    # Display the original frame as background
    ax.imshow(frame_rgb)
    
    # Set axis limits and invert y-axis (imshow already inverts y-axis)
    ax.set_xlim(0, frame_rgb.shape[1])
    ax.set_ylim(0, frame_rgb.shape[0])
    ax.invert_yaxis()
    # Create grid for arrows
    h, w = filtered_flow.shape[:2]
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Get flow vectors at grid points
    fx = filtered_flow[y, x, 0]
    fy = filtered_flow[y, x, 1]
    
    # Filter out zero or near-zero flow values
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > 0  # Use filtered flow (already has min_threshold applied)
    
    # Plot arrows
    ax.quiver(x[mask], y[mask], fx[mask], fy[mask], 
              color='r', angles='xy', scale_units='xy', scale=1, width=0.001)
    
    # Plot reference points
    ax.scatter(gt_point[0], gt_point[1], color='blue', s=120, marker='*', 
               label='Ground Truth', edgecolor='white', linewidth=1)
    ax.scatter(pred_point[0], pred_point[1], color='magenta', s=120, marker='o', 
               label='Ancienne prédiction', edgecolor='white', linewidth=1)
    ax.scatter(center_point[0], center_point[1], color='green', s=100, marker='+', 
               label='Center', linewidth=3)
    
    ax.set_title(f'Vecteurs de Flux\nVidéo {video_idx}, Frame {frame_idx}')
    ax.legend()
    ax.axis('off')


def plot_collinearity_scores(ax, frame_rgb: np.ndarray, filtered_flow: np.ndarray,
                           gt_point: Tuple[float, float], pred_point: Tuple[float, float],
                           center_point: Tuple[float, float], video_idx: int, frame_idx: int, 
                           stride: int = 2):
    """
    Affiche les scores de collinéarité par rapport au point GT.
    """
    # Set axis limits and invert y-axis  
    ax.set_xlim(0, frame_rgb.shape[1])
    ax.set_ylim(0, frame_rgb.shape[0])
    ax.invert_yaxis()
    
    # Create grid for points
    h, w = filtered_flow.shape[:2]
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Get flow vectors at grid points
    fx = filtered_flow[y, x, 0]
    fy = filtered_flow[y, x, 1]
    
    # Filter out zero flow values
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > 0
    
    # Compute colinearity for valid flow vectors
    colinearity = compute_colinearity_for_points(
        gt_point,
        x[mask],
        y[mask],
        fx[mask],
        fy[mask]
    )
    
    # Plot colinearity scores
    scatter = ax.scatter(
        x[mask],
        y[mask],
        c=colinearity,
        cmap='viridis',
        alpha=0.6,
        s=15
    )
    
    # Add colorbar to the right of this subplot
    plt.colorbar(scatter, ax=ax, label='Score Collinéarité', shrink=0.8)
    
    # Plot reference points
    ax.scatter(gt_point[0], gt_point[1], color='blue', s=120, marker='*', 
               label='Ground Truth', edgecolor='white', linewidth=1)
    ax.scatter(pred_point[0], pred_point[1], color='magenta', s=120, marker='o', 
               label='Ancienne prédiction', edgecolor='white', linewidth=1)
    ax.scatter(center_point[0], center_point[1], color='green', s=120, marker='+', 
               label='Center', linewidth=3)
    
    ax.set_title(f'Scores Collinéarité\nVidéo {video_idx}, Frame {frame_idx}')
    ax.legend()
    ax.axis('off')


def plot_optimizer_trajectory(ax, flow: np.ndarray, weights: np.ndarray,
                            gt_point: Tuple[float, float], pred_point: Tuple[float, float],
                            center_point: Tuple[float, float], video_idx: int, frame_idx: int, 
                            image_shape: Tuple[int, int]):
    """
    Lance l'optimisation Adam et affiche la trajectoire sur une carte topologique.
    """
    print("🚀 Lancement optimisation Adam...")
    
    # Convert to MLX for Adam optimizer
    flow_mx = mx.array(flow, dtype=mx.float32)
    weights_mx = mx.array(weights, dtype=mx.float32)
    
    # Run Adam optimization with default parameters
    adam_optimizer = AdamOptimizer(plateau_threshold=0.0001, plateau_patience=10, pixel_patience=0)
    start_point = mx.array([flow.shape[1] // 2, flow.shape[0] // 2], dtype=mx.float32)
    
    start_time = time.time()
    adam_point = adam_optimizer.optimize_single(
        flow_mx,
        starting_point=start_point,
        weights=weights_mx,
        save_trajectories=True
    )
    adam_time = time.time() - start_time
    
    adam_point_np = (float(adam_point[0]), float(adam_point[1]))
    
    print(f"✅ Adam terminé - Point: ({adam_point_np[0]:.1f}, {adam_point_np[1]:.1f}), Temps: {adam_time:.3f}s")
    
    # Create topological map around points of interest
    all_x = [gt_point[0], pred_point[0], adam_point_np[0], center_point[0]]
    all_y = [gt_point[1], pred_point[1], adam_point_np[1], center_point[1]]
    
    margin = 50
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    
    # Create grid for topological map (reduced resolution for speed)
    x = np.linspace(x_min, x_max, 15)
    y = np.linspace(y_min, y_max, 12)
    X, Y = np.meshgrid(x, y)
    
    # Calculate collinearity scores for topological map
    estimator = CollinearityScorer()
    Z = np.zeros_like(X)
    
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = estimator.colin_score(flow, (X[j, i], Y[j, i]), step=5, weights=weights)
    
    # Plot topological map
    contour = ax.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
    
    # Plot trajectory
    if len(adam_optimizer.trajectory) > 1:
        traj = np.array(adam_optimizer.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'r.-', linewidth=3, markersize=8, 
                label='Trajectoire Adam', alpha=0.9)
        ax.annotate('Start', traj[0], color='red', fontsize=10, fontweight='bold')
    
    # Plot final points
    ax.scatter(adam_point_np[0], adam_point_np[1], color='red', s=150, marker='o', 
               label='Nouvelle prédiction Adam', edgecolor='white', linewidth=2)
    ax.scatter(gt_point[0], gt_point[1], color='lime', s=180, marker='*', 
               label='Ground Truth', edgecolor='black', linewidth=2)
    ax.scatter(pred_point[0], pred_point[1], color='magenta', s=150, marker='s', 
               label='Ancienne prédiction', edgecolor='white', linewidth=2)
    ax.scatter(center_point[0], center_point[1], color='orange', s=120, marker='+', 
               label='Center', linewidth=3)
    
    ax.set_title(f'Trajectoire Adam\nVidéo {video_idx}, Frame {frame_idx}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.invert_yaxis()  # Match image coordinate system
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add info text
    distance_to_gt = np.sqrt((adam_point_np[0] - gt_point[0])**2 + (adam_point_np[1] - gt_point[1])**2)
    info_text = f"Distance GT: {distance_to_gt:.1f}px\nTemps: {adam_time:.3f}s\nItérations: {len(adam_optimizer.trajectory)}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))


def visualize_frame_full(run_name: str, video_idx: int = 1, frame_idx: int = 201):
    """
    Visualisation complète avec 3 panneaux côte à côte.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
    """
    print(f"🎯 VISUALISATION COMPLÈTE - Run: {run_name}, Vidéo {video_idx}, Frame {frame_idx}")
    print("=" * 60)
    # print(f"Run: {run_name}, Video: {video_idx}, Frame: {frame_idx}")
    # Load frame data
    frame_rgb, flow_data, filtered_flow, unfiltered_flow, weights, gt_point, pred_point, center_point = load_frame_data(run_name, video_idx, frame_idx)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Plot 1: Flow vectors
    plot_flow_vectors(axes[0], frame_rgb, filtered_flow, gt_point, pred_point, center_point, video_idx, frame_idx)
    
    # Plot 2: Collinearity scores  
    plot_collinearity_scores(axes[1], frame_rgb, filtered_flow, gt_point, pred_point, center_point, video_idx, frame_idx)
    
    # Plot 3: Optimizer trajectory
    plot_optimizer_trajectory(axes[2], unfiltered_flow, weights, gt_point, pred_point, center_point, video_idx, frame_idx, flow_data.shape)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualisation terminée")


def visualize_worst_errors_sequential(run_name="vanilla", k=10):
    """
    Visualise séquentiellement les k pires erreurs globales.
    
    Args:
        run_name: Nom du run à analyser
        k: Nombre de pires erreurs à visualiser
    """
    print(f"🔍 Récupération des {k} pires erreurs pour le run '{run_name}'...")
    
    try:
        worst_errors_coords = get_worst_errors_global(run_name, k)
        print(f"✅ {len(worst_errors_coords)} pires erreurs trouvées")
        
        for i, (video_id, frame_id) in enumerate(worst_errors_coords):
            print(f"\n{'='*60}")
            print(f"📊 PIRE ERREUR {i+1}/{len(worst_errors_coords)}")
            print(f"📹 Vidéo {video_id}, Frame {frame_id}")
            print(f"{'='*60}")
            
            # Appel de la fonction de visualisation de base
            visualize_frame_full(run_name=run_name, video_idx=video_id, frame_idx=frame_id)
            
            # Pause entre les visualisations si pas la dernière
            # if i < len(worst_errors_coords) - 1:
            #     input("Appuyez sur Entrée pour continuer vers la prochaine frame...")
        
        print(f"\n✅ Visualisation de toutes les {len(worst_errors_coords)} pires erreurs terminée")
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des pires erreurs: {e}")
        print("Retour à la visualisation par défaut...")
        visualize_frame_full(run_name=run_name, video_idx=1, frame_idx=201)




def visualize_random_7th_decile_frames(run_name="5"):
    """Visualise 10 frames aléatoires du 7ème décile."""
    selected_frames = select_random_frames_from_7th_decile(run_name, 10)
    
    if not selected_frames:
        return
    
    print(f"\n🖼️  Visualisation de {len(selected_frames)} frames du 7ème décile...")
    
    # Utiliser la fonction de visualisation existante
    for i, (video_id, frame_id) in enumerate(selected_frames, 1):
        print(f"\n--- Frame {i}/{len(selected_frames)} ---")
        visualize_frame_full(video_id, frame_id)
        
        # Demander si continuer (sauf pour la dernière)
        # if i < len(selected_frames):
        #     continue_choice = input("\nContinuer vers la frame suivante ? (o/n): ").strip().lower()
        #     if continue_choice not in ['o', 'oui', 'y', 'yes', '']:
        #         print("Arrêt de la visualisation.")
        #         break

def main(video_idx: int = 1, frame_idx: int = 202):
    """
    Fonction principale - choix entre visualisation par défaut, pires erreurs ou 7ème décile.
    """
    print("🎯 VISUALISATION COMPLÈTE DE FRAMES")
    print("=" * 60)
    print(f"1. Visualisation d'une frame (Vidéo {video_idx}, Frame {frame_idx})")
    print("2. Visualiser les pires erreurs séquentiellement")
    print("3. Visualiser 10 frames aléatoires du 7ème décile")
    print("=" * 60)
    
    choice = input("Votre choix (1, 2 ou 3, Entrée pour par défaut): ").strip()
    
    if choice == "2":
        # Visualiser les pires erreurs
        run_name = input("Nom du run (défaut: 'vanilla'): ").strip()
        if not run_name:
            run_name = "vanilla"
        k_input = input("Nombre de pires erreurs (défaut: 10): ").strip()
        try:
            k = int(k_input) if k_input else 10
        except ValueError:
            k = 10
        visualize_worst_errors_sequential(run_name, k)
    elif choice == "3":
        run_name = input("Nom du run (défaut: '5'): ").strip()
        if not run_name:
            run_name = "5"
        visualize_random_7th_decile_frames(run_name)
    else:
        # Configuration par défaut
        run_name = "5_4"
        print(f"📊 Visualisation d'une frame: Vidéo {video_idx}, Frame {frame_idx}")
        visualize_frame_full(run_name=run_name, video_idx=video_idx, frame_idx=frame_idx)


if __name__ == "__main__":
    main(video_idx=3, frame_idx=119) 