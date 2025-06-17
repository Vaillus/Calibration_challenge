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

from src.utilities.ground_truth import get_frame_pixel, read_ground_truth_pixels
from src.utilities.paths import get_labeled_dir, get_flows_dir, get_pred_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.utilities.load_flows import load_flows
from src.utilities.worst_errors_reader import read_worst_errors
from src.utilities.load_predictions import load_predictions
from src.utilities.pixel_angle_converter import angles_to_pixels
from src.core.flow_filter import FlowFilterBatch, FlowFilterSample
from src.core.collinearity_scorer_batch import BatchCollinearityScorer
from src.core.collinearity_scorer_sample import CollinearityScorer
from src.core.optimizers import AdamOptimizer
import random


def load_frame_data(video_idx: int, frame_idx: int) -> Tuple:
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
    flow_data = load_flows(video_idx, start_frame=frame_idx, end_frame=frame_idx)
    flow_data = flow_data[0]  # Remove batch dimension
    
    # Load ground truth
    gt_pixels = get_frame_pixel(video_idx, frame_idx)
    gt_point = (gt_pixels[0], gt_pixels[1])
    
    # Load prediction
    pred_point = load_predictions(video_idx, frame_idx)
    
    # Center point
    center_point = (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2)
    
    # Apply simple filtering (min_threshold=13 only)
    filter_config = {
        'filtering': {
            'norm': {'is_used': True, 'min_threshold': 13},
            "colinearity": {
                "is_used": True,
                "min_threshold": 0.96
            }
        }
    }
    
    flow_filter = FlowFilterSample(filter_config)
    filtered_flow = flow_filter.filter(flow_data)
    
    print(f"✅ Données chargées - GT: ({gt_point[0]:.1f}, {gt_point[1]:.1f}), Pred: ({pred_point[0]:.1f}, {pred_point[1]:.1f})")
    
    return frame_rgb, flow_data, filtered_flow, gt_point, pred_point, center_point


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


def plot_optimizer_trajectory(ax, filtered_flow: np.ndarray, 
                            gt_point: Tuple[float, float], pred_point: Tuple[float, float],
                            center_point: Tuple[float, float], video_idx: int, frame_idx: int, 
                            image_shape: Tuple[int, int]):
    """
    Lance l'optimisation Adam et affiche la trajectoire sur une carte topologique.
    """
    print("🚀 Lancement optimisation Adam...")
    
    # Convert to MLX for Adam optimizer
    flow_mx = mx.array(filtered_flow, dtype=mx.float32)
    
    # Run Adam optimization with default parameters
    adam_optimizer = AdamOptimizer()
    start_point = mx.array([filtered_flow.shape[1] // 2, filtered_flow.shape[0] // 2], dtype=mx.float32)
    
    start_time = time.time()
    adam_point = adam_optimizer.optimize_single(
        flow_mx,
        starting_point=start_point,
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
            Z[j, i] = estimator.colin_score(filtered_flow, (X[j, i], Y[j, i]), step=5)
    
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


def visualize_frame_full(video_idx: int = 1, frame_idx: int = 201):
    """
    Visualisation complète avec 3 panneaux côte à côte.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
    """
    print(f"🎯 VISUALISATION COMPLÈTE - Vidéo {video_idx}, Frame {frame_idx}")
    print("=" * 60)
    
    # Load frame data
    frame_rgb, flow_data, filtered_flow, gt_point, pred_point, center_point = load_frame_data(video_idx, frame_idx)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Plot 1: Flow vectors
    plot_flow_vectors(axes[0], frame_rgb, filtered_flow, gt_point, pred_point, center_point, video_idx, frame_idx)
    
    # Plot 2: Collinearity scores  
    plot_collinearity_scores(axes[1], frame_rgb, filtered_flow, gt_point, pred_point, center_point, video_idx, frame_idx)
    
    # Plot 3: Optimizer trajectory
    plot_optimizer_trajectory(axes[2], filtered_flow, gt_point, pred_point, center_point, video_idx, frame_idx, flow_data.shape)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualisation terminée")


def visualize_worst_errors_sequential(filename: str = "worst_10_global_errors_5.json"):
    """
    Visualise séquentiellement toutes les frames des pires erreurs.
    
    Args:
        filename: Nom du fichier JSON contenant les pires erreurs
    """
    print(f"🔍 Chargement des pires erreurs depuis {filename}...")
    
    try:
        worst_errors_coords = read_worst_errors(filename)
        print(f"✅ {len(worst_errors_coords)} pires erreurs trouvées")
        
        for i, (video_id, frame_id) in enumerate(worst_errors_coords):
            print(f"\n{'='*60}")
            print(f"📊 PIRE ERREUR {i+1}/{len(worst_errors_coords)}")
            print(f"📹 Vidéo {video_id}, Frame {frame_id}")
            print(f"{'='*60}")
            
            # Appel de la fonction de visualisation de base
            visualize_frame_full(video_idx=video_id, frame_idx=frame_id)
            
            # Pause entre les visualisations si pas la dernière
            # if i < len(worst_errors_coords) - 1:
            #     input("Appuyez sur Entrée pour continuer vers la prochaine frame...")
        
        print(f"\n✅ Visualisation de toutes les {len(worst_errors_coords)} pires erreurs terminée")
        
    except FileNotFoundError:
        print(f"❌ Fichier {filename} non trouvé")
        print("Retour à la visualisation par défaut...")
        visualize_frame_full(video_idx=1, frame_idx=201)
    except Exception as e:
        print(f"❌ Erreur lors du chargement des pires erreurs: {e}")
        print("Retour à la visualisation par défaut...")
        visualize_frame_full(video_idx=1, frame_idx=201)


def select_random_frames_from_7th_decile(run_name="5", n_frames=10):
    """
    Sélectionne n frames aléatoires du 7ème décile de la distribution des distances.
    
    Args:
        run_name: Nom du run à analyser
        n_frames: Nombre de frames à sélectionner
        
    Returns:
        List[Tuple[int, int]]: Liste de (video_id, frame_id)
    """
    print(f"🎲 Sélection de {n_frames} frames aléatoires du 7ème décile...")
    
    # 1. Calculer toutes les distances avec coordonnées
    all_distances_with_coords = []
    
    for video_id in range(5):
        # Charger données (même logique que SaveWorstError)
        gt_path = get_labeled_dir() / f"{video_id}.txt"
        pred_path = get_pred_dir(run_name) / f"{video_id}.txt"
        
        if not gt_path.exists() or not pred_path.exists():
            continue
            
        gt_data = np.loadtxt(gt_path)
        pred_data = np.loadtxt(pred_path)
        
        min_frames = min(len(gt_data), len(pred_data))
        
        for i in range(min_frames):
            if np.isnan(pred_data[i]).any() or np.isnan(gt_data[i]).any():
                continue
                
            # Calculer distance
            pitch_gt, yaw_gt = gt_data[i]
            x_gt, y_gt = angles_to_pixels(yaw_gt, pitch_gt, 910, 1920, 1080)
            
            pitch_pred, yaw_pred = pred_data[i]
            x_pred, y_pred = angles_to_pixels(yaw_pred, pitch_pred, 910, 1920, 1080)
            
            distance = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
            
            all_distances_with_coords.append((distance, video_id, i))
    
    if not all_distances_with_coords:
        print("❌ Aucune distance calculée!")
        return []
    
    # 2. Calculer les percentiles
    distances = [d[0] for d in all_distances_with_coords]
    p70 = np.percentile(distances, 70)  # 7ème décile début
    p80 = np.percentile(distances, 80)  # 7ème décile fin
    
    print(f"📊 7ème décile: {p70:.1f} - {p80:.1f} pixels")
    
    # 3. Filtrer les frames dans le 7ème décile
    frames_in_7th_decile = [
        (video_id, frame_id) 
        for dist, video_id, frame_id in all_distances_with_coords
        if p70 <= dist <= p80
    ]
    
    print(f"🎯 {len(frames_in_7th_decile)} frames dans le 7ème décile")
    
    if len(frames_in_7th_decile) < n_frames:
        print(f"⚠️  Seulement {len(frames_in_7th_decile)} frames disponibles")
        return frames_in_7th_decile
    
    # 4. Sélectionner aléatoirement
    selected_frames = random.sample(frames_in_7th_decile, n_frames)
    
    print(f"✅ {n_frames} frames sélectionnés aléatoirement:")
    for i, (vid, frame) in enumerate(selected_frames, 1):
        print(f"   {i}. Video {vid}, Frame {frame}")
    
    return selected_frames

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

def main():
    """
    Fonction principale - choix entre visualisation par défaut, pires erreurs ou 7ème décile.
    """
    print("🎯 VISUALISATION COMPLÈTE DE FRAMES")
    print("=" * 60)
    print("1. Visualisation par défaut (Vidéo 1, Frame 201)")
    print("2. Visualiser les pires erreurs séquentiellement")
    print("3. Visualiser 10 frames aléatoires du 7ème décile")
    print("=" * 60)
    
    choice = input("Votre choix (1, 2 ou 3, Entrée pour par défaut): ").strip()
    
    if choice == "2":
        # Visualiser les pires erreurs
        visualize_worst_errors_sequential()
    elif choice == "3":
        run_name = input("Nom du run (défaut: '5'): ").strip()
        if not run_name:
            run_name = "5"
        visualize_random_7th_decile_frames(run_name)
    else:
        # Configuration par défaut
        VIDEO_IDX = 1
        FRAME_IDX = 201
        print(f"📊 Visualisation par défaut: Vidéo {VIDEO_IDX}, Frame {FRAME_IDX}")
        visualize_frame_full(video_idx=VIDEO_IDX, frame_idx=FRAME_IDX)


if __name__ == "__main__":
    main() 