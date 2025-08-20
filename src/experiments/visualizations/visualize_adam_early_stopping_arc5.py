#!/usr/bin/env python3
"""
Visualisation des trajectoires d'optimisation Adam - Arc 5.

Compare 2 optimiseurs Adam pour l'estimation de points de fuite (configuration Arc 5) :
- Adam avec early stopping sur plateau (détection de plateau)
- Adam avec early stopping sur pixel (pixel patience)

Utilise les paramètres de filtrage sigmoïdaux de l'arc 5.
Affiche les trajectoires sur une carte topologique avec valeurs négatives du score de colinéarité.
Configurez VIDEO_ID et FRAME_IDX dans main() pour changer la frame analysée.
"""

import time
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt

from src.utilities.load_ground_truth import read_ground_truth_pixels
from src.utilities.paths import get_flows_dir, get_labeled_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.core.flow_filter import FlowFilterSample
from src.core.collinearity_scorer_batch import BatchCollinearityScorer
from src.core.collinearity_scorer_sample import CollinearityScorer
from src.core.optimizers import AdamOptimizer


def load_frame_data(video_id, frame_idx):
    """
    Charge toutes les données nécessaires pour une frame donnée.
    
    Args:
        video_id: ID de la vidéo (0-4)
        frame_idx: Index de la frame
    
    Returns:
        tuple: (frame_rgb, flow_data, gt_point, filtered_flow, weights)
    """
    print(f"📂 Chargement de la frame {frame_idx} (vidéo {video_id})...")
    
    # Load RGB frame
    
    from src.utilities.load_flows import load_flows
    flow_data_batch = load_flows(video_id, use_compressed=False, start_frame=frame_idx, end_frame=frame_idx)
    
    # Extract the single frame from the batch (load_flows returns a batch even for single frame)
    flow_data = flow_data_batch[0] if flow_data_batch is not None else None
    
    # Load ground truth
    gt_pixels = read_ground_truth_pixels(video_id)
    gt_point = gt_pixels[frame_idx + 1]  # GT is 1-indexed
    
    # Apply Arc 5 filtering (sigmoid filters with optimized parameters)
    flow_np = np.array(flow_data)
    filter_config = {
        'norm': {'is_used': True, 'k': 180, 'x0': 8},
        'colinearity': {'is_used': True, 'k': 152, 'x0': 1.245},
        'heatmap': {'is_used': False}
    }
    flow_filter = FlowFilterSample(filter_config)
    filtered_flow_np, weights_np = flow_filter.filter_and_weight(flow_np)
    
    # Convert back to MLX for compatibility with optimizers
    filtered_flow = mx.array(filtered_flow_np, dtype=mx.float32)
    weights = mx.array(weights_np, dtype=mx.float32) if weights_np is not None else None
    
    print(f"✅ Frame {frame_idx} (vidéo {video_id}) chargée")
    print(f"   Flow shape: {flow_data.shape}")
    print(f"   GT point: ({gt_point[0]:.1f}, {gt_point[1]:.1f})")
    
    return flow_data, gt_point, filtered_flow, weights


def run_two_adam_optimizations(filtered_flow, weights):
    """
    Lance les 2 optimisations Adam de l'arc 5 et récupère les trajectoires.
    
    Returns:
        dict: Résultats avec trajectoires et temps d'exécution
    """
    print("\n🚀 Lancement des 2 optimisations Adam Arc 5...")
    
    results = {}
    
    # ===== 1. Adam AVEC early stopping sur plateau =====
    print("--- 1. Adam (AVEC early stopping sur plateau) ---")
    adam_plateau_optimizer = AdamOptimizer(
        plateau_threshold=1e-4,  # Seuil pour détecter un plateau
        plateau_patience=3,      # Nombre d'itérations de patience
        pixel_patience=0         # Pas de pixel patience
    )
    
    start_point = mx.array([filtered_flow.shape[1] // 2, filtered_flow.shape[0] // 2], dtype=mx.float32)
    
    start_time = time.time()
    adam_plateau_point = adam_plateau_optimizer.optimize_single(
        filtered_flow, 
        starting_point=start_point,
        save_trajectories=True
    )
    adam_plateau_time = time.time() - start_time
    
    adam_estimator = BatchCollinearityScorer()
    adam_plateau_score = float(adam_estimator.colin_score(filtered_flow, adam_plateau_point, weights=weights))
    
    results['adam_plateau'] = {
        'point': (float(adam_plateau_point[0]), float(adam_plateau_point[1])),
        'score': adam_plateau_score,
        'time': adam_plateau_time,
        'trajectory': adam_plateau_optimizer.trajectory.copy(),
        'scores': adam_plateau_optimizer.scores.copy(),
        'method': 'Adam (plateau early stopping)'
    }
    
    print(f"  Point: ({float(adam_plateau_point[0]):.2f}, {float(adam_plateau_point[1]):.2f})")
    print(f"  Score: {adam_plateau_score:.6f}")
    print(f"  Temps: {adam_plateau_time:.4f}s")
    print(f"  Itérations: {len(results['adam_plateau']['trajectory'])}")
    
    # ===== 2. Adam AVEC early stopping sur pixel =====
    print("--- 2. Adam (AVEC early stopping sur pixel) ---")
    adam_pixel_optimizer = AdamOptimizer(
        plateau_threshold=0,     # Pas de plateau detection
        plateau_patience=0,      # Pas de plateau patience
        pixel_patience=5         # Pixel patience (paramètres par défaut)
    )
    
    start_point = mx.array([filtered_flow.shape[1] // 2, filtered_flow.shape[0] // 2], dtype=mx.float32)
    
    start_time = time.time()
    adam_pixel_point = adam_pixel_optimizer.optimize_single(
        filtered_flow, 
        starting_point=start_point,
        save_trajectories=True
    )
    adam_pixel_time = time.time() - start_time
    
    adam_pixel_score = float(adam_estimator.colin_score(filtered_flow, adam_pixel_point, weights=weights))
    
    results['adam_pixel'] = {
        'point': (float(adam_pixel_point[0]), float(adam_pixel_point[1])),
        'score': adam_pixel_score,
        'time': adam_pixel_time,
        'trajectory': adam_pixel_optimizer.trajectory.copy(),
        'scores': adam_pixel_optimizer.scores.copy(),
        'method': 'Adam (pixel early stopping)'
    }
    
    print(f"  Point: ({float(adam_pixel_point[0]):.2f}, {float(adam_pixel_point[1]):.2f})")
    print(f"  Score: {adam_pixel_score:.6f}")
    print(f"  Temps: {adam_pixel_time:.4f}s")
    print(f"  Itérations: {len(results['adam_pixel']['trajectory'])}")
    
    # ===== Comparaison des performances =====
    print(f"\n🏁 COMPARAISON DES PERFORMANCES:")
    times = [results['adam_plateau']['time'], results['adam_pixel']['time']]
    fastest_time = min(times)
    
    for key, result in results.items():
        speedup = fastest_time / result['time']
        if speedup == 1.0:
            speed_info = "⚡ Plus rapide"
        else:
            speed_info = f"{1/speedup:.1f}x plus lent"
        print(f"  {result['method']}: {result['time']:.4f}s ({speed_info})")
    
    return results


def plot_trajectories_on_topological_map(filtered_flow, weights, gt_point, results, image_shape, video_id=4, frame_idx=266, show_topological_map=True):
    """
    Visualisation : trajectoires des optimiseurs Adam sur carte topologique avec vecteurs de flux.
    Utilise la valeur NEGATIVE du score de colinéarité pour la carte topologique.
    
    Args:
        show_topological_map: Si True, calcule et affiche la carte topologique (lent). Si False, affiche seulement les trajectoires.
    """
    print("\n📊 Création des trajectoires Adam sur carte topologique...")
    
    # Convertir pour le calcul des scores
    filtered_flow_np = np.array(filtered_flow)
    weights_np = np.array(weights) if weights is not None else None
    
    estimator = CollinearityScorer()
    
    # Centre de l'image
    center_x = image_shape[1] // 2
    center_y = image_shape[0] // 2
    
    # Définir la zone d'intérêt autour des points trouvés
    all_x = [gt_point[0], results['adam_plateau']['point'][0], results['adam_pixel']['point'][0], center_x]
    all_y = [gt_point[1], results['adam_plateau']['point'][1], results['adam_pixel']['point'][1], center_y]
    
    # Extend the range a bit
    margin = 50
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    
    # Créer le plot
    plt.figure(figsize=(16, 12))
    
    # Carte topologique (optionnelle pour itération rapide)
    if show_topological_map:
        # Créer la grille pour la carte topologique (résolution réduite pour vitesse)
        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 15)
        X, Y = np.meshgrid(x, y)
        
        print("  Calcul de la carte topologique (valeurs négatives)...")
        Z = np.zeros_like(X)
        total_points = len(x) * len(y)
        current_point = 0
        
        for i in range(len(x)):
            for j in range(len(y)):
                # UTILISER LA VALEUR NEGATIVE du score de colinéarité
                Z[j, i] = -estimator.colin_score(filtered_flow_np, (X[j, i], Y[j, i]), weights=weights_np, step=5)
                current_point += 1
                if current_point % 50 == 0:
                    print(f"    Progress: {current_point}/{total_points} ({100*current_point/total_points:.0f}%)")
        
        print("  ✅ Carte topologique calculée")
        
        # Afficher la carte topologique
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, label='Score de Collinéarité')
    else:
        print("  ⚡ Carte topologique désactivée pour itération rapide")
    
    # Ajouter les vecteurs de flux optique dans la zone d'intérêt
    print("  Ajout des vecteurs de flux...")
    
    # Créer une grille pour les vecteurs de flux dans la zone d'intérêt
    stride = 15  # Espacement entre les flèches
    h, w = filtered_flow_np.shape[:2]
    
    # Limiter aux coordonnées dans la zone d'intérêt et dans l'image
    y_start = max(0, int(y_min))
    y_end = min(h, int(y_max))
    x_start = max(0, int(x_min))
    x_end = min(w, int(x_max))
    
    # Créer la grille pour les flèches dans la zone d'intérêt
    y_grid, x_grid = np.mgrid[y_start:y_end:stride, x_start:x_end:stride]
    y_flat = y_grid.flatten()
    x_flat = x_grid.flatten()
    
    # Récupérer les vecteurs de flux aux points de la grille
    flow_x = filtered_flow_np[y_flat, x_flat, 0]
    flow_y = filtered_flow_np[y_flat, x_flat, 1]
    
    # Filtrer les vecteurs significatifs
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    mask = magnitude > 1.0  # Seuil plus bas pour avoir plus de vecteurs
    
    # Tracer les flèches de flux
    plt.quiver(x_flat[mask], y_flat[mask], flow_x[mask], flow_y[mask], 
              color='white', angles='xy', scale_units='xy', scale=1, 
              width=0.002, alpha=0.4, zorder=1)
    
    # Trajectoires (zorder plus élevé pour être au-dessus)
    if len(results['adam_plateau']['trajectory']) > 1:
        traj_adam_plateau = np.array(results['adam_plateau']['trajectory'])
        plt.plot(traj_adam_plateau[:, 0], traj_adam_plateau[:, 1], 'b.-', linewidth=1.5, markersize=6, 
                label='Trajectoire Adam (plateau early stopping)', alpha=0.9, zorder=4)
    
    if len(results['adam_pixel']['trajectory']) > 1:
        traj_adam_pixel = np.array(results['adam_pixel']['trajectory'])
        plt.plot(traj_adam_pixel[:, 0], traj_adam_pixel[:, 1], 'm.-', linewidth=1.5, markersize=6, 
                label='Trajectoire Adam (pixel early stopping)', alpha=0.9, zorder=4)
    
    # Points finaux et GT (zorder élevé pour être bien visibles)
    plt.scatter(results['adam_plateau']['point'][0], results['adam_plateau']['point'][1], 
               color='blue', s=120, marker='s', label='Adam (plateau), position finale', 
               edgecolor='white', linewidth=1, zorder=10)
    plt.scatter(results['adam_pixel']['point'][0], results['adam_pixel']['point'][1], 
               color='magenta', s=120, marker='*', label='Adam (pixel), position finale', 
               edgecolor='white', linewidth=1, zorder=10)
    plt.scatter(gt_point[0], gt_point[1], color='lime', s=120, marker='*', 
               label='Label', edgecolor='black', linewidth=1, zorder=10)
    
    # Centre de l'image
    plt.scatter(center_x, center_y, color='orange', s=150, marker='+', 
               label='Centre Image', linewidth=2.5, zorder=10)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    
    # INVERSER l'axe Y pour correspondre au système de coordonnées image
    plt.gca().invert_yaxis()
    
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Ajouter des infos dans un coin
    info_text = f"""Scores finaux:
Adam (plateau): {-results['adam_plateau']['score']:.6f}
Adam (pixel): {-results['adam_pixel']['score']:.6f}
"""
    
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualisation terminée")


def main(video_id=4, frame_idx=266, show_topological_map=True):
    """
    Fonction principale - Analyse comparative des 2 optimiseurs Adam Arc 5.
    
    Args:
        video_id: ID de la vidéo à analyser (0-4)
        frame_idx: Index de la frame à analyser
        show_topological_map: Si True, calcule et affiche la carte topologique (lent). Si False, itération rapide.
    """
    print(f"🎯 ANALYSE FRAME {frame_idx} - VIDÉO {video_id} (Arc 5 - Adam Early Stopping)")
    print("=" * 70)
    print("Comparaison de 2 optimiseurs Adam Arc 5 :")
    print("  1. Adam (avec early stopping sur plateau)")
    print("  2. Adam (avec early stopping sur pixel)")
    print("=" * 70)
    
    # 1. Charger les données
    flow_data, gt_point, filtered_flow, weights = load_frame_data(video_id, frame_idx)
    
    # 2. Lancer les optimisations
    results = run_two_adam_optimizations(filtered_flow, weights)
    
    # 3. Visualisation : trajectoires sur carte topologique
    plot_trajectories_on_topological_map(filtered_flow, weights, gt_point, results, flow_data.shape, video_id, frame_idx, show_topological_map)
    
    # 4. Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ ARC 5 - ADAM EARLY STOPPING")
    print("=" * 70)
    
    adam_plateau_dist = np.sqrt((results['adam_plateau']['point'][0] - gt_point[0])**2 + 
                               (results['adam_plateau']['point'][1] - gt_point[1])**2)
    adam_pixel_dist = np.sqrt((results['adam_pixel']['point'][0] - gt_point[0])**2 + 
                             (results['adam_pixel']['point'][1] - gt_point[1])**2)
    
    print(f"Adam (plateau): Distance GT = {adam_plateau_dist:.1f}px, Score = {results['adam_plateau']['score']:.6f}")
    print(f"Adam (pixel): Distance GT = {adam_pixel_dist:.1f}px, Score = {results['adam_pixel']['score']:.6f}")
    
    # Identifier le meilleur
    distances = [adam_plateau_dist, adam_pixel_dist] 
    methods = ['Adam (plateau)', 'Adam (pixel)']
    best_idx = np.argmin(distances)
    
    print(f"\n🎯 MEILLEUR: {methods[best_idx]} avec {distances[best_idx]:.1f} pixels d'erreur")
    
    # Comparaisons
    print(f"\nComparaison :")
    print(f"  Différence de distance: {adam_plateau_dist - adam_pixel_dist:+.1f} pixels")
    print(f"  Différence de score: {results['adam_plateau']['score'] - results['adam_pixel']['score']:+.6f}")


if __name__ == "__main__":
    # Vous pouvez modifier ces valeurs pour analyser d'autres vidéos/frames
    VIDEO_ID = 1    # ID de la vidéo (0-4) 
    FRAME_IDX = 489  # Index de la frame
    SHOW_TOPOLOGICAL_MAP = False  # True = avec carte topologique (lent), False = itération rapide
    
    main(video_id=VIDEO_ID, frame_idx=FRAME_IDX, show_topological_map=SHOW_TOPOLOGICAL_MAP) 