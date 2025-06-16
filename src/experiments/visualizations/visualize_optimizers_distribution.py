#!/usr/bin/env python3
"""
Visualisation des distributions des optimiseurs.

Ce script compare les performances et distributions spatiales de différents optimiseurs :
1. L-BFGS-B (optimiseur de référence avec paramètres par défaut)
2. Adam AVEC détection de plateau (plateau_threshold=1e-4, plateau_patience=3)
3. Adam AVEC détection de pixel patience (pixel_patience=5, plateau_threshold=0)

Le script génère :
- Des visualisations des distributions spatiales des prédictions
- Des comparaisons de performance (temps, précision, stabilité)
- Des histogrammes des distances aux ground truth
- Des analyses statistiques détaillées

Usage:
    python visualize_optimizers_distribution.py

Configuration par défaut : Vidéo 4, frames 250-299 (50 échantillons)
"""

import time
import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pathlib import Path

# Imports absolus du projet
from src.utilities.load_flows import load_flows
from src.utilities.ground_truth import read_ground_truth_pixels
from src.core.optimizers import AdamOptimizer, LBFGSOptimizer
from src.core.flow_filter import FlowFilterBatch


def load_and_filter_data(video_idx=4, start_frame=250, end_frame=299):
    """
    Charge et filtre les données pour le test.
    
    Returns:
        tuple: (filtered_flows, labels_data)
    """
    print("🔄 Chargement et filtrage des données...")
    
    # Chargement des flows
    flows_data = load_flows(
        video_index=video_idx, 
        start_frame=start_frame, 
        end_frame=end_frame,
        return_mlx=True, 
        verbose=False  # Moins de verbosité
    )
    
    if flows_data is None:
        raise ValueError("Impossible de charger les flows")
    
    # Chargement des labels - ajuster pour inclure end_frame
    gt_pixels_all = read_ground_truth_pixels(video_idx)
    labels_data = mx.array(gt_pixels_all[start_frame:end_frame+1], dtype=mx.float32)  # Frames 250-299 inclus pour correspondre aux flows
    mx.eval(labels_data)
    
    # Vérification des dimensions
    print(f"Dimensions - Flows: {flows_data.shape}, Labels: {labels_data.shape}")
    
    # Filtrage avec mean_threshold=13
    filter_config = {
        'filtering': {
            'norm': {
                'is_used': True,
                'min_threshold': 13.0
            },
            'colinearity': {
                'is_used': False,
                'min_threshold': 0.0
            }
        },
        'weighting': {
            'norm': {
                'is_used': False,
                'type': 'constant'
            },
            'colinearity': {
                'is_used': False,
                'type': 'constant'
            }
        }
    }
    
    flow_filter = FlowFilterBatch(filter_config)
    filtered_flows = flow_filter.filter(flows_data, chunk_size=50)
    mx.eval(filtered_flows)
    
    print(f"✅ Données préparées: {filtered_flows.shape[0]} échantillons")
    
    return filtered_flows, labels_data


def test_adam_with_plateau(flows_data, labels_data):
    """
    Test Adam AVEC détection de plateau.
    
    Returns:
        dict: Résultats du test
    """
    print("\n🔄 TEST 1: Adam AVEC détection de plateau")
    print("-" * 50)
    
    # Optimiseur avec plateau
    adam_plateau = AdamOptimizer(
        lr=10.0,
        beta1=0.6,
        beta2=0.98,
        eps=1e-8,
        max_iter=50,
        plateau_threshold=1e-4,  # AVEC détection
        plateau_patience=3
    )
    
    # Optimisation
    start_time = time.time()
    predictions_plateau = adam_plateau.optimize_batch(flows_data)
    mx.eval(predictions_plateau)
    optimization_time = time.time() - start_time
    
    # Calcul des métriques
    distances = mx.sqrt(mx.sum(mx.square(predictions_plateau - labels_data), axis=1))
    mx.eval(distances)
    distances_np = np.array(distances)
    
    results = {
        'name': 'Adam AVEC plateau',
        'predictions': np.array(predictions_plateau),
        'distances': distances_np,
        'optimization_time': optimization_time,
        'mean_distance': float(mx.mean(distances)),
        'median_distance': float(np.median(distances_np)),
        'std_distance': float(mx.std(distances)),
        'min_distance': float(mx.min(distances)),
        'max_distance': float(mx.max(distances)),
        'config': {
            'plateau_threshold': 1e-4,
            'plateau_patience': 3,
            'max_iter': 50
        }
    }
    
    print(f"✅ Temps d'optimisation: {optimization_time:.2f}s")
    print(f"✅ Distance moyenne: {results['mean_distance']:.2f} pixels")
    print(f"✅ Distance médiane: {results['median_distance']:.2f} pixels")
    
    return results


def test_adam_with_pixel_patience(flows_data, labels_data):
    """
    Test Adam AVEC détection de pixel patience.
    
    Returns:
        dict: Résultats du test
    """
    print("\n🔄 TEST 2: Adam AVEC détection de pixel patience")
    print("-" * 50)
    
    # Optimiseur avec pixel patience (pixel_patience=5 active la détection)
    adam_pixel_patience = AdamOptimizer(
        lr=10.0,
        beta1=0.6,
        beta2=0.98,
        eps=1e-8,
        max_iter=50,
        plateau_threshold=0.0,  # Désactivé car on utilise pixel_patience
        plateau_patience=3,
        pixel_patience=5  # AVEC détection de pixel patience
    )
    
    # Optimisation
    start_time = time.time()
    predictions_pixel_patience = adam_pixel_patience.optimize_batch(flows_data)
    mx.eval(predictions_pixel_patience)
    optimization_time = time.time() - start_time
    
    # Calcul des métriques
    distances = mx.sqrt(mx.sum(mx.square(predictions_pixel_patience - labels_data), axis=1))
    mx.eval(distances)
    distances_np = np.array(distances)
    
    results = {
        'name': 'Adam AVEC pixel patience',
        'predictions': np.array(predictions_pixel_patience),
        'distances': distances_np,
        'optimization_time': optimization_time,
        'mean_distance': float(mx.mean(distances)),
        'median_distance': float(np.median(distances_np)),
        'std_distance': float(mx.std(distances)),
        'min_distance': float(mx.min(distances)),
        'max_distance': float(mx.max(distances)),
        'config': {
            'plateau_threshold': 0.0,
            'plateau_patience': 3,
            'pixel_patience': 5,
            'max_iter': 50
        }
    }
    
    print(f"✅ Temps d'optimisation: {optimization_time:.2f}s")
    print(f"✅ Distance moyenne: {results['mean_distance']:.2f} pixels")
    print(f"✅ Distance médiane: {results['median_distance']:.2f} pixels")
    
    return results


def test_lbfgs_optimizer(flows_data, labels_data):
    """
    Test L-BFGS-B optimizer.
    
    Returns:
        dict: Résultats du test
    """
    print("\n🔄 TEST 3: L-BFGS-B optimizer")
    print("-" * 50)
    
    # Optimiseur L-BFGS-B
    lbfgs_optimizer = LBFGSOptimizer(
        max_iter=100,
        display_warnings=False
    )
    
    # Conversion en numpy pour L-BFGS-B (qui utilise scipy)
    flows_np = np.array(flows_data)
    labels_np = np.array(labels_data)
    
    # Optimisation
    start_time = time.time()
    predictions_lbfgs = []
    
    for i in range(flows_np.shape[0]):
        single_flow = flows_np[i]
        prediction = lbfgs_optimizer.optimize_single(single_flow)
        predictions_lbfgs.append(prediction)
    
    predictions_lbfgs = np.array(predictions_lbfgs)
    optimization_time = time.time() - start_time
    
    # Calcul des métriques
    distances = np.sqrt(np.sum((predictions_lbfgs - labels_np)**2, axis=1))
    
    results = {
        'name': 'L-BFGS-B',
        'predictions': predictions_lbfgs,
        'distances': distances,
        'optimization_time': optimization_time,
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'std_distance': float(np.std(distances)),
        'min_distance': float(np.min(distances)),
        'max_distance': float(np.max(distances)),
        'config': {
            'max_iter': 100,
            'method': 'L-BFGS-B'
        }
    }
    
    print(f"✅ Temps d'optimisation: {optimization_time:.2f}s")
    print(f"✅ Distance moyenne: {results['mean_distance']:.2f} pixels")
    print(f"✅ Distance médiane: {results['median_distance']:.2f} pixels")
    
    return results


def compare_results(results_lbfgs, results_plateau, results_pixel):
    """
    Compare et affiche les résultats des trois tests.
    
    Args:
        results_lbfgs: Résultats L-BFGS-B
        results_plateau: Résultats Adam avec détection de plateau
        results_pixel: Résultats Adam avec détection de pixel patience
    """
    print("\n" + "="*90)
    print("COMPARAISON DÉTAILLÉE DES TROIS OPTIMISEURS")
    print("="*90)
    
    # Table de comparaison étendue
    print(f"{'Métrique':<25} {'L-BFGS-B':<18} {'Adam+plateau':<18} {'Adam+pixel':<18} {'Meilleur':<15}")
    print("-" * 95)
    
    all_results = [results_lbfgs, results_plateau, results_pixel]
    names = ['L-BFGS-B', 'Adam+plateau', 'Adam+pixel']
    
    # Temps d'optimisation (plus bas = mieux)
    times = [r['optimization_time'] for r in all_results]
    best_time_idx = np.argmin(times)
    print(f"{'Temps (s)':<25} {times[0]:<18.2f} {times[1]:<18.2f} {times[2]:<18.2f} {names[best_time_idx]:<15}")
    
    # Distance moyenne (plus bas = mieux)
    means = [r['mean_distance'] for r in all_results]
    best_mean_idx = np.argmin(means)
    print(f"{'Distance moyenne (px)':<25} {means[0]:<18.2f} {means[1]:<18.2f} {means[2]:<18.2f} {names[best_mean_idx]:<15}")
    
    # Distance médiane (plus bas = mieux)
    medians = [r['median_distance'] for r in all_results]
    best_median_idx = np.argmin(medians)
    print(f"{'Distance médiane (px)':<25} {medians[0]:<18.2f} {medians[1]:<18.2f} {medians[2]:<18.2f} {names[best_median_idx]:<15}")
    
    # Distance minimum (plus bas = mieux)
    mins = [r['min_distance'] for r in all_results]
    best_min_idx = np.argmin(mins)
    print(f"{'Distance min (px)':<25} {mins[0]:<18.2f} {mins[1]:<18.2f} {mins[2]:<18.2f} {names[best_min_idx]:<15}")
    
    # Distance maximum (plus bas = mieux)
    maxs = [r['max_distance'] for r in all_results]
    best_max_idx = np.argmin(maxs)
    print(f"{'Distance max (px)':<25} {maxs[0]:<18.2f} {maxs[1]:<18.2f} {maxs[2]:<18.2f} {names[best_max_idx]:<15}")
    
    # Écart-type (plus bas = mieux)
    stds = [r['std_distance'] for r in all_results]
    best_std_idx = np.argmin(stds)
    print(f"{'Écart-type (px)':<25} {stds[0]:<18.2f} {stds[1]:<18.2f} {stds[2]:<18.2f} {names[best_std_idx]:<15}")
    
    # Performance (frames/seconde) (plus haut = mieux)
    fps = [50 / t for t in times]
    best_fps_idx = np.argmax(fps)
    print(f"{'Performance (fps)':<25} {fps[0]:<18.1f} {fps[1]:<18.1f} {fps[2]:<18.1f} {names[best_fps_idx]:<15}")
    
    print()
    
    # Comptage des victoires
    print("🏆 COMPTAGE DES VICTOIRES")
    print("-" * 30)
    
    victories = [0, 0, 0]
    metrics = ['temps', 'moyenne', 'médiane', 'min', 'max', 'std', 'fps']
    best_indices = [best_time_idx, best_mean_idx, best_median_idx, best_min_idx, best_max_idx, best_std_idx, best_fps_idx]
    
    for i, best_idx in enumerate(best_indices):
        victories[best_idx] += 1
        print(f"   {metrics[i]:<12}: {names[best_idx]}")
    
    print(f"\n📊 SCORE FINAL:")
    for i, name in enumerate(names):
        print(f"   {name:<15}: {victories[i]}/7 victoires")
    
    # Déterminer le gagnant global
    winner_idx = np.argmax(victories)
    print(f"\n🥇 GAGNANT GLOBAL: {names[winner_idx]}")
    
    # Analyse détaillée
    print(f"\n📈 ANALYSE DÉTAILLÉE")
    print("-" * 25)
    
    # Comparaison Adam plateau vs pixel patience
    time_improvement = (times[1] - times[2]) / times[1] * 100
    mean_improvement = (means[1] - means[2]) / means[1] * 100
    
    print(f"Adam plateau patience vs Adam pixel patience:")
    print(f"   ⏱️  Temps: {time_improvement:+.1f}% ({'✅ pixel patience plus rapide' if time_improvement > 0 else '❌ plateau patience plus rapide'})")
    print(f"   🎯 Précision: {mean_improvement:+.1f}% ({'✅ pixel patience plus précis' if mean_improvement > 0 else '❌ plateau patience plus précis'})")
    
    # Comparaison avec L-BFGS-B
    best_adam_idx = 1 if means[1] < means[2] else 2
    best_adam_name = names[best_adam_idx]
    precision_diff = (means[0] - means[best_adam_idx]) / means[0] * 100
    time_diff = (times[0] - times[best_adam_idx]) / times[0] * 100
    
    print(f"\nL-BFGS-B vs {best_adam_name}:")
    print(f"   ⏱️  Temps: {time_diff:+.1f}% ({'✅ L-BFGS-B plus rapide' if time_diff > 0 else '❌ Adam plus rapide'})")
    print(f"   🎯 Précision: {precision_diff:+.1f}% ({'✅ L-BFGS-B plus précis' if precision_diff > 0 else '❌ Adam plus précis'})")
    
    # Recommandation finale
    print(f"\n🎯 RECOMMANDATIONS:")
    if winner_idx == 0:
        print("   ✅ UTILISER L-BFGS-B (précision maximale)")
    elif winner_idx == 1:
        print("   ✅ UTILISER Adam avec détection de plateau (meilleur équilibre)")
    else:
        print("   ✅ UTILISER Adam avec détection de pixel patience (convergence rapide)")
        
    print(f"   📋 Pour la vitesse: {names[best_time_idx]}")
    print(f"   📋 Pour la précision: {names[best_mean_idx]}")
    print(f"   📋 Pour la stabilité: {names[best_std_idx]}")


def create_legends():
    """Crée les légendes pour les couleurs et les symboles."""
    color_legend = [
        Line2D([0], [0], color='black', label='Ground Truth'),
        Line2D([0], [0], color='green', label='L-BFGS-B'),
        Line2D([0], [0], color='blue', label='Adam+plateau'),
        Line2D([0], [0], color='red', label='Adam+pixel'),
        Line2D([0], [0], color='gray', label='Centre frame')
    ]

    symbol_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Moyenne'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Médiane'),
        Line2D([0], [0], marker='+', color='gray', markersize=12, label='Centre frame'),
        Line2D([0], [0], linestyle=':', color='gray', label='Écart-type (ellipse)')
    ]
    
    return color_legend + symbol_legend


def plot_distribution_on_axis(ax, pixels, color, label, alpha_points=0.3, show_frame_center=False, frame_size=None):
    """
    Trace la distribution des points sur un axe donné.
    
    Args:
        ax: Axe matplotlib
        pixels: Points en pixels (N, 2)
        color: Couleur pour cette distribution
        label: Label pour cette distribution
        alpha_points: Transparence des points individuels
        show_frame_center: Si True, affiche le centre de la frame
        frame_size: Tuple (width, height) de la frame pour calculer le centre
    """
    # Calcul des statistiques
    mean_point = np.mean(pixels, axis=0)
    median_point = np.median(pixels, axis=0)
    std_point = np.std(pixels, axis=0)
    
    # Points individuels avec transparence
    ax.scatter(pixels[:, 0], pixels[:, 1], c=color, alpha=alpha_points, s=20, label=f'{label} (points)')
    
    # Moyenne (cercle plein)
    ax.plot(mean_point[0], mean_point[1], 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=1)
    
    # Médiane (triangle)
    ax.plot(median_point[0], median_point[1], '^', color=color, markersize=12, markeredgecolor='white', markeredgewidth=1)
    
    # Écart-type (ellipse)
    ellipse = Ellipse((mean_point[0], mean_point[1]), 
                     2*std_point[0], 2*std_point[1],
                     fill=False, color=color, linestyle=':', linewidth=2, alpha=0.8)
    ax.add_patch(ellipse)
    
    # Centre de la frame (si demandé)
    if show_frame_center and frame_size is not None:
        center_x, center_y = frame_size[0] // 2, frame_size[1] // 2
        ax.plot(center_x, center_y, '+', color='gray', markersize=15, markeredgewidth=3, 
                label='Centre frame')
        # Lignes de référence (optionnel)
        ax.axvline(x=center_x, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=center_y, color='gray', linestyle='--', alpha=0.3)
    
    return mean_point, median_point, std_point


def visualize_comparison(results_lbfgs, results_plateau, results_pixel, labels_data):
    """
    Visualise la comparaison des résultats avec distribution des points.
    
    Args:
        results_lbfgs: Résultats L-BFGS-B
        results_plateau: Résultats Adam avec détection de plateau
        results_pixel: Résultats Adam avec détection de pixel patience
        labels_data: Labels ground truth
    """
    print("\n🔄 Génération des visualisations comparatives...")
    
    # Préparer les données
    labels_np = np.array(labels_data)
    predictions_lbfgs = results_lbfgs['predictions']
    predictions_with = results_plateau['predictions']
    predictions_pixel = results_pixel['predictions']
    
    distances_lbfgs = results_lbfgs['distances']
    distances_with = results_plateau['distances']
    distances_pixel = results_pixel['distances']
    
    # Calculer la taille de la frame à partir des données de flows (disponible depuis load_and_filter_data)
    # On utilise la taille standard des flows qu'on avait : (50, 874, 1164, 2)
    frame_height, frame_width = 874, 1164
    frame_size = (frame_width, frame_height)
    
    # Créer la figure avec 2x2 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparaison des distributions de points - Ground Truth vs 3 Optimiseurs (50 frames)', fontsize=16)
    
    # Couleurs pour chaque méthode
    colors = ['black', 'green', 'blue', 'red']
    names = ['Ground Truth', 'L-BFGS-B', 'Adam+plateau', 'Adam+pixel']
    all_predictions = [labels_np, predictions_lbfgs, predictions_with, predictions_pixel]
    
    # Utiliser les limites de la frame (0 à frame_width/frame_height)
    x_min, x_max = 0, frame_width
    y_min, y_max = 0, frame_height
    
    # 1. Graphique combiné avec toutes les distributions
    ax_combined = axes[0, 0]
    ax_combined.set_title('Toutes les distributions (superposées)', fontsize=14)
    
    stats_summary = []
    for i, (pixels, color, name) in enumerate(zip(all_predictions, colors, names)):
        alpha = 0.7 if i == 0 else 0.4  # Ground truth plus visible
        show_center = (i == 0)  # Montrer le centre seulement pour le premier (Ground Truth)
        mean_pt, median_pt, std_pt = plot_distribution_on_axis(ax_combined, pixels, color, name, alpha, show_center, frame_size)
        stats_summary.append({
            'name': name,
            'mean': mean_pt,
            'median': median_pt,
            'std': std_pt
        })
    
    ax_combined.set_xlim(x_min, x_max)
    ax_combined.set_ylim(y_max, y_min)  # Inverser Y : plus petit en haut, plus grand en bas
    ax_combined.set_xlabel('X (pixels)')
    ax_combined.set_ylabel('Y (pixels)')
    ax_combined.grid(True, alpha=0.3)
    ax_combined.legend(handles=create_legends(), fontsize=8, loc='upper right')
    
    # 2. Comparaison des distances (histogrammes)
    ax_distances = axes[0, 1]
    ax_distances.set_title('Distribution des distances aux labels', fontsize=14)
    
    distances_all = [distances_lbfgs, distances_with, distances_pixel]
    optimizer_names = ['L-BFGS-B', 'Adam+plateau', 'Adam+pixel']
    optimizer_colors = ['green', 'blue', 'red']
    
    for i, (distances, name, color) in enumerate(zip(distances_all, optimizer_names, optimizer_colors)):
        ax_distances.hist(distances, bins=15, alpha=0.6, color=color, label=f'{name} (μ={np.mean(distances):.1f}px)', edgecolor='black')
    
    ax_distances.set_xlabel('Distance euclidienne (pixels)')
    ax_distances.set_ylabel('Fréquence')
    ax_distances.legend()
    ax_distances.grid(True, alpha=0.3)
    
    # 3. Comparaison Adam+plateau vs Ground Truth
    ax_adam_plateau = axes[1, 0]
    ax_adam_plateau.set_title('Adam+plateau vs Ground Truth', fontsize=14)
    
    plot_distribution_on_axis(ax_adam_plateau, labels_np, 'black', 'Ground Truth', 0.5, True, frame_size)
    plot_distribution_on_axis(ax_adam_plateau, predictions_with, 'blue', 'Adam+plateau', 0.5)
    
    ax_adam_plateau.set_xlim(x_min, x_max)
    ax_adam_plateau.set_ylim(y_max, y_min)  # Inverser Y : plus petit en haut, plus grand en bas
    ax_adam_plateau.set_xlabel('X (pixels)')
    ax_adam_plateau.set_ylabel('Y (pixels)')
    ax_adam_plateau.grid(True, alpha=0.3)
    
    # 4. Comparaison Adam+pixel vs Ground Truth
    ax_adam_pixel = axes[1, 1]
    ax_adam_pixel.set_title('Adam+pixel vs Ground Truth', fontsize=14)
    
    plot_distribution_on_axis(ax_adam_pixel, labels_np, 'black', 'Ground Truth', 0.5, True, frame_size)
    plot_distribution_on_axis(ax_adam_pixel, predictions_pixel, 'red', 'Adam+pixel', 0.5)
    
    ax_adam_pixel.set_xlim(x_min, x_max)
    ax_adam_pixel.set_ylim(y_max, y_min)  # Inverser Y : plus petit en haut, plus grand en bas
    ax_adam_pixel.set_xlabel('X (pixels)')
    ax_adam_pixel.set_ylabel('Y (pixels)')
    ax_adam_pixel.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher un résumé textuel des statistiques
    print("\n📊 RÉSUMÉ DES DISTRIBUTIONS")
    print("="*60)
    print(f"{'Méthode':<15} {'Moyenne X':<10} {'Moyenne Y':<10} {'Std X':<8} {'Std Y':<8}")
    print("-" * 60)
    
    # Afficher le centre de la frame en premier
    center_x, center_y = frame_size[0] // 2, frame_size[1] // 2
    print(f"{'Centre frame':<15} {center_x:<10.1f} {center_y:<10.1f} {'--':<8} {'--':<8}")
    print("-" * 60)
    
    for stat in stats_summary:
        print(f"{stat['name']:<15} {stat['mean'][0]:<10.1f} {stat['mean'][1]:<10.1f} "
              f"{stat['std'][0]:<8.1f} {stat['std'][1]:<8.1f}")
    
    print("✅ Visualisations des distributions générées")


def main(video_idx=4, start_frame=250, end_frame=299):
    """Fonction principale de la visualisation comparative des optimiseurs."""
    print("="*90)
    print("VISUALISATION DES DISTRIBUTIONS DES OPTIMISEURS")
    print("="*90)
    print("Configuration:")
    print(f"- Dataset: Vidéo {video_idx}, frames {start_frame}-{end_frame} ({end_frame-start_frame+1} échantillons)")
    print("- Filtrage: mean_threshold=13")
    print("- Test 1: L-BFGS-B (paramètres par défaut)")
    print("- Test 2: Adam avec plateau patience (lr=10.0, plateau_threshold=1e-4, plateau_patience=3)")
    print("- Test 3: Adam avec pixel patience (lr=10.0, pixel_patience=5)")
    print("="*90)
    
    try:
        # Chargement et filtrage des données (une seule fois)
        filtered_flows, labels_data = load_and_filter_data(video_idx, start_frame, end_frame)
        
        # Test 1: L-BFGS-B
        results_lbfgs = test_lbfgs_optimizer(filtered_flows, labels_data)
        
        # Test 2: Adam avec détection de plateau
        results_with_plateau = test_adam_with_plateau(filtered_flows, labels_data)
        
        # Test 3: Adam avec détection de pixel patience
        results_with_pixel = test_adam_with_pixel_patience(filtered_flows, labels_data)
        
        # Comparaison détaillée des trois optimiseurs
        compare_results(results_lbfgs, results_with_plateau, results_with_pixel)
        
        # Visualisations comparatives
        visualize_comparison(results_lbfgs, results_with_plateau, results_with_pixel, labels_data)
        
        print("\n" + "="*90)
        print("VISUALISATION TERMINÉE")
        print("="*90)
        
    except Exception as e:
        print(f"❌ Erreur lors de la visualisation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    video_idx = 4
    start_frame = 250
    end_frame = 299
    main(video_idx, start_frame, end_frame) 