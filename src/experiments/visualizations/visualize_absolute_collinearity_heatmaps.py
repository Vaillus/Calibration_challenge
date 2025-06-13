#!/usr/bin/env python3
"""
Génération de heatmaps de collinéarité par vidéo.

Ce script calcule pour chaque vidéo une heatmap globale des scores de collinéarité.
Pour chaque pixel, on calcule la moyenne des scores de collinéarité sur toutes les frames,
en ignorant les frames où le pixel a été filtré (flow nul après filtrage avec seuil min=13).

Usage:
    python create_collinearity_heatmaps.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import mlx.core as mx
import gc

from src.utilities.load_flows import load_flows
from src.utilities.paths import get_outputs_dir, get_data_dir
from src.core.flow_filter import FlowFilterBatch
from src.core.collinearity_scorer_batch import BatchCollinearityScorer


def load_labels_for_video(video_id: int) -> Optional[np.ndarray]:
    """
    Charge tous les labels (points de fuite) pour une vidéo spécifique.
    Un label par frame.
    
    Args:
        video_id: ID de la vidéo
        
    Returns:
        Array numpy (n_frames, 2) des labels en pixels, ou None si erreur
    """
    try:
        print(f"   📍 Chargement des labels pour vidéo {video_id}...")
        
        # Utiliser la fonction existante qui convertit déjà en pixels
        from src.utilities.ground_truth import read_ground_truth_pixels
        labels_pixels = read_ground_truth_pixels(video_id)
        
        if len(labels_pixels) == 0:
            print(f"      ⚠️  Aucun label trouvé")
            return None
            
        # Ignorer le premier label (index 0) et prendre les suivants
        labels_to_use = labels_pixels[1:] if len(labels_pixels) > 1 else labels_pixels
        
        labels_array = np.array(labels_to_use, dtype=np.float32)
        print(f"      ✅ {len(labels_pixels)} labels totaux chargés, {len(labels_to_use)} utilisés (premier ignoré)")
        print(f"      📊 Premier label utilisé: ({labels_to_use[0][0]:.1f}, {labels_to_use[0][1]:.1f}) pixels")
        print(f"      📊 Dernier label utilisé: ({labels_to_use[-1][0]:.1f}, {labels_to_use[-1][1]:.1f}) pixels")
        
        return labels_array
        
    except Exception as e:
        print(f"      ❌ Erreur lors du chargement des labels: {e}")
        return None


def create_collinearity_heatmap_for_video(
    video_id: int, 
    apply_filter: bool = True,
    min_threshold: float = 13.0,
    chunk_size: int = 100
) -> Tuple[np.ndarray, Dict]:
    """
    Crée une heatmap de collinéarité globale pour une vidéo.
    Inspiré de l'approche chunking d'analyse_flow_vectors_count.py pour optimiser la mémoire.
    
    Args:
        video_id: ID de la vidéo à analyser
        apply_filter: Si True, applique le filtre de norme. Si False, utilise tous les flow vectors
        min_threshold: Seuil minimum de norme pour le filtrage (utilisé seulement si apply_filter=True)
        chunk_size: Taille des chunks pour éviter le manque de mémoire GPU
        
    Returns:
        Tuple (global_heatmap, stats_dict)
        - global_heatmap: Array (H, W) - moyenne des scores par pixel
        - stats_dict: Dictionnaire avec statistiques
    """
    print(f"\n📹 Génération heatmap collinéarité pour vidéo {video_id}...")
    
    # 1. Charger les flows directement en MLX
    flows_mx = load_flows(video_id, use_compressed=False, verbose=True, return_mlx=True)
    
    if flows_mx is None:
        print(f"❌ Impossible de charger les flows pour la vidéo {video_id}")
        return None, None
    
    n_frames, H, W, _ = flows_mx.shape
    print(f"   📊 Dimensions: {n_frames} frames, {H}x{W} pixels")
    print(f"   🔄 Traitement par chunks de {chunk_size} frames")
    
    # Charger les labels pour cette vidéo (un par frame)
    labels_array = load_labels_for_video(video_id)
    if labels_array is None:
        print(f"   ⚠️  Impossible de charger les labels, utilisation du centre de l'image")
        use_labels = False
    elif len(labels_array) != n_frames:
        print(f"   ⚠️  Nombre de labels ({len(labels_array)}) != nombre de frames ({n_frames})")
        print(f"   ⚠️  Utilisation du centre de l'image")
        use_labels = False
    else:
        print(f"   ✅ Labels chargés et validés pour toutes les frames")
        use_labels = True
    
    # 2. Configuration du filtrage (optionnel)
    if apply_filter:
        filter_config = {
            'filtering': {
                'norm': {
                    'is_used': True,
                    'min_threshold': min_threshold
                }
            }
        }
        flow_filter = FlowFilterBatch(filter_config)
        print(f"   🔍 Filtrage ACTIVÉ avec seuil minimum: {min_threshold}")
    else:
        flow_filter = None
        print(f"   🔓 Filtrage DÉSACTIVÉ - utilisation de tous les flow vectors")
    
    # 3. Initialiser le scorer de collinéarité
    scorer = BatchCollinearityScorer()
    
    # 5. Initialiser les accumulateurs pour la heatmap globale
    sum_scores = np.zeros((H, W), dtype=np.float64)
    count_valid = np.zeros((H, W), dtype=np.int64)
    
    # 6. Traiter par chunks pour éviter les problèmes de mémoire GPU
    print(f"   🔢 Calcul des collinearity maps par chunks...")
    
    for start_idx in range(0, n_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, n_frames)
        chunk_frames = end_idx - start_idx
        
        print(f"      📦 Chunk {start_idx}-{end_idx-1} ({chunk_frames} frames)")
        
        # Extraire le chunk de flows
        flows_chunk = flows_mx[start_idx:end_idx]
        
        # Filtrer le chunk (si filtrage activé)
        if flow_filter is not None:
            filtered_flows_chunk = flow_filter.filter(flows_chunk)
            mx.eval(filtered_flows_chunk)
        else:
            filtered_flows_chunk = flows_chunk
        
        # Créer les points de référence pour le chunk
        if use_labels:
            # Utiliser les labels spécifiques à ce chunk
            labels_chunk = labels_array[start_idx:end_idx]  # Shape: (chunk_frames, 2)
            
            # Vérifier s'il y a des labels NaN (convertis en centre d'image par angles_to_pixels)
            center_point = (W // 2, H // 2)
            nan_mask = []
            
            for i, (x, y) in enumerate(labels_chunk):
                # Si le label est exactement au centre, c'était probablement un NaN
                is_center = (abs(x - center_point[0]) < 1) and (abs(y - center_point[1]) < 1)
                nan_mask.append(is_center)
                if is_center:
                    print(f"         ⚠️  Frame {start_idx + i}: Label était NaN, ignoré pour collinéarité")
            
            reference_points_batch = mx.array(labels_chunk, dtype=mx.float32)
        else:
            # Utiliser le centre de l'image pour toutes les frames
            center_point = (W // 2, H // 2)
            reference_points_batch = mx.array([center_point] * chunk_frames, dtype=mx.float32)
            nan_mask = [False] * chunk_frames  # Aucun NaN quand on utilise le centre explicitement
        
        # Calculer les collinearity maps pour ce chunk
        collinearity_maps_chunk = scorer.compute_colinearity_map_batch(filtered_flows_chunk, reference_points_batch)
        mx.eval(collinearity_maps_chunk)
        
        # Convertir en numpy pour accumulation
        collinearity_maps_np = np.array(collinearity_maps_chunk)
        
        # Accumuler dans les totaux globaux
        # Masque pour les pixels valides (scores non nuls)
        valid_mask_chunk = collinearity_maps_np != 0  # (chunk_frames, H, W)
        
        # Exclure les frames avec des labels NaN de l'accumulation
        if use_labels:
            for frame_idx, is_nan in enumerate(nan_mask):
                if is_nan:
                    # Mettre à zéro les scores et masques pour cette frame
                    collinearity_maps_np[frame_idx] = 0
                    valid_mask_chunk[frame_idx] = False
        
        # Ajouter aux accumulateurs
        sum_scores += np.sum(collinearity_maps_np, axis=0)  # (H, W)
        count_valid += np.sum(valid_mask_chunk, axis=0)  # (H, W)
        
        # Nettoyage mémoire du chunk 
        del flows_chunk, filtered_flows_chunk, reference_points_batch
        del collinearity_maps_chunk, collinearity_maps_np, valid_mask_chunk
        gc.collect()
    
    # Nettoyer flows_mx maintenant qu'on n'en a plus besoin
    del flows_mx
    gc.collect()
    
    # 7. Calculer la heatmap globale finale
    print(f"   📊 Calcul de la heatmap globale finale...")
    
    # Calculer la moyenne en évitant division par zéro
    global_heatmap = np.where(count_valid > 0, sum_scores / count_valid, 0)
    
    # 8. Calculer des statistiques
    total_pixels = H * W
    pixels_with_data = np.sum(count_valid > 0)
    mean_frames_per_pixel = np.mean(count_valid[count_valid > 0]) if pixels_with_data > 0 else 0
    
    valid_scores = global_heatmap[global_heatmap != 0]
    
    stats = {
        'video_id': video_id,
        'total_frames': n_frames,
        'image_size': (H, W),
        'uses_labels': use_labels,
        'apply_filter': apply_filter,
        'min_threshold': min_threshold if apply_filter else None,
        'total_pixels': total_pixels,
        'pixels_with_data': int(pixels_with_data),
        'pixels_coverage': float(pixels_with_data / total_pixels),
        'mean_frames_per_pixel': float(mean_frames_per_pixel),
        'global_scores': {
            'mean': float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0,
            'median': float(np.median(valid_scores)) if len(valid_scores) > 0 else 0,
            'std': float(np.std(valid_scores)) if len(valid_scores) > 0 else 0,
            'min': float(np.min(valid_scores)) if len(valid_scores) > 0 else 0,
            'max': float(np.max(valid_scores)) if len(valid_scores) > 0 else 0,
        }
    }
    
    print(f"   📊 Statistiques:")
    print(f"      - Pixels avec données: {pixels_with_data}/{total_pixels} ({stats['pixels_coverage']:.1%})")
    print(f"      - Frames moyennes par pixel: {mean_frames_per_pixel:.1f}")
    print(f"      - Score collinéarité moyen: {stats['global_scores']['mean']:.3f}")
    print(f"      - Score collinéarité médian: {stats['global_scores']['median']:.3f}")
    
    return global_heatmap, stats


def display_heatmap_visualization(
    global_heatmap: np.ndarray,
    stats: Dict,
    colormap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Affiche la visualisation de la heatmap sans sauvegarder.
    
    Args:
        global_heatmap: Heatmap globale (H, W)
        stats: Dictionnaire des statistiques
        colormap: Colormap à utiliser
        figsize: Taille de la figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Masquer les pixels sans données (score = 0)
    masked_heatmap = np.ma.masked_where(global_heatmap == 0, global_heatmap)
    
    # Créer la heatmap
    im = ax.imshow(masked_heatmap, cmap=colormap, aspect='equal')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score de Collinéarité Moyen', rotation=270, labelpad=20)
    
    # Titre et labels
    video_id = stats['video_id']
    coverage = stats['pixels_coverage']
    mean_score = stats['global_scores']['mean']
    filter_status = "Avec filtre" if stats['apply_filter'] else "Sans filtre"
    threshold_text = f" (seuil: {stats['min_threshold']})" if stats['apply_filter'] else ""
    ref_status = "Labels/frame" if stats['uses_labels'] else "Centre image"
    
    ax.set_title(f'Heatmap Collinéarité - Vidéo {video_id} - {filter_status}{threshold_text}\n'
                f'Référence: {ref_status} | Couverture: {coverage:.1%} | Score moyen: {mean_score:.3f}')
    ax.set_xlabel('Largeur (pixels)')
    ax.set_ylabel('Hauteur (pixels)')
    
    # Ajouter des statistiques en texte
    stats_text = (f"Frames: {stats['total_frames']}\n"
                 f"Filtrage: {'Oui' if stats['apply_filter'] else 'Non'}\n"
                 f"Ref: {'Labels' if stats['uses_labels'] else 'Centre'}\n"
                 f"Médiane: {stats['global_scores']['median']:.3f}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def display_comparison_plot(
    all_heatmaps: Dict[int, np.ndarray],
    all_stats: Dict[int, Dict],
    colormap: str = 'viridis',
    figsize: Tuple[int, int] = (20, 12)
) -> None:
    """
    Affiche un plot de comparaison entre toutes les vidéos.
    
    Args:
        all_heatmaps: Dictionnaire {video_id: heatmap}
        all_stats: Dictionnaire {video_id: stats}
        colormap: Colormap à utiliser
        figsize: Taille de la figure
    """
    n_videos = len(all_heatmaps)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Calculer les limites communes pour la colorbar
    all_valid_scores = []
    for heatmap in all_heatmaps.values():
        valid_scores = heatmap[heatmap != 0]
        if len(valid_scores) > 0:
            all_valid_scores.extend(valid_scores)
    
    if len(all_valid_scores) > 0:
        vmin, vmax = np.min(all_valid_scores), np.max(all_valid_scores)
    else:
        vmin, vmax = 0, 1
    
    for i, video_id in enumerate(sorted(all_heatmaps.keys())):
        ax = axes[i]
        heatmap = all_heatmaps[video_id]
        stats = all_stats[video_id]
        
        # Masquer les pixels sans données
        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
        
        # Afficher la heatmap
        im = ax.imshow(masked_heatmap, cmap=colormap, vmin=vmin, vmax=vmax, aspect='equal')
        
        # Titre avec statistiques
        coverage = stats['pixels_coverage']
        mean_score = stats['global_scores']['mean']
        filter_text = "F" if stats['apply_filter'] else "NF"
        ref_text = "L" if stats['uses_labels'] else "C"  # L=Label, C=Centre
        ax.set_title(f'Vidéo {video_id} ({filter_text}, {ref_text})\nCouv: {coverage:.1%} | Moy: {mean_score:.3f}')
        ax.set_xlabel('Largeur')
        ax.set_ylabel('Hauteur')
    
    # Masquer le dernier subplot s'il n'est pas utilisé
    if n_videos < 6:
        axes[-1].set_visible(False)
    
    # Colorbar commune
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Score de Collinéarité Moyen', rotation=270, labelpad=20)
    
    # Titre avec info sur le filtrage et les références
    first_stats = next(iter(all_stats.values()))
    filter_status = "Avec filtre" if first_stats['apply_filter'] else "Sans filtre"
    threshold_text = f" (seuil: {first_stats['min_threshold']})" if first_stats['apply_filter'] else ""
    ref_status = "Labels par frame" if first_stats['uses_labels'] else "Centre image"
    
    plt.suptitle(f'Comparaison Heatmaps Collinéarité - {filter_status}{threshold_text} - Ref: {ref_status}', fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale du script."""
    print("🚀 GÉNÉRATION DES HEATMAPS DE COLLINÉARITÉ")
    print("=" * 60)
    
    # Demander à l'utilisateur s'il veut appliquer le filtre
    print("\n❓ Choix du mode de traitement:")
    print("   1. Avec filtre de norme (seuil minimum: 13.0)")
    print("   2. Sans filtre (tous les flow vectors)")
    
    while True:
        choice = input("\nVotre choix (1 ou 2): ").strip()
        if choice == "1":
            apply_filter = True
            min_threshold = 13.0
            break
        elif choice == "2":
            apply_filter = False
            min_threshold = None
            break
        else:
            print("❌ Choix invalide. Veuillez entrer 1 ou 2.")
    
    # Configuration - Focus sur vidéo 0 uniquement
    video_ids = [0, 1, 2, 3, 4]  # Seulement vidéo 0 pour aller plus vite
    chunk_size = 300  # Chunks plus grands maintenant
    
    print(f"\n⚙️  Configuration:")
    print(f"   - Filtrage: {'Activé' if apply_filter else 'Désactivé'}")
    if apply_filter:
        print(f"   - Seuil minimum de norme: {min_threshold}")
    print(f"   - Vidéos à analyser: {video_ids}")
    print(f"   - Taille des chunks: {chunk_size}")
    
    # Traitement de toutes les vidéos
    all_heatmaps = {}
    all_stats = {}
    total_start_time = time.time()
    
    for video_id in video_ids:
        start_time = time.time()
        
        heatmap, stats = create_collinearity_heatmap_for_video(
            video_id, apply_filter, min_threshold, chunk_size=chunk_size
        )
        
        if heatmap is not None and stats is not None:
            all_heatmaps[video_id] = heatmap
            all_stats[video_id] = stats
            
        else:
            print(f"⚠️  Vidéo {video_id} ignorée due à une erreur")
        
        elapsed = time.time() - start_time
        print(f"   ⏱️  Temps pour vidéo {video_id}: {elapsed:.2f}s")
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️  Temps total: {total_time:.2f}s")
    
    if not all_heatmaps:
        print("❌ Aucune vidéo n'a pu être traitée!")
        return
    
    # Afficher un résumé global
    print(f"\n📋 RÉSUMÉ GLOBAL:")
    print("=" * 50)
    print(f"   - Vidéos traitées: {len(all_heatmaps)}")
    print(f"   - Mode: {'Avec filtre' if apply_filter else 'Sans filtre'}")
    if apply_filter:
        print(f"   - Seuil appliqué: {min_threshold}")
    
    # Statistiques détaillées
    for video_id, stats in all_stats.items():
        ref_type = "labels (par frame)" if stats['uses_labels'] else "centre image"
        print(f"\n   📹 Vidéo {video_id} (référence: {ref_type}):")
        print(f"     * Couverture: {stats['pixels_coverage']:.1%}")
        print(f"     * Score moyen: {stats['global_scores']['mean']:.3f}")
        print(f"     * Score médian: {stats['global_scores']['median']:.3f}")
        print(f"     * Score min/max: {stats['global_scores']['min']:.3f} / {stats['global_scores']['max']:.3f}")
        print(f"     * Frames moyennes par pixel: {stats['mean_frames_per_pixel']:.1f}")
    
    # Afficher la visualisation (comparaison ou individuelle selon le nombre de vidéos)
    if len(all_heatmaps) > 1:
        print(f"\n📊 Affichage du plot de comparaison...")
        display_comparison_plot(all_heatmaps, all_stats)
    else:
        print(f"\n🖼️  Affichage de la heatmap...")
        video_id = list(all_heatmaps.keys())[0]
        heatmap = all_heatmaps[video_id]
        stats = all_stats[video_id]
        display_heatmap_visualization(heatmap, stats)
    
    print(f"\n✅ Analyse terminée!")


if __name__ == "__main__":
    main() 