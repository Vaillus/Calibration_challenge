#!/usr/bin/env python3
"""
Visualisation de heatmaps de collinéarité.

Ce module contient toutes les fonctions pour visualiser les heatmaps de collinéarité,
que ce soit individuelles, comparatives, relatives ou absolues.

Usage:
    from src.experiments.visualizations.heatmap_visualizer import HeatmapVisualizer
    
    visualizer = HeatmapVisualizer()
    visualizer.display_individual_heatmap(heatmap, stats)
    visualizer.display_comparison_plot(all_heatmaps, all_stats)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

from src.utilities.heatmap_loader import (
    list_available_heatmaps, 
    load_global_heatmap, 
    load_individual_heatmap,
    load_mean_pixel_for_video
)


def convert_absolute_to_relative_heatmap(
    absolute_heatmap: np.ndarray,
    mean_pixel: Tuple[float, float],
    stats: Dict
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Convertit une heatmap absolue en heatmap relative.
    
    Transforme les coordonnées de (x,y) absolu vers (x-mean_x, y-mean_y) relatif,
    centrant ainsi la heatmap sur le point de fuite moyen de la vidéo.
    
    Args:
        absolute_heatmap: Heatmap absolue (H, W)
        mean_pixel: Coordonnées (x, y) de la moyenne des pixels
        stats: Statistiques de la heatmap absolue
        
    Returns:
        Tuple (relative_heatmap, relative_stats, coordinate_mapping)
    """
    H, W = absolute_heatmap.shape
    mean_x, mean_y = mean_pixel
    
    print(f"      🔄 Transformation géométrique vers espace relatif...")
    print(f"         Centre de référence: ({mean_x:.2f}, {mean_y:.2f})")
    
    # Créer les grilles de coordonnées absolues
    y_abs, x_abs = np.mgrid[0:H, 0:W]
    
    # Transformer vers coordonnées relatives
    x_rel = x_abs - mean_x  # Coordonnées relatives en x
    y_rel = y_abs - mean_y  # Coordonnées relatives en y
    
    # Calculer les limites de l'espace relatif
    x_min, x_max = np.min(x_rel), np.max(x_rel)
    y_min, y_max = np.min(y_rel), np.max(y_rel)
    
    print(f"         Espace relatif - X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}]")
    
    # Pour l'instant, on garde la même grille mais on stocke la transformation
    # La heatmap relative aura les mêmes valeurs mais dans un espace conceptuellement différent
    relative_heatmap = absolute_heatmap.copy()
    
    # Mettre à jour les stats pour refléter la nature relative
    relative_stats = stats.copy()
    relative_stats['mean_pixel'] = mean_pixel
    relative_stats['is_relative'] = True
    relative_stats['relative_space_bounds'] = {
        'x_min': float(x_min), 'x_max': float(x_max),
        'y_min': float(y_min), 'y_max': float(y_max)
    }
    
    # Renommer les clés de scores pour clarifier
    if 'global_scores' in relative_stats:
        relative_stats['relative_scores'] = relative_stats.pop('global_scores')
    
    # Ajouter la transformation de coordonnées
    coordinate_mapping = {
        'x_relative': x_rel,
        'y_relative': y_rel,
        'transformation': f"(x,y) -> (x-{mean_x:.2f}, y-{mean_y:.2f})"
    }
    
    return relative_heatmap, relative_stats, coordinate_mapping


def create_unified_relative_space(
    all_heatmaps: Dict[int, np.ndarray],
    all_coordinate_mappings: Dict[int, Dict],
    resolution_factor: float = 1.0
) -> Tuple[np.ndarray, Dict]:
    """
    Crée un espace relatif unifié en combinant toutes les heatmaps relatives.
    
    Args:
        all_heatmaps: Dictionnaire {video_id: heatmap_relative}
        all_coordinate_mappings: Dictionnaire {video_id: coordinate_mapping}
        resolution_factor: Facteur pour ajuster la résolution finale
        
    Returns:
        Tuple (unified_heatmap, unified_space_info)
    """
    print(f"\n🔗 Création de l'espace relatif unifié...")
    
    # 1. Calculer les limites globales de l'espace relatif
    all_x_mins, all_x_maxs = [], []
    all_y_mins, all_y_maxs = [], []
    
    for video_id, coord_map in all_coordinate_mappings.items():
        x_rel, y_rel = coord_map['x_relative'], coord_map['y_relative']
        all_x_mins.append(np.min(x_rel))
        all_x_maxs.append(np.max(x_rel))
        all_y_mins.append(np.min(y_rel))
        all_y_maxs.append(np.max(y_rel))
    
    global_x_min, global_x_max = min(all_x_mins), max(all_x_maxs)
    global_y_min, global_y_max = min(all_y_mins), max(all_y_maxs)
    
    print(f"   📏 Limites globales de l'espace relatif:")
    print(f"      X: [{global_x_min:.1f}, {global_x_max:.1f}]")
    print(f"      Y: [{global_y_min:.1f}, {global_y_max:.1f}]")
    
    # 2. Définir la grille unifiée
    # Utiliser une résolution similaire aux images originales mais ajustable
    range_x = global_x_max - global_x_min
    range_y = global_y_max - global_y_min
    
    # Estimation de la résolution basée sur les heatmaps originales
    first_heatmap = next(iter(all_heatmaps.values()))
    orig_h, orig_w = first_heatmap.shape
    
    # Calculer les nouvelles dimensions en gardant une résolution similaire
    new_w = int(range_x * resolution_factor)
    new_h = int(range_y * resolution_factor)
    
    # S'assurer d'avoir des dimensions minimales raisonnables
    new_w = max(new_w, 200)
    new_h = max(new_h, 200)
    
    print(f"   🗂️  Nouvelle grille unifiée: {new_h} x {new_w}")
    print(f"      Résolution: {range_x/new_w:.2f} pixels/unité en X, {range_y/new_h:.2f} pixels/unité en Y")
    
    # 3. Créer la grille unifiée
    unified_sum = np.zeros((new_h, new_w), dtype=np.float64)
    unified_count = np.zeros((new_h, new_w), dtype=np.int64)
    
    # Coordonnées de la grille unifiée
    x_unified = np.linspace(global_x_min, global_x_max, new_w)
    y_unified = np.linspace(global_y_min, global_y_max, new_h)
    
    # 4. Redistribuer chaque heatmap dans l'espace unifié
    for video_id, heatmap in all_heatmaps.items():
        print(f"   📹 Redistribution vidéo {video_id}...")
        
        coord_map = all_coordinate_mappings[video_id]
        x_rel, y_rel = coord_map['x_relative'], coord_map['y_relative']
        
        # Pour chaque pixel de la heatmap originale ayant une valeur non nulle
        valid_mask = heatmap != 0
        valid_indices = np.where(valid_mask)
        
        for i, j in zip(valid_indices[0], valid_indices[1]):
            # Coordonnées relatives de ce pixel
            x_rel_val = x_rel[i, j]
            y_rel_val = y_rel[i, j]
            
            # Trouver les indices correspondants dans la grille unifiée
            x_idx = np.argmin(np.abs(x_unified - x_rel_val))
            y_idx = np.argmin(np.abs(y_unified - y_rel_val))
            
            # Ajouter la valeur à la grille unifiée
            unified_sum[y_idx, x_idx] += heatmap[i, j]
            unified_count[y_idx, x_idx] += 1
        
        valid_pixels = np.sum(valid_mask)
        print(f"      ✅ {valid_pixels} pixels redistribués")
    
    # 5. Calculer la heatmap finale (moyenne)
    unified_heatmap = np.where(unified_count > 0, unified_sum / unified_count, 0)
    
    # 6. Statistiques de l'espace unifié
    pixels_with_data = np.sum(unified_count > 0)
    total_pixels = new_h * new_w
    
    unified_space_info = {
        'bounds': {
            'x_min': global_x_min, 'x_max': global_x_max,
            'y_min': global_y_min, 'y_max': global_y_max
        },
        'dimensions': (new_h, new_w),
        'coordinates': {'x': x_unified, 'y': y_unified},
        'resolution': {'x': range_x/new_w, 'y': range_y/new_h},
        'coverage': float(pixels_with_data / total_pixels),
        'pixels_with_data': int(pixels_with_data),
        'max_videos_per_pixel': int(np.max(unified_count)) if pixels_with_data > 0 else 0
    }
    
    print(f"   ✅ Espace unifié créé!")
    print(f"      - Couverture: {unified_space_info['coverage']:.1%}")
    print(f"      - Pixels avec données: {pixels_with_data}/{total_pixels}")
    print(f"      - Maximum vidéos par pixel: {unified_space_info['max_videos_per_pixel']}")
    
    return unified_heatmap, unified_space_info


class HeatmapVisualizer:
    """
    Visualiseur de heatmaps de collinéarité.
    """
    
    def __init__(self):
        """Initialise le visualiseur."""
        pass

    def display_individual_heatmap(
        self,
        global_heatmap: np.ndarray,
        stats: Dict,
        colormap: str = 'viridis',
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Affiche la visualisation d'une heatmap individuelle.
        
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
        self,
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

    def display_global_absolute_heatmap(
        self,
        global_heatmap: np.ndarray,
        global_stats: Dict,
        colormap: str = 'viridis',
        figsize: Tuple[int, int] = (14, 10)
    ) -> None:
        """
        Affiche la heatmap globale absolue dans l'espace de coordonnées d'image original.
        
        Args:
            global_heatmap: Heatmap globale absolue (H, W)
            global_stats: Dictionnaire des statistiques globales
            colormap: Colormap à utiliser
            figsize: Taille de la figure
        """
        print(f"\n🖼️  Affichage de la heatmap globale absolue...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Masquer les pixels sans données (score = 0)
        masked_heatmap = np.ma.masked_where(global_heatmap == 0, global_heatmap)
        
        # Afficher avec les coordonnées d'image standard (origine en haut à gauche)
        im = ax.imshow(masked_heatmap, cmap=colormap, aspect='equal')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score de Collinéarité Absolu Moyen', rotation=270, labelpad=20)
        
        # Titre et informations
        n_videos = global_stats['n_videos']
        coverage = global_stats['coverage']['coverage_ratio']
        # Gérer les deux formats de clés (ancien et nouveau)
        scores_key = 'global_absolute_scores' if 'global_absolute_scores' in global_stats else 'global_scores'
        mean_score = global_stats[scores_key]['mean']
        filter_status = "Avec filtre" if "filtered" in global_stats['config'] else "Sans filtre"
        
        # Extraire le seuil de manière robuste
        threshold_text = ""
        if "filtered" in global_stats['config'] and "thresh" in global_stats['config']:
            try:
                threshold_part = global_stats['config'].split('thresh')[1]
                threshold_text = f" (seuil: {threshold_part})"
            except (IndexError, ValueError):
                threshold_text = ""
        
        ax.set_title(f'Heatmap Collinéarité Absolue Globale - {n_videos} Vidéos\n'
                    f'{filter_status}{threshold_text} | Couverture: {coverage:.1%} | Score moyen: {mean_score:.3f}')
        ax.set_xlabel('Position X (pixels absolus)')
        ax.set_ylabel('Position Y (pixels absolus)')
        
        # Ajouter des statistiques en texte
        dimensions = global_stats['dimensions']
        max_videos_per_pixel = global_stats['pixel_statistics']['max_videos_per_pixel']
        mean_videos_per_pixel = global_stats['pixel_statistics']['mean_videos_per_pixel']
        
        stats_text = (f"Vidéos: {global_stats['video_ids']}\n"
                     f"Filtrage: {'Oui' if 'filtered' in global_stats['config'] else 'Non'}\n"
                     f"Référence: Coordonnées absolues\n" 
                     f"Grille: {dimensions[1]}x{dimensions[0]}\n"
                     f"Max vidéos/pixel: {max_videos_per_pixel}\n"
                     f"Médiane: {global_stats[scores_key]['median']:.3f}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Afficher les statistiques de couverture
        print(f"\n📊 Statistiques de couverture:")
        print(f"   - Pixels avec données: {global_stats['coverage']['pixels_with_data']:,}")
        print(f"   - Pixels totaux: {global_stats['coverage']['total_pixels']:,}")
        print(f"   - Couverture: {coverage:.1%}")
        print(f"   - Moyenne de vidéos par pixel: {mean_videos_per_pixel:.1f}")
        
        plt.tight_layout()
        plt.show()

    def display_global_relative_heatmap(
        self,
        global_heatmap: np.ndarray,
        global_stats: Dict,
        colormap: str = 'viridis',
        figsize: Tuple[int, int] = (14, 10)
    ) -> None:
        """
        Affiche la heatmap globale relative dans l'espace unifié.
        
        Args:
            global_heatmap: Heatmap globale (H, W)
            global_stats: Dictionnaire des statistiques globales
            colormap: Colormap à utiliser
            figsize: Taille de la figure
        """
        print(f"\n🖼️  Affichage de la heatmap globale relative...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Récupérer les informations de l'espace unifié
        unified_space = global_stats['unified_space']
        bounds = unified_space['bounds']
        x_coords = unified_space['coordinates']['x']
        y_coords = unified_space['coordinates']['y']
        
        # Masquer les pixels sans données (score = 0)
        masked_heatmap = np.ma.masked_where(global_heatmap == 0, global_heatmap)
        
        # Créer la heatmap avec les vraies coordonnées relatives
        extent = [bounds['x_min'], bounds['x_max'], bounds['y_max'], bounds['y_min']]  # Note: y inversé pour imshow
        im = ax.imshow(masked_heatmap, cmap=colormap, aspect='equal', extent=extent)
        
        # Ajouter des lignes de référence pour montrer le centre (0,0)
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score de Collinéarité Relative Moyen', rotation=270, labelpad=20)
        
        # Titre et informations
        n_videos = global_stats['n_videos']
        coverage = unified_space['coverage']
        mean_score = global_stats['global_relative_scores']['mean']
        filter_status = "Avec filtre" if "filtered" in global_stats['config'] else "Sans filtre"
        
        # Extraire le seuil de manière robuste
        threshold_text = ""
        if "filtered" in global_stats['config'] and "thresh" in global_stats['config']:
            try:
                threshold_part = global_stats['config'].split('thresh')[1]
                threshold_text = f" (seuil: {threshold_part})"
            except (IndexError, ValueError):
                threshold_text = ""
        
        ax.set_title(f'Heatmap Collinéarité Relative Globale - {n_videos} Vidéos\n'
                    f'{filter_status}{threshold_text} | Couverture: {coverage:.1%} | Score moyen: {mean_score:.3f}')
        ax.set_xlabel('Position Relative X (pixels par rapport au point de fuite moyen)')
        ax.set_ylabel('Position Relative Y (pixels par rapport au point de fuite moyen)')
        
        # Ajouter des statistiques en texte
        resolution_x = unified_space['resolution']['x']
        resolution_y = unified_space['resolution']['y']
        dimensions = unified_space['dimensions']
        
        stats_text = (f"Vidéos: {global_stats['video_ids']}\n"
                     f"Filtrage: {'Oui' if 'filtered' in global_stats['config'] else 'Non'}\n"
                     f"Référence: Centre (0,0)\n" 
                     f"Grille: {dimensions[1]}x{dimensions[0]}\n"
                     f"Résolution: {resolution_x:.1f}x{resolution_y:.1f}\n"
                     f"Médiane: {global_stats['global_relative_scores']['median']:.3f}")
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Afficher les moyennes utilisées et l'espace relatif
        print(f"\n📊 Espace relatif unifié:")
        print(f"   - Limites X: [{bounds['x_min']:.1f}, {bounds['x_max']:.1f}] pixels")
        print(f"   - Limites Y: [{bounds['y_min']:.1f}, {bounds['y_max']:.1f}] pixels")
        print(f"   - Résolution: {resolution_x:.2f} x {resolution_y:.2f} pixels/unité")
        print(f"   - Centre (0,0) correspond au point de fuite moyen de chaque vidéo")
        
        print(f"\n📊 Moyennes utilisées par vidéo:")
        for video_id in global_stats['video_ids']:
            mean_pixel = global_stats['individual_stats'][video_id]['mean_pixel']
            print(f"   - Vidéo {video_id}: ({mean_pixel[0]:.2f}, {mean_pixel[1]:.2f}) pixels absolus")
        
        plt.tight_layout()
        plt.show()

    def create_global_relative_heatmap_from_saved(
        self,
        video_ids: List[int],
        config: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Crée une heatmap globale en chargeant et combinant les heatmaps absolues sauvegardées
        dans un espace relatif unifié.
        
        Args:
            video_ids: Liste des IDs de vidéos à traiter
            config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
            
        Returns:
            Tuple (global_heatmap, global_stats)
        """
        print(f"\n🌍 CRÉATION DE LA HEATMAP GLOBALE RELATIVE À PARTIR DES FICHIERS SAUVEGARDÉS")
        print("=" * 80)
        
        all_heatmaps = {}
        all_stats = {}
        all_coordinate_mappings = {}
        
        # 1. Charger chaque heatmap absolue et la convertir en relative
        for video_id in video_ids:
            print(f"\n📹 Traitement vidéo {video_id}...")
            
            # Charger la moyenne des pixels
            print(f"   📍 Chargement de la moyenne des pixels...")
            mean_pixel = load_mean_pixel_for_video(video_id)
            if mean_pixel is None:
                print(f"⚠️  Vidéo {video_id} ignorée: impossible de charger la moyenne")
                continue
            
            # Charger la heatmap absolue
            print(f"   📊 Chargement de la heatmap absolue...")
            absolute_heatmap, absolute_stats = load_individual_heatmap(video_id, config)
            if absolute_heatmap is None or absolute_stats is None:
                print(f"⚠️  Vidéo {video_id} ignorée: impossible de charger la heatmap absolue")
                continue
            
            # Convertir en heatmap relative
            print(f"   🔄 Conversion en heatmap relative...")
            relative_heatmap, relative_stats, coordinate_mapping = convert_absolute_to_relative_heatmap(
                absolute_heatmap, mean_pixel, absolute_stats
            )
            
            all_heatmaps[video_id] = relative_heatmap
            all_stats[video_id] = relative_stats
            all_coordinate_mappings[video_id] = coordinate_mapping
            
            print(f"   ✅ Vidéo {video_id} traitée avec succès")
            print(f"      - Référence moyenne: ({mean_pixel[0]:.2f}, {mean_pixel[1]:.2f})")
            print(f"      - Score moyen: {relative_stats['relative_scores']['mean']:.3f}")
        
        if not all_heatmaps:
            print("❌ Aucune vidéo n'a pu être traitée!")
            return None, None
        
        # 2. Créer l'espace relatif unifié
        global_heatmap, unified_space_info = create_unified_relative_space(
            all_heatmaps, all_coordinate_mappings
        )
        
        # 3. Calculer les statistiques globales
        print(f"\n📊 Calcul des statistiques globales...")
        
        valid_scores = global_heatmap[global_heatmap != 0]
        
        # Collecter toutes les moyennes de pixels utilisées
        all_mean_pixels = [stats['mean_pixel'] for stats in all_stats.values()]
        
        global_stats = {
            'video_ids': list(all_heatmaps.keys()),
            'n_videos': len(all_heatmaps),
            'unified_space': unified_space_info,
            'mean_pixels_used': all_mean_pixels,
            'config': config,
            'global_relative_scores': {
                'mean': float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0,
                'median': float(np.median(valid_scores)) if len(valid_scores) > 0 else 0,
                'std': float(np.std(valid_scores)) if len(valid_scores) > 0 else 0,
                'min': float(np.min(valid_scores)) if len(valid_scores) > 0 else 0,
                'max': float(np.max(valid_scores)) if len(valid_scores) > 0 else 0,
            },
            'individual_stats': all_stats,
            'source': 'loaded_from_saved_absolute_heatmaps_unified_relative_space'
        }
        
        print(f"   ✅ Heatmap globale créée!")
        print(f"      - Vidéos combinées: {global_stats['n_videos']}")
        print(f"      - Espace relatif: {unified_space_info['dimensions'][1]} x {unified_space_info['dimensions'][0]}")
        print(f"      - Couverture: {unified_space_info['coverage']:.1%}")
        print(f"      - Score global moyen: {global_stats['global_relative_scores']['mean']:.3f}")
        print(f"      - Score global médian: {global_stats['global_relative_scores']['median']:.3f}")
        
        return global_heatmap, global_stats


def main():
    """Fonction principale interactive pour visualiser les heatmaps."""
    print("🖼️  VISUALISATION DES HEATMAPS DE COLLINÉARITÉ")
    print("=" * 60)
    
    # Lister les configurations disponibles
    available_configs = list_available_heatmaps()
    
    if not available_configs:
        print("❌ Aucune heatmap sauvegardée trouvée!")
        print("   Veuillez d'abord exécuter la génération des heatmaps.")
        return
    
    print(f"\n📋 Configurations disponibles:")
    for i, config in enumerate(available_configs):
        print(f"   {i+1}. {config}")
    
    # Demander à l'utilisateur quelle configuration utiliser
    while True:
        try:
            choice = input(f"\nChoisissez une configuration (1-{len(available_configs)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_configs):
                selected_config = available_configs[choice_idx]
                break
            else:
                print(f"❌ Choix invalide. Veuillez entrer un nombre entre 1 et {len(available_configs)}.")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide.")
    
    # Demander le type de visualisation
    print(f"\n📊 Type de visualisation:")
    print(f"   1. Heatmap globale absolue")
    print(f"   2. Heatmap globale relative")
    print(f"   3. Comparaison heatmaps individuelles")
    
    while True:
        try:
            viz_choice = input(f"\nChoisissez le type de visualisation (1-3): ").strip()
            if viz_choice in ["1", "2", "3"]:
                break
            else:
                print(f"❌ Choix invalide. Veuillez entrer 1, 2 ou 3.")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide.")
    
    # Configuration
    video_ids = [0, 1, 2, 3, 4]
    visualizer = HeatmapVisualizer()
    
    print(f"\n⚙️  Configuration sélectionnée: {selected_config}")
    
    if viz_choice == "1":
        # Heatmap globale absolue
        global_heatmap, metadata = load_global_heatmap(selected_config)
        if global_heatmap is not None and metadata is not None:
            if metadata.get('type') == 'absolute':
                visualizer.display_global_absolute_heatmap(global_heatmap, metadata)
            else:
                print("❌ Cette configuration ne contient pas de heatmap globale absolue")
        else:
            print("❌ Impossible de charger la heatmap globale")
    
    elif viz_choice == "2":
        # Heatmap globale relative
        global_heatmap, global_stats = visualizer.create_global_relative_heatmap_from_saved(video_ids, selected_config)
        if global_heatmap is not None and global_stats is not None:
            visualizer.display_global_relative_heatmap(global_heatmap, global_stats)
        else:
            print("❌ Impossible de créer la heatmap globale relative")
    
    elif viz_choice == "3":
        # Comparaison heatmaps individuelles
        all_heatmaps = {}
        all_stats = {}
        
        for video_id in video_ids:
            heatmap, stats = load_individual_heatmap(video_id, selected_config)
            if heatmap is not None and stats is not None:
                all_heatmaps[video_id] = heatmap
                all_stats[video_id] = stats
        
        if all_heatmaps:
            visualizer.display_comparison_plot(all_heatmaps, all_stats)
        else:
            print("❌ Aucune heatmap individuelle disponible")
    
    print(f"\n✅ Visualisation terminée!")


if __name__ == "__main__":
    main()