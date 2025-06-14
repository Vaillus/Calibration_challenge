#!/usr/bin/env python3
"""
Génération de heatmaps de collinéarité.

Ce module génère les heatmaps de collinéarité individuelles par vidéo et peut créer
une heatmap globale moyenne absolue à partir des heatmaps individuelles.

Usage:
    from src.utilities.heatmap_generator import HeatmapGenerator
    
    generator = HeatmapGenerator()
    generator.generate_all_individual_heatmaps(video_ids=[0,1,2,3,4])
    generator.create_and_save_global_absolute_heatmap(video_ids=[0,1,2,3,4])
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import mlx.core as mx
import gc
import json

from src.utilities.load_flows import load_flows
from src.utilities.paths import get_outputs_dir, get_data_dir
from src.core.flow_filter import FlowFilterBatch
from src.core.collinearity_scorer_batch import BatchCollinearityScorer


class HeatmapGenerator:
    """
    Générateur de heatmaps de collinéarité.
    """
    
    def __init__(self):
        """Initialise le générateur."""
        self.scorer = BatchCollinearityScorer()
    
    def load_labels_for_video(self, video_id: int) -> Optional[np.ndarray]:
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
        self,
        video_id: int, 
        apply_filter: bool = True,
        min_threshold: float = 13.0,
        chunk_size: int = 100
    ) -> Tuple[np.ndarray, Dict]:
        """
        Crée une heatmap de collinéarité globale pour une vidéo.
        
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
        labels_array = self.load_labels_for_video(video_id)
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
            collinearity_maps_chunk = self.scorer.compute_colinearity_map_batch(filtered_flows_chunk, reference_points_batch)
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

    def save_individual_heatmaps(
        self,
        all_heatmaps: Dict[int, np.ndarray],
        all_stats: Dict[int, Dict],
        apply_filter: bool,
        min_threshold: Optional[float] = None
    ) -> str:
        """
        Sauvegarde les heatmaps individuelles dans data/intermediate/heatmaps/
        avec la structure hiérarchique organisée par type de filtrage.
        
        Args:
            all_heatmaps: Dictionnaire {video_id: heatmap}
            all_stats: Dictionnaire {video_id: stats}
            apply_filter: Si un filtre a été appliqué
            min_threshold: Seuil utilisé pour le filtre
            
        Returns:
            Nom de la configuration créée
        """
        print(f"\n💾 Sauvegarde des heatmaps individuelles...")
        
        # Créer la structure de dossiers hiérarchique
        data_dir = get_data_dir()
        heatmaps_dir = data_dir / "intermediate" / "heatmaps"
        
        # Nom du dossier de filtrage
        if apply_filter:
            filter_dir_name = f"filtered_thresh{min_threshold:.1f}"
        else:
            filter_dir_name = "unfiltered"
        
        # Créer les dossiers
        filter_dir = heatmaps_dir / filter_dir_name
        individual_dir = filter_dir / "individual"
        global_dir = filter_dir / "global"
        
        individual_dir.mkdir(parents=True, exist_ok=True)
        global_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   📁 Structure créée: {filter_dir}")
        
        # Sauvegarder chaque heatmap dans individual/
        for video_id, heatmap in all_heatmaps.items():
            # Nom du fichier pour la heatmap (simplifié)
            heatmap_filename = f"video{video_id}_heatmap.npy"
            heatmap_path = individual_dir / heatmap_filename
            
            # Sauvegarder la heatmap
            np.save(heatmap_path, heatmap)
            print(f"   ✅ Heatmap vidéo {video_id} sauvegardée: {heatmap_filename}")
            
            # Sauvegarder les statistiques associées
            stats_filename = f"video{video_id}_stats.json"
            stats_path = individual_dir / stats_filename
            
            # Préparer les stats pour la sérialisation JSON
            stats_to_save = all_stats[video_id].copy()
            
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"   ✅ Stats vidéo {video_id} sauvegardées: {stats_filename}")
        
        # Sauvegarder un fichier de résumé dans le dossier de filtrage
        summary_filename = "summary.json"
        summary_path = filter_dir / summary_filename
        
        summary_data = {
            'processing_info': {
                'apply_filter': apply_filter,
                'min_threshold': min_threshold,
                'videos_processed': list(all_heatmaps.keys()),
                'total_videos': len(all_heatmaps)
            },
            'global_stats': {
                video_id: {
                    'coverage': stats['pixels_coverage'],
                    'mean_score': stats['global_scores']['mean'],
                    'median_score': stats['global_scores']['median'],
                    'uses_labels': stats['uses_labels']
                }
                for video_id, stats in all_stats.items()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"   ✅ Résumé global sauvegardé: {summary_filename}")
        print(f"   📁 Dossier de destination: {filter_dir}")
        print(f"   📁 Heatmaps individuelles: {individual_dir}")
        print(f"   📁 Dossier global préparé: {global_dir}")
        
        return filter_dir_name

    def generate_all_individual_heatmaps(
        self,
        video_ids: List[int],
        apply_filter: bool = True,
        min_threshold: float = 13.0,
        chunk_size: int = 300
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict], str]:
        """
        Génère toutes les heatmaps individuelles pour les vidéos spécifiées.
        
        Args:
            video_ids: Liste des IDs de vidéos à traiter
            apply_filter: Si True, applique le filtre de norme
            min_threshold: Seuil minimum de norme pour le filtrage
            chunk_size: Taille des chunks pour le traitement
            
        Returns:
            Tuple (all_heatmaps, all_stats, config_name)
        """
        print("🚀 GÉNÉRATION DES HEATMAPS DE COLLINÉARITÉ INDIVIDUELLES")
        print("=" * 60)
        
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
            
            heatmap, stats = self.create_collinearity_heatmap_for_video(
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
            return {}, {}, ""
        
        # Sauvegarder les heatmaps
        config_name = self.save_individual_heatmaps(all_heatmaps, all_stats, apply_filter, min_threshold)
        
        print(f"\n✅ Génération terminée! Configuration: {config_name}")
        return all_heatmaps, all_stats, config_name

    def create_global_absolute_heatmap_from_saved(
        self,
        video_ids: List[int],
        config: str
    ) -> Tuple[np.ndarray, Dict]:
        """
        Crée une heatmap globale absolue en moyennant directement les heatmaps absolues
        sans transformation relative.
        
        Args:
            video_ids: Liste des IDs de vidéos à traiter
            config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
            
        Returns:
            Tuple (global_absolute_heatmap, global_stats)
        """
        print(f"\n🌍 CRÉATION DE LA HEATMAP GLOBALE ABSOLUE")
        print("=" * 50)
        
        all_heatmaps = {}
        all_stats = {}
        
        # 1. Charger chaque heatmap absolue
        for video_id in video_ids:
            print(f"\n📹 Chargement vidéo {video_id}...")
            
            # Charger la heatmap absolue depuis les utilitaires
            from src.utilities.heatmap_loader import load_individual_heatmap
            absolute_heatmap, absolute_stats = load_individual_heatmap(video_id, config)
            if absolute_heatmap is None or absolute_stats is None:
                print(f"⚠️  Vidéo {video_id} ignorée: impossible de charger la heatmap absolue")
                continue
            
            all_heatmaps[video_id] = absolute_heatmap
            all_stats[video_id] = absolute_stats
            
            print(f"   ✅ Vidéo {video_id} chargée avec succès")
            print(f"      - Dimensions: {absolute_heatmap.shape}")
            print(f"      - Couverture: {absolute_stats['pixels_coverage']:.1%}")
            print(f"      - Score moyen: {absolute_stats['global_scores']['mean']:.3f}")
        
        if not all_heatmaps:
            print("❌ Aucune vidéo n'a pu être chargée!")
            return None, None
        
        # 2. Vérifier que toutes les heatmaps ont les mêmes dimensions
        first_shape = next(iter(all_heatmaps.values())).shape
        for video_id, heatmap in all_heatmaps.items():
            if heatmap.shape != first_shape:
                print(f"❌ Erreur: La vidéo {video_id} a des dimensions différentes: {heatmap.shape} vs {first_shape}")
                return None, None
        
        print(f"\n🔗 Création de la heatmap globale absolue...")
        print(f"   - Dimensions communes: {first_shape}")
        print(f"   - Nombre de vidéos: {len(all_heatmaps)}")
        
        # 3. Calculer la moyenne des heatmaps absolues
        heatmap_sum = np.zeros(first_shape, dtype=np.float64)
        heatmap_count = np.zeros(first_shape, dtype=np.int64)
        
        for video_id, heatmap in all_heatmaps.items():
            # Masquer les pixels sans données (score = 0)
            valid_mask = heatmap != 0
            heatmap_sum[valid_mask] += heatmap[valid_mask]
            heatmap_count[valid_mask] += 1
            
            valid_pixels = np.sum(valid_mask)
            print(f"   📹 Vidéo {video_id}: {valid_pixels} pixels valides ajoutés")
        
        # Calculer la moyenne (éviter division par zéro)
        global_absolute_heatmap = np.where(heatmap_count > 0, heatmap_sum / heatmap_count, 0)
        
        # 4. Calculer les statistiques globales
        print(f"\n📊 Calcul des statistiques globales...")
        
        valid_scores = global_absolute_heatmap[global_absolute_heatmap != 0]
        pixels_with_data = np.sum(heatmap_count > 0)
        total_pixels = first_shape[0] * first_shape[1]
        
        global_stats = {
            'video_ids': list(all_heatmaps.keys()),
            'n_videos': len(all_heatmaps),
            'config': config,
            'dimensions': first_shape,
            'coordinate_system': 'absolute_image_coordinates',
            'global_absolute_scores': {
                'mean': float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0,
                'median': float(np.median(valid_scores)) if len(valid_scores) > 0 else 0,
                'std': float(np.std(valid_scores)) if len(valid_scores) > 0 else 0,
                'min': float(np.min(valid_scores)) if len(valid_scores) > 0 else 0,
                'max': float(np.max(valid_scores)) if len(valid_scores) > 0 else 0,
            },
            'coverage': {
                'pixels_with_data': int(pixels_with_data),
                'total_pixels': int(total_pixels),
                'coverage_ratio': float(pixels_with_data / total_pixels)
            },
            'pixel_statistics': {
                'max_videos_per_pixel': int(np.max(heatmap_count)) if pixels_with_data > 0 else 0,
                'mean_videos_per_pixel': float(np.mean(heatmap_count[heatmap_count > 0])) if pixels_with_data > 0 else 0
            },
            'individual_stats': all_stats,
            'source': 'loaded_from_saved_absolute_heatmaps_direct_average'
        }
        
        print(f"   ✅ Heatmap globale absolue créée!")
        print(f"      - Vidéos combinées: {global_stats['n_videos']}")
        print(f"      - Dimensions: {first_shape[1]} x {first_shape[0]}")
        print(f"      - Couverture: {global_stats['coverage']['coverage_ratio']:.1%}")
        print(f"      - Score global moyen: {global_stats['global_absolute_scores']['mean']:.3f}")
        print(f"      - Score global médian: {global_stats['global_absolute_scores']['median']:.3f}")
        print(f"      - Maximum vidéos par pixel: {global_stats['pixel_statistics']['max_videos_per_pixel']}")
        
        return global_absolute_heatmap, global_stats

    def save_global_absolute_heatmap(
        self,
        global_heatmap: np.ndarray,
        global_stats: Dict,
        config: str
    ) -> None:
        """
        Sauvegarde la heatmap globale absolue et ses métadonnées.
        
        Args:
            global_heatmap: Heatmap globale absolue (H, W)
            global_stats: Dictionnaire des statistiques globales
            config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
        """
        print(f"\n💾 Sauvegarde de la heatmap globale absolue...")
        
        data_dir = get_data_dir()
        global_dir = data_dir / "intermediate" / "heatmaps" / config / "global"
        
        # Créer le dossier global s'il n'existe pas
        global_dir.mkdir(parents=True, exist_ok=True)
        
        # Noms des fichiers
        heatmap_filename = "global_heatmap.npy"
        metadata_filename = "global_heatmap_metadata.json"
        
        heatmap_path = global_dir / heatmap_filename
        metadata_path = global_dir / metadata_filename
        
        try:
            # 1. Sauvegarder la heatmap
            np.save(heatmap_path, global_heatmap)
            print(f"   ✅ Heatmap sauvegardée: {heatmap_path}")
            
            # 2. Préparer les métadonnées pour heatmap absolue
            metadata = {
                'type': 'absolute',
                'config': config,
                'video_ids': global_stats['video_ids'],
                'n_videos': global_stats['n_videos'],
                'source': global_stats['source'],
                'coordinate_system': global_stats['coordinate_system'],
                'dimensions': global_stats['dimensions'],
                'global_scores': global_stats['global_absolute_scores'],
                'coverage': global_stats['coverage'],
                'pixel_statistics': global_stats['pixel_statistics'],
                'individual_stats_summary': {
                    video_id: {
                        'pixels_coverage': stats['pixels_coverage'],
                        'scores_mean': stats['global_scores']['mean'],
                        'scores_median': stats['global_scores']['median']
                    }
                    for video_id, stats in global_stats['individual_stats'].items()
                }
            }
            
            # Ajouter les informations de création
            metadata['creation_info'] = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Heatmap globale de collinéarité absolute',
                'coordinate_system': 'Coordonnées absolues d\'image',
                'reference': 'Coordonnées d\'image originales'
            }
            
            # 3. Sauvegarder les métadonnées
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"   ✅ Métadonnées sauvegardées: {metadata_path}")
            
            # 4. Résumé de la sauvegarde
            heatmap_size_mb = heatmap_path.stat().st_size / (1024 * 1024)
            metadata_size_kb = metadata_path.stat().st_size / 1024
            
            print(f"   📊 Résumé de la sauvegarde:")
            print(f"      - Type: ABSOLUTE")
            print(f"      - Dossier: {global_dir}")
            print(f"      - Heatmap: {heatmap_filename} ({heatmap_size_mb:.1f} MB)")
            print(f"      - Métadonnées: {metadata_filename} ({metadata_size_kb:.1f} KB)")
            print(f"      - Dimensions: {global_heatmap.shape}")
            print(f"      - Espace absolu: {global_stats['dimensions']}")
            
        except Exception as e:
            print(f"   ❌ Erreur lors de la sauvegarde: {e}")

    def create_and_save_global_absolute_heatmap(
        self,
        video_ids: List[int],
        config: str
    ) -> bool:
        """
        Crée et sauvegarde la heatmap globale absolue à partir des heatmaps individuelles.
        
        Args:
            video_ids: Liste des IDs de vidéos à combiner
            config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
            
        Returns:
            True si succès, False sinon
        """
        global_heatmap, global_stats = self.create_global_absolute_heatmap_from_saved(video_ids, config)
        
        if global_heatmap is None or global_stats is None:
            print("❌ Impossible de créer la heatmap globale!")
            return False
        
        self.save_global_absolute_heatmap(global_heatmap, global_stats, config)
        print(f"\n✅ Heatmap globale absolue créée et sauvegardée pour la configuration: {config}")
        return True