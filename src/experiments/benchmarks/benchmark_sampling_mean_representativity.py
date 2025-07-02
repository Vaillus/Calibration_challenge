#!/usr/bin/env python3
"""
Version debug simplifiée pour investiguer le bug de sampling.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from src.utilities.worst_errors import select_frames_from_all_deciles
from src.utilities.load_predictions import load_filtered_predictions, load_predictions
from src.utilities.pixel_angle_converter import angles_to_pixels


def compute_sampling_distance(video_id: int = 0, 
                            run_name: str = "5_4", 
                            n_frames_per_decile: int = 110,
                            seed: int = 42, verbose: bool = False) -> Dict:
    """
    Debug la différence entre point moyen global et échantillonné.
    """
    # print(f"🔍 Vidéo {video_id}, {n_frames_per_decile} frames/décile, seed {seed}")
    
    # Charger les prédictions
    filtered_predictions = load_filtered_predictions(video_id, frame_index=None, predictions_dir=run_name)
    all_predictions = np.array(load_predictions(video_id, frame_index=None, predictions_dir=run_name))
    
    # Point moyen global
    global_mean = np.mean(filtered_predictions, axis=0)
    
    # Obtenir les frames samplées
    sampled_frames = select_frames_from_all_deciles(
        run_name=run_name,
        n_frames_per_decile=n_frames_per_decile,
        video_ids=[video_id],
        seed=seed
    )
    
    target_frame_ids = [frame_id for _, frame_id in sampled_frames]
    sampled_predictions = all_predictions[target_frame_ids]
    sampled_mean = np.mean(sampled_predictions, axis=0)
    
    # Calculer la différence
    distance = np.linalg.norm(global_mean - sampled_mean)
    
    if verbose:
        print(f"  Sampled frames: {len(target_frame_ids)}")
        print(f"  Global: ({global_mean[0]:.1f}, {global_mean[1]:.1f}) | "
            f"Échantillon: ({sampled_mean[0]:.1f}, {sampled_mean[1]:.1f}) | "
            f"Distance: {distance:.1f}px")
    
    return {
        'global_mean': global_mean,
        'sampled_mean': sampled_mean,
        'sampled_frames_count': len(target_frame_ids),
        'distance': distance
    }


def analyze_multi_seed_sampling(video_id: int = 0,
                               run_name: str = "5_4", 
                               n_frames_per_decile: int = 110,
                               seeds: List[int] = None) -> Dict:
    """
    Analyse les statistiques de distance sur plusieurs seeds.
    """
    if seeds is None:
        seeds = list(range(10))  # Seeds 0-9 par défaut
    
    print(f"📊 Multi-seed analyse: Vidéo {video_id}, {n_frames_per_decile} frames/décile, {len(seeds)} seeds")
    
    distances = []
    for seed in seeds:
        result = compute_sampling_distance(
            video_id=video_id,
            run_name=run_name,
            n_frames_per_decile=n_frames_per_decile,
            seed=seed,
            verbose=False
        )
        distances.append(result['distance'])
    
    # Calculer les statistiques
    stats = {
        'distances': distances,
        'min': np.min(distances),
        'max': np.max(distances), 
        'mean': np.mean(distances),
        'median': np.median(distances),
        'seeds_tested': seeds
    }
    
    print(f"  Distances: min={stats['min']:.1f}, max={stats['max']:.1f}, "
          f"mean={stats['mean']:.1f}, median={stats['median']:.1f}px")
    
    return stats


def analyze_all_videos_sampling(run_name: str = "5_4", 
                               n_frames_per_decile: int = 110,
                               seeds: List[int] = None,
                               video_ids: List[int] = None) -> Dict:
    """
    Analyse les statistiques de distance sur toutes les vidéos.
    """
    if video_ids is None:
        video_ids = [0, 1, 2, 3, 4]  # Toutes les vidéos par défaut
    
    print(f"🎬 Analyse toutes vidéos: {n_frames_per_decile} frames/décile")
    
    all_results = {}
    for video_id in video_ids:
        result = analyze_multi_seed_sampling(
            video_id=video_id,
            run_name=run_name,
            n_frames_per_decile=n_frames_per_decile,
            seeds=seeds
        )
        all_results[video_id] = result
    
    # Statistiques globales
    all_distances = []
    for result in all_results.values():
        all_distances.extend(result['distances'])
    
    global_stats = {
        'min': np.min(all_distances),
        'max': np.max(all_distances),
        'mean': np.mean(all_distances),
        'median': np.median(all_distances)
    }
    
    print(f"📈 Global: min={global_stats['min']:.1f}, max={global_stats['max']:.1f}, "
          f"mean={global_stats['mean']:.1f}, median={global_stats['median']:.1f}px")
    
    return {
        'video_results': all_results,
        'global_stats': global_stats
    }


if __name__ == "__main__":
    # Test multi-seed sur toutes les vidéos
    print("🚨 ANALYSE MULTI-SEED TOUTES VIDÉOS")
    print("=" * 50)
    
    test_configs = [
        (1, "très petit"),
        (10, "petit"), 
        (25, "moyen"),
        (50, "grand"),
        (110, "très grand")
    ]
    
    for n_frames, description in test_configs:
        print(f"\n📊 Test {description}: {n_frames} frames/décile")
        analyze_all_videos_sampling(
            run_name="5_4", 
            n_frames_per_decile=n_frames
        ) 