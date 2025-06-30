"""
Générateur de prédictions à partir de flux optiques avec filtrage.

Ce module génère des prédictions de points de fuite à partir de flux optiques 
pré-calculés. Il peut appliquer différents filtres et pondérations sur les flux 
avant l'estimation. Il fait partie du 4ème arc du projet : filtrage des vecteurs.

Fonctionnalités :
- Génération de prédictions par batch avec filtrage optionnel
- Support de multiples configurations de filtrage
- Traitement efficace des flux optiques pré-calculés (.npz float16)
- Sauvegarde des prédictions en format angles
- Gestion efficace de la mémoire pour gros datasets

Auteur: Projet calib_challenge  
Dernière modification: 2024
"""

import time
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import gc
import json

from src.utilities.paths import (
    get_flows_dir,
    get_pred_dir,
    ensure_dir_exists,
    get_intermediate_dir
)
from src.utilities.ground_truth import read_ground_truth_pixels, read_ground_truth_angles
from src.core.flow_filter import FlowFilterBatch
from src.utilities.pixel_angle_converter import pixels_to_angles
from src.core.optimizers import AdamOptimizer
from src.utilities.project_constants import get_project_constants
from src.utilities.load_flows import load_flows
from src.utilities.load_mean_point import load_mean_point




def process_multiple_videos(
    video_indices: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    run_name: str = "default",
    mean_point_run_name: str = "8",
    is_mean_point_used: bool = False
) -> None:
    """
    Traite plusieurs vidéos avec la même configuration.
    
    Args:
        video_indices: Liste des indices à traiter (défaut: toutes 0-4)
        config: Configuration de filtrage
        run_name: Nom du run d'expérimentation
    """
    if video_indices is None:
        video_indices = list(range(5))  # Toutes les vidéos
    
    print(f"🚀 DÉMARRAGE RUN: {run_name}")
    print(f"📂 Flux source: {get_flows_dir()}")
    print(f"📂 Prédictions: {get_pred_dir(run_name)}")
    print(f"🎯 Vidéos: {video_indices}")
    
    success_count = 0
    for video_idx in video_indices:
        video_processed = process_video(video_idx, config, run_name, mean_point_run_name=mean_point_run_name, is_mean_point_used=is_mean_point_used)
        if video_processed:
            success_count += 1
    
    print(f"\n🏁 RUN '{run_name}' TERMINÉ: {success_count}/{len(video_indices)} vidéos réussies")

def process_video(
    video_index: int, 
    config: Optional[Dict[str, Any]] = None,
    run_name: str = "default",
    batch_size: int = 200,
    mean_point_run_name: str = "8",
    is_mean_point_used: bool = False
) -> bool:
    """
    Traite une vidéo complète avec la configuration donnée.
    
    Args:
        video_index: Index de la vidéo (0-4)
        config: Configuration de filtrage (None = pas de filtrage)
        run_name: Nom du run d'expérimentation
        batch_size: Taille des batches pour traitement
        
    Returns:
        True si succès, False sinon
    """
    print(f"\n🎬 Traitement vidéo {video_index} - Run: {run_name}")
    
    # Préparation
    pred_run_dir = prepare_output_directories(run_name, config)
    if is_mean_point_used:
        mean_point = load_mean_point(mean_point_run_name, video_index)
    else:
        mean_point = None
    # Chargement des données
    flows_data = load_video_flows(video_index)
    if flows_data is None:
        return False
    
    total_frames = flows_data.shape[0]
    
    # Traitement principal
    all_predictions, total_time = process_batches(
        flows_data, config, batch_size,
         mean_point
    )
    
    # Post-traitement
    predictions_angles = post_process_predictions(all_predictions)
    
    # Sauvegarde et statistiques
    save_results_and_print_stats(
        predictions_angles, pred_run_dir, video_index, total_time, total_frames
    )
    
    # Nettoyage final
    del flows_data, all_predictions
    gc.collect()
    
    return True

def process_batches(
    flows_data: mx.array, 
    config: Optional[Dict[str, Any]], 
    batch_size: int,
    mean_point: Optional[Tuple[float, float]] = None
) -> Tuple[List[mx.array], float]:
    """Traite les flux par batches et retourne les prédictions et le temps total."""
    total_frames = flows_data.shape[0]
    all_predictions = []
    total_time = 0
    
    print(f"🔄 Traitement par batches de {batch_size}...")
    if config is None:
        print("⚙️  Mode: AUCUN FILTRAGE")
    else:
        print("⚙️  Mode: FILTRAGE ACTIVÉ")
    
    for start_idx in range(0, total_frames, batch_size):
        end_idx = min(start_idx + batch_size, total_frames)
        
        # Extraction batch
        flow_batch = flows_data[start_idx:end_idx]
        mx.eval(flow_batch)
        
        # Optimisation avec/sans filtrage
        start_time = time.time()
        predictions = optimize_batch_with_filtering(flow_batch, config, mean_point)
        mx.eval(predictions)
        batch_time = time.time() - start_time
        total_time += batch_time
        
        all_predictions.extend(predictions)
        
        print(f"  Frames {start_idx:4d}-{end_idx-1:4d}: {batch_time:.2f}s")
        
        # Nettoyage mémoire
        del flow_batch, predictions
        gc.collect()
        
        # Nettoyage périodique
        if (start_idx // batch_size) % 10 == 9:
            print(f"    🧹 Nettoyage mémoire batch {start_idx // batch_size + 1}")
            if all_predictions:
                temp_pred = mx.stack(all_predictions, axis=0)
                mx.eval(temp_pred)
                all_predictions = list(temp_pred)
                del temp_pred
            gc.collect()
    
    return all_predictions, total_time

def optimize_batch_with_filtering(
    flow_batch: mx.array, 
    config: Optional[Dict[str, Any]] = None,
    mean_point: Tuple[float, float] = None
) -> mx.array:
    """
    Optimise un batch de flux optiques avec filtrage optionnel.
    
    Args:
        flow_batch: Batch de flux optiques de shape (batch_size, h, w, 2)
        config: Configuration de filtrage. Si None, aucun filtrage appliqué.
                Structure:
                {
                    'filtering': {
                        'norm': {'is_used': bool, 'min_threshold': float},
                        'colinearity': {'is_used': bool, 'min_threshold': float}
                    },
                    'weighting': {
                        'norm': {'is_used': bool, 'type': str},
                        'colinearity': {'is_used': bool, 'type': str}
                    }
                }
        plateau_threshold: Seuil d'amélioration minimum pour continuer (défaut: 1e-4)
        plateau_patience: Nombre d'itérations pour vérifier amélioration (défaut: 3)
        
    Returns:
        Prédictions des points de fuite de shape (batch_size, 2)
    """
    
    # Vérifier si du filtrage est demandé
    filtering_enabled = False
    if config is not None:
        filtering_enabled = (
            config.get('norm', {}).get('is_used', False) or 
            config.get('colinearity', {}).get('is_used', False) or
            config.get('heatmap', {}).get('is_used', False)
        )
    
    if filtering_enabled:
        # Appliquer pipeline filtrage/pondération
        flow_filter = FlowFilterBatch(config)
        flow_batch, weights_batch = flow_filter.filter_and_weight(flow_batch, mean_point)
        # flow_batch = flow_filter.filter(flow_batch)
        mx.eval(flow_batch)
        mx.eval(weights_batch)
    # mean_point = mx.array([mean_point[0], mean_point[1]], dtype=mx.float32)
    # starting_points = mx.tile(mean_point, (flow_batch.shape[0], 1))

    # Optimiser avec l'optimiseur Adam
    predictions = AdamOptimizer().optimize_batch(
        flow_batch, 
        # starting_points=starting_points,
        weights_batch=weights_batch,
    )
    mx.eval(predictions)
    
    return predictions


def prepare_output_directories(
    run_name: str, 
    config: Optional[Dict[str, Any]]
) -> Path:
    """Prépare les dossiers de sortie et sauvegarde la configuration."""
    pred_run_dir = ensure_dir_exists(get_pred_dir(run_name))
    config_file = pred_run_dir / "config.json"
    save_config(config_file, config, run_name)
    return pred_run_dir


def load_video_flows(video_index: int) -> Optional[mx.array]:
    """Charge les flux optiques d'une vidéo."""
    try:
        flows_data = load_flows(video_index, return_mlx=True)
        mx.eval(flows_data)
        print(f"📊 Total frames: {flows_data.shape[0]}")
        return flows_data
    except Exception as e:
        print(f"❌ Erreur chargement flux vidéo {video_index}: {e}")
        return None

def post_process_predictions(all_predictions: List[mx.array]) -> np.array:
    """Post-traite les prédictions : stack, correction offset, conversion angles."""
    # Finalisation prédictions
    all_predictions = mx.stack(all_predictions, axis=0)
    mx.eval(all_predictions)
    predictions_np = np.array(all_predictions)
    
    # Correction offset temporel (duplication première frame)
    first_prediction = predictions_np[0:1]
    corrected_predictions_pixels = np.vstack([first_prediction, predictions_np])
    
    # Conversion pixels -> angles
    corrected_predictions_angles = []
    for x, y in corrected_predictions_pixels:
        pitch, yaw = pixels_to_angles(x, y)
        corrected_predictions_angles.append([pitch, yaw])
    
    return np.array(corrected_predictions_angles)


def save_results_and_print_stats(
    predictions_angles: np.array, 
    pred_run_dir: Path, 
    video_index: int, 
    total_time: float, 
    total_frames: int
) -> None:
    """Sauvegarde les résultats et affiche les statistiques."""
    output_file = pred_run_dir / f"{video_index}.txt"
    np.savetxt(output_file, predictions_angles, fmt='%.15e')
    
    # Statistiques
    fps = total_frames / total_time
    print(f"\n✅ Vidéo {video_index} terminée:")
    print(f"   ⏱️  Temps total: {total_time:.2f}s")
    print(f"   🚀 FPS: {fps:.2f}")
    print(f"   💾 Prédictions: {output_file}")
    print(f"   📏 Frames corrigées: {len(predictions_angles)}")


def save_config(config_file: Path, config: Optional[Dict[str, Any]], run_name: str) -> None:
    """Sauvegarde la configuration du run."""
    config_to_save = {
        "run_name": run_name,
        "config": config,
        "description": "Aucun filtrage appliqué" if config is None else "Filtrage appliqué",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    config_to_save['config']['heatmap']['path'] = str(config_to_save['config']['heatmap']['path'])
    with open(config_file, 'w') as f:
        json.dump(config_to_save, f, indent=2)





def main(
    video_indices: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    run_name: str = "default",
    mean_point_run_name: str = "8",
    is_mean_point_used: bool = False
) -> None:
    """
    Point d'entrée principal pour la génération de prédictions à partir de flux.
    
    Args:
        video_indices: Vidéos à traiter (défaut: toutes)
        config: Configuration de filtrage (défaut: aucun filtrage)
        run_name: Nom du run pour organisation des résultats
    """
    process_multiple_videos(video_indices, config, run_name, mean_point_run_name, is_mean_point_used)


if __name__ == "__main__":
    config = {
    "norm": {
      "is_used": True,
      "k": -10,
      "x0": 1.0
    },
    "colinearity": {
      "is_used": True,
      "k": 118.25,
      "x0": 1.1053
    },
    "heatmap": {
      "is_used": True,
      "weight": 0.79,
      "path": "/Users/hugovaillaud/Documents/code/ibiza_project/calib_challenge/data/intermediate/heatmaps/unfiltered/global/global_heatmap.npy"
    }
  }
    # Exemple d'utilisation sans filtrage
    main(video_indices=[0,1,2,3,4], config=config, run_name="7", mean_point_run_name="8", is_mean_point_used=True) 