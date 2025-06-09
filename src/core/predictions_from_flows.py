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
    ensure_dir_exists
)
from src.utilities.ground_truth import read_ground_truth_pixels, read_ground_truth_angles
from src.core.flow_filter import FlowFilterBatch
from src.utilities.pixel_angle_converter import pixels_to_angles
from src.core.optimizers import AdamOptimizer

# Camera parameters
FOCAL_LENGTH = 910


def optimize_batch_with_filtering(
    flow_batch: mx.array, 
    config: Optional[Dict[str, Any]] = None,
    plateau_threshold: float = 1e-4,
    plateau_patience: int = 3
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
    batch_size = flow_batch.shape[0]
    
    # Vérifier si du filtrage est demandé
    filtering_enabled = False
    if config is not None:
        filtering_enabled = (
            config.get('filtering', {}).get('norm', {}).get('is_used', False) or 
            config.get('filtering', {}).get('colinearity', {}).get('is_used', False) or
            config.get('weighting', {}).get('norm', {}).get('is_used', False) or
            config.get('weighting', {}).get('colinearity', {}).get('is_used', False)
        )
    
    if filtering_enabled:
        # Appliquer pipeline filtrage/pondération
        flow_filter = FlowFilterBatch(config)
        filtered_flows, weights = flow_filter.filter_and_weight(flow_batch)
        
        mx.eval(filtered_flows)
        mx.eval(weights)
        
        # Utiliser flux filtrés pour optimisation
        flows_to_optimize = filtered_flows
    else:
        # Pas de filtrage - utiliser flux originaux directement
        flows_to_optimize = flow_batch
    
    # Optimiser avec l'optimiseur Adam
    predictions = AdamOptimizer(plateau_threshold=plateau_threshold, plateau_patience=plateau_patience).optimize_batch(flows_to_optimize)
    mx.eval(predictions)
    
    return predictions


class FlowPredictor:
    """
    Classe principale pour la génération de prédictions à partir de flux optiques.
    
    Gère le traitement complet d'une ou plusieurs vidéos avec différentes
    configurations de filtrage et sauvegarde les prédictions résultantes.
    """
    
    def __init__(self):
        """Initialise le générateur de prédictions avec les chemins du projet."""
        self.flows_dir = get_flows_dir()
        self.pred_base_dir = get_pred_dir()
        
    def process_video(
        self, 
        video_index: int, 
        config: Optional[Dict[str, Any]] = None,
        run_name: str = "default",
        batch_size: int = 200
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
        
        # Vérification existence fichier flux
        npz_path = self.flows_dir / f"{video_index}_float16.npz"
        if not npz_path.exists():
            print(f"❌ Fichier flux non trouvé: {npz_path}")
            return False
        
        # Préparation dossier sortie
        pred_run_dir = ensure_dir_exists(self.pred_base_dir / run_name)
        output_file = pred_run_dir / f"{video_index}.txt"
        
        # Sauvegarde configuration
        config_file = pred_run_dir / "config.json"
        self._save_config(config_file, config, run_name)
        
        try:
            # Chargement flux (float16 -> float32)
            print(f"📂 Chargement flux depuis: {npz_path}")
            with np.load(npz_path) as data:
                flows_data_f16 = mx.array(data['flow'])
            mx.eval(flows_data_f16)
            
            flows_data = flows_data_f16.astype(mx.float32)
            mx.eval(flows_data)
            del flows_data_f16
            gc.collect()
            
            total_frames = flows_data.shape[0]
            print(f"📊 Total frames: {total_frames}")
            
            # Chargement ground truth pour conversion angles
            _, image_width, image_height = read_ground_truth_angles(video_index)
            
            # Traitement par batches
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
                predictions = optimize_batch_with_filtering(flow_batch, config)
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
                yaw, pitch = pixels_to_angles(x, y, FOCAL_LENGTH, image_width, image_height)
                corrected_predictions_angles.append([yaw, pitch])
            corrected_predictions_angles = np.array(corrected_predictions_angles)
            
            # Sauvegarde résultats
            np.savetxt(output_file, corrected_predictions_angles, fmt='%.15e')
            
            # Statistiques
            fps = total_frames / total_time
            print(f"\n✅ Vidéo {video_index} terminée:")
            print(f"   ⏱️  Temps total: {total_time:.2f}s")
            print(f"   🚀 FPS: {fps:.2f}")
            print(f"   💾 Prédictions: {output_file}")
            print(f"   📏 Frames corrigées: {len(corrected_predictions_angles)}")
            
            # Nettoyage final
            del flows_data, all_predictions
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur traitement vidéo {video_index}: {e}")
            return False
    
    def process_multiple_videos(
        self,
        video_indices: Optional[List[int]] = None,
        config: Optional[Dict[str, Any]] = None,
        run_name: str = "default"
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
        print(f"📂 Flux source: {self.flows_dir}")
        print(f"📂 Prédictions: {self.pred_base_dir / run_name}")
        print(f"🎯 Vidéos: {video_indices}")
        
        success_count = 0
        for video_idx in video_indices:
            if self.process_video(video_idx, config, run_name):
                success_count += 1
        
        print(f"\n🏁 RUN '{run_name}' TERMINÉ: {success_count}/{len(video_indices)} vidéos réussies")
    
    def _save_config(self, config_file: Path, config: Optional[Dict[str, Any]], run_name: str) -> None:
        """Sauvegarde la configuration du run."""
        config_to_save = {
            "run_name": run_name,
            "config": config,
            "description": "Aucun filtrage appliqué" if config is None else "Filtrage appliqué",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)


def main(
    video_indices: Optional[List[int]] = None,
    config: Optional[Dict[str, Any]] = None,
    run_name: str = "default"
) -> None:
    """
    Point d'entrée principal pour la génération de prédictions à partir de flux.
    
    Args:
        video_indices: Vidéos à traiter (défaut: toutes)
        config: Configuration de filtrage (défaut: aucun filtrage)
        run_name: Nom du run pour organisation des résultats
    """
    predictor = FlowPredictor()
    predictor.process_multiple_videos(video_indices, config, run_name)


if __name__ == "__main__":
    # Exemple d'utilisation sans filtrage
    main(video_indices=[1, 2, 3, 4], config=None, run_name="no_filtering") 