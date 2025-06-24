"""
Script de prédiction de points de fuite à partir de vidéos de conduite.

Utilise l'analyse de flux optique avec segmentation véhicule optionnelle.
Deux méthodes disponibles : 'flow' (séparation directe) et 'colinearity' (optimisation).


Configuration via config.json dans le même dossier.
"""

import cv2
import numpy as np
import os
import json
import mlx.core as mx
from typing import Optional, Dict, Any, List
from pathlib import Path
import gc
import time

from src.core.flow import calculate_flow, find_separation_points
from src.utilities.pixel_angle_converter import pixels_to_angles
from src.core.segmentation import VehicleDetector
from src.core.optimizers import AdamOptimizer
from src.utilities.paths import get_labeled_dir, get_pred_dir, get_masks_dir, ensure_dir_exists
from src.utilities.project_constants import get_project_constants
from src.utilities.fix_predictions import fix_predictions
from src.core.flow_filter import FlowFilterSample


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Charge la configuration depuis config.json."""
    if config_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'config.json')
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return {
        'prediction_method': config.get('prediction_method', 'colinearity'),
        'use_segmentation': config.get('use_segmentation', True),
        'filter_config': config.get('filter_config', None),
        'optimizer_config': config.get('optimizer_config', None)
    }


class VideoProcessor:
    """
    Processeur vidéo pour prédiction de points de fuite.
    
    Gère le traitement complet d'une vidéo avec flux optique,
    segmentation véhicule optionnelle et estimation du point de fuite.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = None
        self.manual_mask = None
        self.prev_vehicle_mask = None
        self.prev_gray = None
        self.frame_count = 0
        self.total_frames = 0
        self.results = []
        
        # Initialisation filtre et optimiseur avec configs
        filter_config = config.get('filter_config')
        if filter_config:
            self.filter = FlowFilterSample(filter_config)
        else:
            self.filter = FlowFilterSample()  # Utilise config par défaut
        
        optimizer_config = config.get('optimizer_config')
        if optimizer_config:
            self.optimizer = AdamOptimizer(**optimizer_config)
        else:
            self.optimizer = AdamOptimizer()  # Utilise config par défaut

    def process_video(
        self, 
        video_index: int,
        run_name: str = "default"
    ) -> bool:
        """
        Traite une vidéo complète et sauvegarde les prédictions.
        
        Args:
            video_index: Index de la vidéo 
            run_name: Nom du run d'expérimentation
            
        Returns:
            True si succès, False sinon
        """
        # Create video path
        video_path = get_labeled_dir() / f"{video_index}.hevc"
        if not os.path.exists(video_path):
            print(f"⚠️  Vidéo {video_index} introuvable")
            return False

        # Préparation dossiers de sortie
        pred_run_dir = self._prepare_output_directories(run_name)
        
        # Initialisation
        cap = self._initialize_video(str(video_path))
        if cap is None:
            return False
        
        self._setup_processing_components(str(video_path))
        
        # Première frame
        first_frame = self._process_first_frame(cap)
        if first_frame is None:
            cap.release()
            return False
        
        # Traitement de toutes les frames
        self._process_all_frames(cap)
        
        # Nettoyage
        cap.release()
        gc.collect()
        
        # Sauvegarde prédictions
        self._save_predictions(pred_run_dir, video_index)
        
        print(f"\n✅ Vidéo {video_index} terminée: {len(self.results)} frames")
        return True

    def _prepare_output_directories(self, run_name: str) -> Path:
        """Prépare les dossiers de sortie et sauvegarde la configuration."""
        pred_run_dir = ensure_dir_exists(get_pred_dir(run_name))
        
        # Sauvegarde configuration
        config_file = pred_run_dir / "config.json"
        self._save_config(config_file, run_name)
        
        return pred_run_dir

    def _save_config(self, config_file: Path, run_name: str) -> None:
        """Sauvegarde la configuration du run."""
        config_to_save = {
            "run_name": run_name,
            "config": self.config,
            "description": f"Make predictions - méthode {self.config['prediction_method']}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_to_save, f, indent=2)

    def _save_predictions(self, pred_run_dir: Path, video_index: int) -> None:
        """Sauvegarde les prédictions dans le dossier du run."""
        output_file = pred_run_dir / f"{video_index}.txt"
        
        with open(output_file, 'w') as f:
            for yaw, pitch in self.results:
                f.write(f"{yaw:.6f} {pitch:.6f}\n")
        
        print(f"💾 Prédictions: {output_file}")

    def _initialize_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """Initialise la capture vidéo et compte les frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la vidéo")
            return None
        
        self.total_frames, cap = self._get_video_frame_count(cap, video_path)
        return cap
    
    def _get_video_frame_count(self, cap: cv2.VideoCapture, video_path: str) -> tuple:
        """Compte le nombre réel de frames dans la vidéo."""
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = 0
        max_frames = 2000
        while frame_count < max_frames:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        
        print(f"📊 Total frames: {frame_count}")
        
        cap.release()
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        return frame_count, cap

    def _setup_processing_components(self, video_path: str) -> None:
        """Configure les composants de traitement."""
        frame_width = get_project_constants()["frame_width"]
        frame_height = get_project_constants()["frame_height"]
        
        # Initialisation segmentation si activée
        if self.config['use_segmentation']:
            self.detector = VehicleDetector()
            self.manual_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            self.prev_vehicle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            
            # Chargement masque manuel
            self._load_manual_mask(video_path)

    def _load_manual_mask(self, video_path: str) -> None:
        """Charge le masque manuel si disponible."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mask_path = get_masks_dir() / f"{video_name}_mask.png"
        
        if os.path.exists(mask_path):
            print(f"📂 Masque chargé: {mask_path}")
            self.manual_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            print("📂 Pas de masque manuel trouvé")

    def _process_first_frame(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Traite la première frame et initialise prev_gray."""
        ret, frame = cap.read()
        if not ret:
            print("❌ Impossible de lire la première frame")
            return None
        
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _process_all_frames(self, cap: cv2.VideoCapture) -> None:
        """Traite toutes les frames de la vidéo."""
        while True:
            # Affichage progression
            progress = (self.frame_count / (self.total_frames - 1)) * 100
            print(f"\rProgression: {progress:.1f}%", end="", flush=True)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            self._process_single_frame(frame)

    def _process_single_frame(self, frame: np.ndarray) -> None:
        """Traite une frame individuelle."""
        # Conversion grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gestion masques
        combined_mask = self._create_mask(frame)
        # Calcul flux optique et filtrage
        flow, _, _ = calculate_flow(self.prev_gray, gray, combined_mask)
        flow, weights = self.filter.filter_and_weight(flow)
        
        # Prédiction selon méthode choisie
        x, y = self._predict_vanishing_point(flow, combined_mask, weights)
        
        # Conversion angles et stockage
        yaw, pitch = pixels_to_angles(x, y)
        self._store_result(x, y, yaw, pitch)
        
        # Mise à jour état
        self.prev_gray = gray

    def _create_mask(self, frame: np.ndarray) -> np.ndarray:
        """détecte les véhicules sur la frame, crée le masque correspondant 
        dilate le masque 
        combine avec le masque du capot et le masque de la frame précédente.
        """
        if not self.config['use_segmentation']:
            return np.zeros_like(self.prev_gray)
        
        current_vehicle_mask = self.detector.detect_vehicles(frame)
        current_vehicle_mask = self.detector.dilate_mask(current_vehicle_mask)
        combined_mask = self.detector.combine_masks(
            self.manual_mask, 
            current_vehicle_mask, 
            self.prev_vehicle_mask
        )
        self.prev_vehicle_mask = current_vehicle_mask
        return combined_mask

    def _predict_vanishing_point(self, flow: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> tuple:
        """Prédit le point de fuite selon la méthode configurée."""
        if self.config['prediction_method'] == "flow":
            return find_separation_points(flow, mask)
        elif self.config['prediction_method'] == "colinearity":
            vanishing_point = self.optimizer.optimize_single(flow, weights=weights)
            return map(int, vanishing_point) # cleaner than converting to int element by element
        else:
            raise ValueError(f"Méthode inconnue: {self.config['prediction_method']}")

    def _store_result(self, x: int, y: int, yaw: float, pitch: float) -> None:
        """Stocke le résultat de prédiction."""
        self.results.append([yaw, pitch])



def main(
    video_indices: Optional[List[int]] = None,
    run_name: str = "default"
) -> None:
    """
    Traite les vidéos pour prédiction de points de fuite.
    
    Args:
        video_indices: Indices des vidéos à traiter (défaut: toutes)
        run_name: Nom du run d'expérimentation
    """
    if video_indices is None:
        video_indices = list(range(5))

    # Configuration
    config = load_config()
    
    print(f"🚀 DÉMARRAGE RUN: {run_name}")
    print(f"📂 Prédictions: {get_pred_dir(run_name)}")
    print(f"🎯 Vidéos: {video_indices}")
    print(f"⚙️  Méthode: {config['prediction_method']}")
    
    # Traitement vidéos
    success_count = 0
    for video_index in video_indices:
        print(f"\n🎬 Traitement vidéo {video_index} - Run: {run_name}")
        # Traitement
        processor = VideoProcessor(config)
        success = processor.process_video(video_index, run_name)
        if success:
            success_count += 1

    print(f"\n🏁 RUN '{run_name}' TERMINÉ: {success_count}/{len(video_indices)} vidéos réussies")


if __name__ == "__main__":
    main() 