"""
Générateur de flux optiques pour le challenge de calibration caméra.

Ce module traite les vidéos du dataset pour générer des champs de flux optique 
(optical flow) en appliquant des masques pour filtrer les éléments perturbateurs 
(véhicules, capot de voiture).

Le flux optique généré servira ensuite à l'estimation du point de fuite pour 
déterminer les angles de pitch et yaw de la caméra.

Auteur: Projet calib_challenge
Dernière modification: 2024
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.utilities.paths import (
    get_inputs_dir, 
    get_flows_dir,
    get_masks_dir,
    ensure_dir_exists
)
from src.core.flow import calculate_flow
from src.core.segmentation import VehicleDetector


class VideoFlowProcessor:
    """
    Processeur principal pour générer les flux optiques à partir des vidéos.
    
    Cette classe gère :
    - Le chargement et traitement frame par frame des vidéos
    - L'application des masques (manuels + détection automatique véhicules)  
    - Le calcul et stockage des flux optiques
    - La sauvegarde optimisée des résultats
    """
    
    def __init__(self):
        """Initialise le processeur avec les chemins du projet."""
        self.labeled_videos_dir = get_inputs_dir() / "labeled"
        self.flows_dir = ensure_dir_exists(get_flows_dir())
        self.masks_dir = get_masks_dir()  # Masques manuels si disponibles
        
        # État de traitement
        self.vehicle_detector = VehicleDetector()
        self.reset_frame_state()
    
    def reset_frame_state(self):
        """Remet à zéro l'état entre les vidéos."""
        self.prev_gray = None
        self.prev_vehicle_mask = None
        self.manual_mask = None
        self.flows_storage = None
        self.frame_count = 0
    
    def _count_video_frames(self, video_path: Path) -> int:
        """
        Compte manuellement le nombre de frames réel dans une vidéo.
        
        OpenCV peut donner des comptes incorrects, donc on compte manuellement
        avec une limite de sécurité.
        
        Args:
            video_path: Chemin vers le fichier vidéo
            
        Returns:
            Nombre de frames dans la vidéo
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        frame_count = 0
        max_frames = 2000  # Limite de sécurité
        
        print(f"📊 Comptage des frames de {video_path.name}...")
        while frame_count < max_frames:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        
        cap.release()
        print(f"✅ {frame_count} frames détectées")
        return frame_count
    
    def _load_manual_mask(self, video_index: int, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Charge le masque manuel pour une vidéo si disponible.
        
        Args:
            video_index: Index de la vidéo (0-4)
            frame_shape: Forme (height, width) des frames
            
        Returns:
            Masque manuel ou masque vide si non trouvé
        """
        mask_path = self.masks_dir / f"{video_index}_mask.png"
        
        if mask_path.exists():
            print(f"🎭 Chargement masque manuel: {mask_path}")
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask
            else:
                print(f"⚠️  Erreur lecture masque: {mask_path}")
        
        # Masque vide par défaut
        height, width = frame_shape
        return np.zeros((height, width), dtype=np.uint8)
    
    def _process_single_frame(self, frame: np.ndarray, total_frames: int) -> Optional[np.ndarray]:
        """
        Traite une frame individuelle pour calculer le flux optique.
        
        Args:
            frame: Frame courante en BGR
            total_frames: Nombre total de frames (pour affichage progression)
            
        Returns:
            Flux optique calculé ou None pour la première frame
        """
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Affichage progression
        if total_frames > 1:
            progress = (self.frame_count / (total_frames - 1)) * 100
            print(f"\r🔄 Progression: {progress:.1f}%", end="", flush=True)
        
        # Pas de flux pour la première frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.frame_count += 1
            return None
        
        # Détection véhicules et création masque combiné
        current_vehicle_mask = self.vehicle_detector.detect_vehicles(frame)
        current_vehicle_mask = self.vehicle_detector.dilate_mask(current_vehicle_mask)
        
        # Combinaison des masques
        combined_mask = self.vehicle_detector.combine_masks(
            self.manual_mask, 
            current_vehicle_mask, 
            self.prev_vehicle_mask or np.zeros_like(current_vehicle_mask)
        )
        
        # Calcul du flux optique avec masquage
        flow, _, _ = calculate_flow(self.prev_gray, gray, combined_mask)
        
        # Mise à jour de l'état
        self.prev_gray = gray
        self.prev_vehicle_mask = current_vehicle_mask
        self.frame_count += 1
        
        return flow
    
    def process_video(self, video_index: int) -> bool:
        """
        Traite une vidéo complète pour générer ses flux optiques.
        
        Args:
            video_index: Index de la vidéo à traiter (0-4)
            
        Returns:
            True si succès, False sinon
        """
        # Réinitialisation pour nouvelle vidéo
        self.reset_frame_state()
        
        # Vérification existence vidéo
        video_path = self.labeled_videos_dir / f"{video_index}.hevc"
        if not video_path.exists():
            print(f"❌ Vidéo non trouvée: {video_path}")
            return False
        
        print(f"\n🎬 Traitement vidéo {video_index}: {video_path}")
        
        try:
            # Comptage frames et ouverture vidéo
            total_frames = self._count_video_frames(video_path)
            cap = cv2.VideoCapture(str(video_path))
            
            # Lecture première frame pour initialisation
            ret, first_frame = cap.read()
            if not ret:
                print("❌ Impossible de lire la première frame")
                return False
            
            # Initialisation masques
            self.manual_mask = self._load_manual_mask(video_index, first_frame.shape[:2])
            
            # Initialisation stockage flux
            flows_list = []
            
            # Traitement frame par frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Retour au début
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                flow = self._process_single_frame(frame, total_frames)
                if flow is not None:
                    flows_list.append(flow)
            
            cap.release()
            print("\n✅ Traitement terminé!")
            
            # Conversion en array numpy et sauvegarde
            if flows_list:
                flows_array = np.stack(flows_list, axis=0)
                output_path = self.flows_dir / f"{video_index}.npy"
                
                print(f"💾 Sauvegarde: {output_path}")
                print(f"📏 Shape: {flows_array.shape}")
                
                np.save(output_path, flows_array)
                return True
            else:
                print("⚠️  Aucun flux généré")
                return False
                
        except Exception as e:
            print(f"❌ Erreur traitement vidéo {video_index}: {e}")
            return False
    
    def process_multiple_videos(self, video_indices: Optional[List[int]] = None) -> None:
        """
        Traite plusieurs vidéos en séquence.
        
        Args:
            video_indices: Liste des indices à traiter, ou None pour toutes (0-4)
        """
        if video_indices is None:
            video_indices = list(range(5))  # Toutes les vidéos par défaut
        
        print(f"🚀 Démarrage génération flux optiques")
        print(f"📂 Dossier vidéos: {self.labeled_videos_dir}")
        print(f"📂 Dossier sortie: {self.flows_dir}")
        print(f"🎯 Vidéos à traiter: {video_indices}")
        
        success_count = 0
        for video_idx in video_indices:
            if self.process_video(video_idx):
                success_count += 1
        
        print(f"\n🏁 Traitement terminé: {success_count}/{len(video_indices)} vidéos réussies")


def main(video_indices: Optional[List[int]] = None) -> None:
    """
    Point d'entrée principal pour la génération de flux optiques.
    
    Args:
        video_indices: Liste des indices de vidéos à traiter.
                      Si None, traite les vidéos 1,2,3,4 (pas la 0 non labellisée)
    """
    # Configuration par défaut : vidéos labellisées (pas la 0)
    if video_indices is None:
        video_indices = [1, 2, 3, 4]
    
    # Traitement
    processor = VideoFlowProcessor()
    processor.process_multiple_videos(video_indices)


if __name__ == "__main__":
    # Traitement des vidéos labellisées par défaut
    main()
