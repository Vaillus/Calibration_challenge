import cv2
import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path='yolov8x-seg.pt', device='mps'):
        """
        Initialise le détecteur de véhicules.
        
        Args:
            model_path: Chemin vers le modèle YOLO
            device: Device pour l'inférence (cpu, cuda, mps)
        """
        self.model = YOLO(model_path, verbose=False)
        self.device = device
    
    def detect_vehicles(self, frame, conf_threshold=0.3):
        """
        Détecte les véhicules dans une frame.
        
        Args:
            frame: Image à analyser
            conf_threshold: Seuil de confiance pour les détections
        
        Returns:
            Masque des véhicules détectés
        """
        # Faire la prédiction sur la frame
        results = self.model(frame, device=self.device, verbose=False)
        
        # Créer un masque pour les véhicules détectés
        vehicle_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Filtrer les prédictions pour ne garder que les véhicules avec score > conf_threshold
        for r in results:
            if r.masks is not None:  # Vérifier si des masques ont été détectés
                for i, (box, mask, conf, cls) in enumerate(zip(r.boxes.xyxy, r.masks.data, r.boxes.conf, r.boxes.cls)):
                    if conf > conf_threshold and cls <= 8:  # 0: human, 1: bicycle, 2: car, 3: motorcycle, 4: airplane, 5: bus, 6: train, 7: truck, 8: boat
                        # Convertir le masque en format numpy
                        mask_np = mask.cpu().numpy()
                        # Redimensionner le masque à la taille de l'image
                        mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
                        # Ajouter au masque des véhicules
                        vehicle_mask = cv2.bitwise_or(vehicle_mask, (mask_np > 0.5).astype(np.uint8) * 255)
        
        return vehicle_mask
    
    def dilate_mask(self, mask, kernel_size=10):
        """
        Dilate le masque des véhicules.
        
        Args:
            mask: Masque à dilater
            kernel_size: Taille du kernel de dilatation
        
        Returns:
            Masque dilaté
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    
    def combine_masks(self, manual_mask, current_vehicle_mask, prev_vehicle_mask):
        """
        Combine les différents masques.
        
        Args:
            manual_mask: Masque manuel
            current_vehicle_mask: Masque des véhicules de la frame courante
            prev_vehicle_mask: Masque des véhicules de la frame précédente
        
        Returns:
            Masque combiné
        """
        combined_mask = cv2.bitwise_or(manual_mask, current_vehicle_mask)
        return cv2.bitwise_or(combined_mask, prev_vehicle_mask) 