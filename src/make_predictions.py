import cv2
import numpy as np
import os
import json
import argparse
from flow import calculate_flow, find_separation_points
from conversion import pixels_to_angles
from colinearity_optimization import VanishingPointEstimator
from segmentation import VehicleDetector

class PredictionConfig:
    def __init__(self, config_path=None):
        # Si aucun chemin n'est fourni, utiliser le fichier config.json dans le même dossier que le script
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config.json')
        
        # Obtenir le chemin absolu du dossier calib_challenge
        self.project_root = os.path.abspath(os.path.join(script_dir, '..'))
        
        # Charger la configuration depuis le fichier JSON
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Méthode de prédiction
        self.prediction_method = config.get('prediction_method', 'flow')
        
        # Options de segmentation
        self.use_segmentation = config.get('use_segmentation', False)
        
        # Paramètres de prédiction
        self.prediction_parameters = config.get('prediction_parameters', {})
        
        # Paramètres de la caméra (valeur fixe)
        self.focal_length = 910
        
        # Dossiers (chemins relatifs au dossier calib_challenge)
        directories = config.get('directories', {})
        self.video_dir = os.path.join(self.project_root, 'labeled')
        self.pred_dir = os.path.join(self.project_root, directories.get('pred_dir', 'pred/4'))
        self.mask_dir = os.path.join(self.project_root, 'masks')

class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.detector = None
        self.vp_estimator = None
        self.manual_mask = None
        self.prev_vehicle_mask = None
        self.prev_gray = None
        self.frame_count = 0
        self.total_frames = 0
        self.results = []

    def initialize(self, frame_width, frame_height):
        """Initialize processing components"""
        # Initialize vehicle detector if segmentation is enabled
        if self.config.use_segmentation:
            self.detector = VehicleDetector()
            self.manual_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            self.prev_vehicle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        # Get parameters from prediction parameters
        use_max_distance = self.config.prediction_parameters.get('use_max_distance', False)
        use_reoptimization = self.config.prediction_parameters.get('use_reoptimization', False)
        
        # Initialize vanishing point estimator
        self.vp_estimator = VanishingPointEstimator(
            frame_width, 
            frame_height, 
            self.config.focal_length, 
            use_max_distance=use_max_distance,
            use_reoptimization=use_reoptimization
        )

    def load_manual_mask(self, video_name):
        """Load manual mask if it exists"""
        if not self.config.use_segmentation:
            return

        mask_path = os.path.join(self.config.mask_dir, f"{video_name}_mask.png")
        if os.path.exists(mask_path):
            print(f"Chargement du masque depuis {mask_path}")
            self.manual_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Masque non trouvé à {mask_path}, utilisation d'un masque vide")

    def process_frame(self, frame):
        """Process a single frame and return prediction results"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate progress
        progress = (self.frame_count / (self.total_frames - 1)) * 100
        print(f"\rProgression: {progress:.1f}%", end="", flush=True)
        
        # Handle masks if segmentation is enabled
        if self.config.use_segmentation:
            current_vehicle_mask = self.detector.detect_vehicles(frame)
            current_vehicle_mask = self.detector.dilate_mask(current_vehicle_mask)
            combined_mask = self.detector.combine_masks(
                self.manual_mask, 
                current_vehicle_mask, 
                self.prev_vehicle_mask
            )
            self.prev_vehicle_mask = current_vehicle_mask
        else:
            combined_mask = np.zeros_like(gray)
        
        # Calculate optical flow
        flow, _, _ = calculate_flow(self.prev_gray, gray, combined_mask)
        
        # Find separation points based on chosen method
        if self.config.prediction_method == "flow":
            x, y = find_separation_points(flow, combined_mask)
        elif self.config.prediction_method == "colinearity":
            vanishing_point = self.vp_estimator.estimate_vanishing_point(flow, visualize=False)
            x, y = map(int, vanishing_point)
        else:
            raise ValueError(f"Méthode de prédiction inconnue: {self.config.prediction_method}")
        
        # Convert to angles
        yaw, pitch = self.vp_estimator.get_angles((x, y))
        
        # Store results
        self.results.append({
            'frame': self.frame_count,
            'x': x,
            'y': y,
            'yaw': yaw,
            'pitch': pitch
        })
        
        # Update state for next iteration
        self.prev_gray = gray
        self.frame_count += 1

def get_video_frame_count(cap, video_path):
    """Count the actual number of frames in the video"""
    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    max_frames = 2000  # Safety limit
    while frame_count < max_frames:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    
    print(f"Comptage manuel: {frame_count} frames")
    
    cap.release()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    return frame_count, cap

def process_video(video_path, config):
    """Process a video and return prediction results"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get total frames
    total_frames, cap = get_video_frame_count(cap, video_path)
    
    # Initialize processor
    processor = VideoProcessor(config)
    processor.initialize(frame_width, frame_height)
    processor.total_frames = total_frames
    
    # Load manual mask if needed
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    processor.load_manual_mask(video_name)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la première frame")
        return None
    
    # Initialize first frame
    processor.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processor.process_frame(frame)
    
    print("\nTraitement terminé!")
    cap.release()
    
    return processor.results

def save_predictions(results, pred_file):
    """Save prediction results to a file"""
    with open(pred_file, 'w') as f:
        for result in results:
            f.write(f"{result['pitch']:.6f} {result['yaw']:.6f}\n")

def main(video_indices=None):
    """
    Process videos for vanishing point prediction.
    
    Args:
        video_indices (list, optional): List of video indices to process. If None, processes all videos.
    """
    # Load configuration
    config = PredictionConfig()
    
    # Display paths for debugging
    print("\nChemins utilisés :")
    print(f"Vidéos : {os.path.abspath(config.video_dir)}")
    print(f"Prédictions : {os.path.abspath(config.pred_dir)}")
    print(f"Masques : {os.path.abspath(config.mask_dir)}\n")
    
    # Create predictions directory if it doesn't exist
    os.makedirs(config.pred_dir, exist_ok=True)
    
    # Determine which videos to process
    video_indices = video_indices if video_indices is not None else range(5)
    
    # Process selected videos
    for video_index in video_indices:
        print(f"\nTraitement de la vidéo {video_index}...")
        video_path = os.path.join(config.video_dir, f"{video_index}.hevc")
        print(f"Recherche de la vidéo : {os.path.abspath(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"Vidéo {video_index} non trouvée, passage à la suivante...")
            continue
        
        results = process_video(video_path, config)
        if results:
            pred_file = os.path.join(config.pred_dir, f"{video_index}.txt")
            save_predictions(results, pred_file)
            print(f"Prédictions pour la vidéo {video_index} sauvegardées dans {pred_file}")
    
    # Fix predictions to match ground truth length
    from fix_predictions import fix_predictions
    print("\nAjustement des prédictions...")
    fix_predictions(pred_dir=config.pred_dir, gt_dir='calib_challenge/labeled')
    
    print("\nTraitement de toutes les vidéos terminé!")

if __name__ == "__main__":
    video_indices = [0]
    main(video_indices=video_indices) 