import cv2
import numpy as np
import os
import json
from flow import calculate_flow
from segmentation import VehicleDetector

class FlowConfig:
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
        
        # Dossiers (chemins relatifs au dossier calib_challenge)
        directories = config.get('directories', {})
        self.video_dir = os.path.join(self.project_root, 'labeled')
        self.flow_dir = os.path.join(self.project_root, 'flows')
        self.mask_dir = os.path.join(self.project_root, 'masks')

class VideoProcessor:
    def __init__(self, config, total_frames):
        self.config = config
        self.prev_gray = None
        self.frame_count = 0
        self.total_frames = total_frames
        # Initialiser un tableau pour stocker tous les flows
        # On ne sait pas encore la taille exacte des flows, on l'initialisera au premier flow
        self.flows = None
        # Initialiser le détecteur de véhicules
        self.detector = VehicleDetector()
        self.manual_mask = None
        self.prev_vehicle_mask = None

    def initialize_masks(self, frame_shape):
        """Initialize masks with the correct size"""
        height, width = frame_shape[:2]
        self.manual_mask = np.zeros((height, width), dtype=np.uint8)
        self.prev_vehicle_mask = np.zeros((height, width), dtype=np.uint8)

    def load_manual_mask(self, video_name):
        """Load manual mask if it exists"""
        mask_path = os.path.join(self.config.mask_dir, f"{video_name}_mask.png")
        if os.path.exists(mask_path):
            print(f"Chargement du masque depuis {mask_path}")
            self.manual_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Masque non trouvé à {mask_path}, utilisation d'un masque vide")

    def process_frame(self, frame):
        """Process a single frame and store flow"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate progress
        progress = (self.frame_count / (self.total_frames - 1)) * 100
        print(f"\rProgression: {progress:.1f}%", end="", flush=True)
        
        if self.prev_gray is not None:
            # Détecter les véhicules et créer le masque
            current_vehicle_mask = self.detector.detect_vehicles(frame)
            current_vehicle_mask = self.detector.dilate_mask(current_vehicle_mask)
            combined_mask = self.detector.combine_masks(
                self.manual_mask, 
                current_vehicle_mask, 
                self.prev_vehicle_mask
            )
            self.prev_vehicle_mask = current_vehicle_mask
            
            # Calculate optical flow with combined mask
            flow, _, _ = calculate_flow(self.prev_gray, gray, combined_mask)
            
            # Initialize flows array if not done yet
            if self.flows is None:
                self.flows = np.zeros((self.total_frames - 1, *flow.shape), dtype=flow.dtype)
            
            # Store flow in array
            self.flows[self.frame_count - 1] = flow
        
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

def process_video(video_path, config, video_index):
    """Process a video and save flows"""
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return None
    
    # Get total frames
    total_frames, cap = get_video_frame_count(cap, video_path)
    
    # Initialize processor
    processor = VideoProcessor(config, total_frames)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la première frame")
        return None
    
    # Initialize masks with frame size
    processor.initialize_masks(frame.shape)
    
    # Initialize first frame
    processor.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load manual mask
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    processor.load_manual_mask(video_name)
    
    # Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processor.process_frame(frame)
    
    print("\nTraitement terminé!")
    cap.release()
    
    # Save all flows at once
    if processor.flows is not None:
        flow_file = os.path.join(config.flow_dir, f"{video_index}.npy")
        np.save(flow_file, processor.flows)
        print(f"Flows sauvegardés dans {flow_file}")
        print(f"Shape des flows: {processor.flows.shape}")

def main(video_indices=None):
    """
    Process videos to generate optical flows.
    
    Args:
        video_indices (list, optional): List of video indices to process. If None, processes all videos.
    """
    # Load configuration
    config = FlowConfig()
    
    # Display paths for debugging
    print("\nChemins utilisés :")
    print(f"Vidéos : {os.path.abspath(config.video_dir)}")
    print(f"Flows : {os.path.abspath(config.flow_dir)}")
    print(f"Masques : {os.path.abspath(config.mask_dir)}\n")
    
    # Create flows directory if it doesn't exist
    os.makedirs(config.flow_dir, exist_ok=True)
    
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
        
        process_video(video_path, config, video_index)
    
    print("\nTraitement de toutes les vidéos terminé!")

if __name__ == "__main__":
    video_indices = [1, 2, 3, 4] 
    main(video_indices=video_indices)
