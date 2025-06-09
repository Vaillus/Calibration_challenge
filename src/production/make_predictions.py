"""
Camera Calibration Challenge - Vanishing Point Prediction Script
==============================================================

This script processes driving videos to predict vanishing points using optical flow analysis.
It represents an intermediate version of the vanishing point estimation pipeline.

CURRENT APPROACH (Version implemented here):
------------------------------------------
1. **Basic Optical Flow**: Uses cv2.calcOpticalFlowFarneback with default parameters
2. **Vehicle Segmentation**: YOLO-based detection + manual masks for car hood
3. **Two Prediction Methods**:
   - "flow": Direct separation point finding from flow field
   - "colinearity": Vanishing point estimation via colinearity optimization
4. **Simple Processing**: No advanced vector filtering or optimization

RESEARCH EVOLUTION (Based on development history):
------------------------------------------------
This script corresponds to the "2nd-3rd arc" of development:
- ✅ Arc 1: Basic optical flow implementation
- ✅ Arc 2: Vehicle segmentation to filter distracting elements  
- ✅ Arc 3: Colinearity-based vanishing point estimation
- ❌ Arc 4: Advanced vector filtering (NOT implemented here)

KNOWN LIMITATIONS:
-----------------
1. **No Advanced Filtering**: Missing optimized vector filtering that provided 83.8% improvement
2. **Suboptimal Flow Parameters**: Uses default cv2.calcOpticalFlowFarneback parameters
3. **No Vector Quality Assessment**: Doesn't filter vectors by norm, colinearity, or distance
4. **Simple Post-processing**: Basic angle conversion without smoothing

PERFORMANCE:
-----------
- Expected performance: ~200-500% error (intermediate baseline)
- Best achieved performance: ~44% error (with advanced filtering)
- This version likely produces results similar to early development stages

POTENTIAL IMPROVEMENTS (For future versions):
--------------------------------------------
1. **Advanced Vector Filtering**:
   - Norm threshold filtering (optimal: >13)
   - Colinearity filtering with center (optimal: >0.96) 
   - Distance-based weighting from center
   
2. **Optimized Flow Parameters**:
   - pyr_scale=0.7, levels=7, winsize=23, iterations=2
   - poly_n=5, poly_sigma=0.8
   
3. **Intelligent Frame Handling**:
   - Adaptive thresholds for frames with insufficient good vectors
   - Temporal smoothing (exponential moving average)
   - Frame sampling strategies for difficult cases

4. **Enhanced Post-processing**:
   - Zone constraints (legal vanishing point region)
   - Temporal consistency checks
   - Prediction smoothing

USAGE:
------
    # Process all videos
    python make_predictions.py
    
    # Process specific videos
    main(video_indices=[0, 1, 2])

OUTPUT:
-------
- Creates prediction files in pred/{config_dir}/ containing yaw,pitch angles per frame
- Automatically adjusts prediction length to match ground truth
- Processes videos from labeled/ directory using masks from masks/ directory

CONFIG:
-------
Uses config.json for parameters:
- prediction_method: "flow" or "colinearity" 
- use_segmentation: boolean for vehicle detection
- directories: paths for input/output
- prediction_parameters: method-specific settings

DEPENDENCIES:
------------
- OpenCV for video processing and optical flow
- YOLO for vehicle segmentation  
- Custom modules: flow, conversion, colinearity_optimization, segmentation
"""

import cv2
import numpy as np
import os
import json
import argparse
import sys

# Imports absolus propres grâce à pip install -e .
from src.core.flow import calculate_flow, find_separation_points
from src.utilities.pixel_angle_converter import pixels_to_angles
from src.core.colinearity_optimization import VanishingPointEstimator
from src.core.segmentation import VehicleDetector
from src.utilities.paths import get_project_root, get_inputs_dir, get_outputs_dir

class PredictionConfig:
    """
    Configuration manager for vanishing point prediction.
    
    Loads settings from config.json and sets up directory paths.
    This represents a basic configuration system that could be enhanced
    with the advanced filtering parameters discovered during research.
    
    Current Configuration Options:
    - prediction_method: "flow" or "colinearity"
    - use_segmentation: Enable YOLO vehicle detection
    - prediction_parameters: Basic method parameters
    
    Missing Advanced Options (for future enhancement):
    - Vector filtering thresholds (norm, colinearity)
    - Optimized optical flow parameters  
    - Temporal smoothing parameters
    - Zone constraint parameters
    """
    def __init__(self, config_path=None):
        # Si aucun chemin n'est fourni, utiliser le fichier config.json dans le même dossier que le script
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, 'config.json')
        
        # Utilisation de notre module de gestion des chemins
        self.project_root = get_project_root()
        
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
        
        # Dossiers avec gestion centralisée des chemins
        directories = config.get('directories', {})
        self.video_dir = self.project_root / 'labeled'
        self.pred_dir = self.project_root / directories.get('pred_dir', 'pred/4')
        self.mask_dir = self.project_root / 'masks'

class VideoProcessor:
    """
    Core video processing engine for vanishing point prediction.
    
    CURRENT IMPLEMENTATION:
    ----------------------
    This processor represents the "intermediate" version of the pipeline:
    - Basic optical flow calculation (default OpenCV parameters)
    - Vehicle segmentation for noise reduction
    - Two prediction methods: flow-based and colinearity-based
    - Simple frame-by-frame processing without advanced filtering
    
    PROCESSING PIPELINE:
    -------------------
    1. Frame preprocessing (grayscale conversion)
    2. Vehicle detection and mask creation  
    3. Optical flow calculation (cv2.calcOpticalFlowFarneback)
    4. Vanishing point estimation (method-dependent)
    5. Pixel-to-angle conversion
    
    LIMITATIONS vs RESEARCH FINDINGS:
    --------------------------------
    ❌ No advanced vector filtering (norm thresholds, colinearity filters)
    ❌ Suboptimal optical flow parameters (missing optimized pyr_scale=0.7, etc.)
    ❌ No temporal smoothing or consistency checks
    ❌ Missing adaptive handling for frames with poor vector quality
    
    PERFORMANCE EXPECTATION:
    -----------------------
    - Current: ~200-500% error (baseline performance)
    - Potential with improvements: ~44% error (research best)
    """
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
        """
        Process a single frame and return prediction results.
        
        CURRENT APPROACH:
        ----------------
        1. Convert to grayscale
        2. Apply vehicle segmentation (if enabled)
        3. Calculate optical flow using basic parameters
        4. Estimate vanishing point using selected method
        5. Convert to angles and store results
        
        MISSING OPTIMIZATIONS (from 4th arc research):
        ---------------------------------------------
        🔍 Advanced Vector Filtering:
           - No norm threshold filtering (research optimal: >13)
           - No colinearity filtering with center (research optimal: >0.96)
           - No distance-based weighting from image center
           
        🔍 Flow Quality Assessment:
           - No detection of frames with insufficient good vectors
           - No adaptive threshold adjustment per frame
           - No vector quality scoring
           
        🔍 Temporal Consistency:
           - No exponential moving average smoothing
           - No outlier detection between consecutive frames
           - No interpolation for problematic frames
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            None (results stored in self.results)
            
        NOTE: This method processes ~97.4% more vectors than the optimized version,
              as it doesn't apply the research-discovered filtering that removes
              low-quality vectors while maintaining prediction accuracy.
        """
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
        
        # Calculate optical flow (LIMITATION: using default parameters)
        # Research found optimal: pyr_scale=0.7, levels=7, winsize=23, iterations=2
        flow, _, _ = calculate_flow(self.prev_gray, gray, combined_mask)
        
        # Find separation points based on chosen method
        # LIMITATION: No advanced vector filtering applied here
        if self.config.prediction_method == "flow":
            x, y = find_separation_points(flow, combined_mask)
        elif self.config.prediction_method == "colinearity":
            # LIMITATION: Uses raw flow without filtering optimization
            vanishing_point = self.vp_estimator.estimate_vanishing_point(flow, visualize=False)
            x, y = map(int, vanishing_point)
        else:
            raise ValueError(f"Méthode de prédiction inconnue: {self.config.prediction_method}")
        
        # Convert to angles (basic conversion, no post-processing smoothing)
        yaw, pitch = pixels_to_angles(x, y)
        
        # Store results (LIMITATION: no temporal consistency checks)
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
            f.write(f"{result['yaw']:.6f} {result['pitch']:.6f}\n")

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
    from utilities.fix_predictions import fix_predictions
    print("\nAjustement des prédictions...")
    fix_predictions(pred_dir=config.pred_dir, gt_dir='calib_challenge/labeled')
    
    print("\nTraitement de toutes les vidéos terminé!")
    
    # DEVELOPMENT NOTE:
    # ================
    # This script represents an intermediate version of the vanishing point estimation pipeline.
    # Based on research findings, significant improvements are possible:
    #
    # 🎯 EXPECTED PERFORMANCE WITH CURRENT VERSION: ~200-500% error
    # 🎯 ACHIEVABLE WITH OPTIMIZATIONS: ~44% error (83.8% improvement demonstrated)
    #
    # Key improvements to implement in future versions:
    # 1. Advanced vector filtering (norm >13, colinearity >0.96)
    # 2. Optimized optical flow parameters (pyr_scale=0.7, levels=7, winsize=23)
    # 3. Temporal smoothing and consistency checks
    # 4. Adaptive handling for frames with poor vector quality
    #
    # See flow_filter.py and research documentation for implementation details.

if __name__ == "__main__":
    video_indices = [0]
    main(video_indices=video_indices) 