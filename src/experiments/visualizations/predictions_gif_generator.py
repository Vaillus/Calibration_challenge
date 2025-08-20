import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
from tqdm import tqdm

# Imports absolus propres grâce à pip install -e .
from src.utilities.load_ground_truth import read_ground_truth_pixels, read_ground_truth_angles
from src.utilities.pixel_angle_converter import angles_to_pixels
from src.utilities.paths import (
    get_project_root, 
    get_labeled_dir, 
    get_pred_dir
)

# Constants
FOCAL_LENGTH = 910  # Camera focal length in pixels

def initialize_video(video_path):
    """
    Initialize video capture and get video properties.
    Same approach as interactive_viewer.py
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (cap, frame_width, frame_height, fps, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Verify video opened correctly
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Count total frames manually (more reliable for HEVC)
    total_frames, cap = get_video_frame_count(cap, video_path)
    
    print(f"Video properties:")
    print(f"- Resolution: {frame_width}x{frame_height}")
    print(f"- FPS: {fps}")
    print(f"- Total frames: {total_frames}")
    
    return cap, frame_width, frame_height, fps, total_frames

def load_ground_truth(video_index, frame_width, frame_height):
    """
    Load ground truth data for the video.
    
    Args:
        video_index (int): Index of the video
        frame_width (int): Width of video frames
        frame_height (int): Height of video frames
        
    Returns:
        tuple: (gt_pixels, gt_angles)
    """
    gt_pixels = read_ground_truth_pixels(video_index, FOCAL_LENGTH)
    gt_angles = read_ground_truth_angles(video_index)
    return gt_pixels, gt_angles

def load_predictions(video_index, frame_width, frame_height, predictions_dir):
    """
    Load predictions for the video and convert them to pixel coordinates.
    
    Args:
        video_index (int): Index of the video
        frame_width (int): Width of video frames
        frame_height (int): Height of video frames
        predictions_dir (str): Directory name for predictions
        
    Returns:
        list: List of predicted pixel coordinates
    """
    predictions_path = get_pred_dir(predictions_dir) / f"{video_index}.txt"
    pred_pixels = []
    
    if predictions_path.exists():
        with open(predictions_path, 'r') as f:
            for line in f:
                pitch, yaw = map(float, line.strip().split())
                x, y = angles_to_pixels(pitch, yaw, FOCAL_LENGTH, frame_width, frame_height)
                pred_pixels.append((x, y))
        print(f"Loaded {len(pred_pixels)} predictions from {predictions_path}")
    else:
        print(f"Warning: Predictions file not found: {predictions_path}")
    
    return pred_pixels

def get_video_frame_count(cap, video_path):
    """
    Compte le nombre réel de frames dans la vidéo en les lisant une par une.
    Same as interactive_viewer.py
    """
    # Sauvegarder la position actuelle
    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    # Aller au début
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Compter les frames
    frame_count = 0
    max_frames = 2000  # Augmentation de la limite pour s'assurer de compter toutes les frames
    while frame_count < max_frames:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    
    print(f"Comptage manuel: {frame_count} frames")
    
    # Réinitialiser complètement la vidéo
    cap.release()
    cap = cv2.VideoCapture(video_path)
    
    # Restaurer la position
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
    
    return frame_count, cap

def position_video_at_frame(cap, start_frame):
    """
    Position the video at a specific frame.
    Same approach as interactive_viewer.py - sequential reading is more reliable for HEVC
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        start_frame (int): Frame number to position at
        
    Returns:
        tuple: (bool, numpy.ndarray)
            - bool: Success status
            - numpy.ndarray: First frame after positioning
    """
    print(f"Positionnement à la frame {start_frame}...")
    for _ in range(start_frame):
        ret, _ = cap.read()
        if not ret:
            print("Erreur: Impossible d'atteindre la frame de départ")
            return False, None
    
    # Read the first frame after positioning
    ret, first_frame = cap.read()
    if not ret:
        print("Erreur: Impossible de lire la première frame")
        return False, None
        
    return True, first_frame

def visualize_frame_with_predictions(frame, frame_number, gt_pixels, pred_pixels, frame_width, frame_height, point_size_factor=1.0, legend_scale=0.5):
    """
    Visualize a frame with ground truth, predictions, and center point.
    Amélioration avec des couleurs plus contrastées et des bordures.
    
    Args:
        frame (numpy.ndarray): Video frame
        frame_number (int): Current frame number
        gt_pixels (list): Ground truth pixel coordinates
        pred_pixels (list): Predicted pixel coordinates
        frame_width (int): Width of the frame
        frame_height (int): Height of the frame
        point_size_factor (float): Factor to scale point sizes (default: 1.0)
        legend_scale (float): Factor to scale legend size (default: 0.5)
        
    Returns:
        numpy.ndarray: Annotated frame
    """
    output = frame.copy()
    
    # Paramètres visuels améliorés avec facteur de taille
    base_point_radius = 12
    point_radius = int(base_point_radius * point_size_factor)
    border_thickness = max(1, int(3 * point_size_factor))
    text_thickness = max(1, int(2 * point_size_factor))
    font_scale = 0.6 * point_size_factor
    
    # Draw center point - BLANC avec bordure noire
    center_x, center_y = frame_width // 2, frame_height // 2
    # Bordure noire
    cv2.circle(output, (center_x, center_y), point_radius + border_thickness, (0, 0, 0), -1)
    # Point blanc
    cv2.circle(output, (center_x, center_y), point_radius, (255, 255, 255), -1)
    
    # Draw ground truth point - BLANC avec bordure noire et croix noire
    if frame_number < len(gt_pixels):
        gt_x, gt_y = gt_pixels[frame_number]
        # Bordure noire
        cv2.circle(output, (gt_x, gt_y), point_radius + border_thickness, (0, 0, 0), -1)
        # Point blanc
        cv2.circle(output, (gt_x, gt_y), point_radius, (255, 255, 255), -1)
        # Croix noire au centre pour distinguer du CENTER
        cross_size = int(6 * point_size_factor)
        cross_thickness = max(1, int(2 * point_size_factor))
        cv2.line(output, (gt_x - cross_size, gt_y), (gt_x + cross_size, gt_y), (0, 0, 0), cross_thickness)
        cv2.line(output, (gt_x, gt_y - cross_size), (gt_x, gt_y + cross_size), (0, 0, 0), cross_thickness)
    
    # Draw prediction - BLANC avec bordure noire et triangle noir
    if frame_number < len(pred_pixels):
        pred_x, pred_y = int(pred_pixels[frame_number][0]), int(pred_pixels[frame_number][1])
        # Bordure noire
        cv2.circle(output, (pred_x, pred_y), point_radius + border_thickness, (0, 0, 0), -1)
        # Point blanc
        cv2.circle(output, (pred_x, pred_y), point_radius, (255, 255, 255), -1)
        # Triangle noir au centre pour différencier
        triangle_size = int(5 * point_size_factor)
        triangle_points = np.array([
            [pred_x, pred_y - triangle_size],
            [pred_x - triangle_size, pred_y + int(triangle_size * 0.8)],
            [pred_x + triangle_size, pred_y + int(triangle_size * 0.8)]
        ], np.int32)
        cv2.fillPoly(output, [triangle_points], (0, 0, 0))
    
    # Draw legend in top-right corner (controlled by legend_scale parameter)
    base_legend_width = 300
    base_legend_y = 60
    base_legend_spacing = 70
    base_legend_point_size = 16
    base_legend_border_size = 4
    base_legend_font_scale = 1.0
    base_legend_text_thickness = 2
    
    legend_x = frame_width - int(base_legend_width * legend_scale)
    legend_y = int(base_legend_y * legend_scale)
    legend_spacing = int(base_legend_spacing * legend_scale)
    legend_point_size = int(base_legend_point_size * legend_scale)
    legend_border_size = max(1, int(base_legend_border_size * legend_scale))
    legend_font_scale = base_legend_font_scale * legend_scale
    legend_text_thickness = max(1, int(base_legend_text_thickness * legend_scale))
    
    # Legend background (semi-transparent black rectangle)
    legend_bg_x1 = legend_x - int(40 * legend_scale)
    legend_bg_y1 = legend_y - int(40 * legend_scale)
    legend_bg_x2 = frame_width - int(20 * legend_scale)
    legend_bg_y2 = legend_y + int(2.5 * legend_spacing)
    cv2.rectangle(output, (legend_bg_x1, legend_bg_y1), (legend_bg_x2, legend_bg_y2), (0, 0, 0), -1)
    
    # Center point legend
    cv2.circle(output, (legend_x, legend_y), legend_point_size + legend_border_size, (0, 0, 0), -1)
    cv2.circle(output, (legend_x, legend_y), legend_point_size, (255, 255, 255), -1)
    cv2.putText(output, "Centre image", (legend_x + int(30 * legend_scale), legend_y + int(10 * legend_scale)), 
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 255, 255), legend_text_thickness)
    
    # Label point legend
    label_y = legend_y + legend_spacing
    cv2.circle(output, (legend_x, label_y), legend_point_size + legend_border_size, (0, 0, 0), -1)
    cv2.circle(output, (legend_x, label_y), legend_point_size, (255, 255, 255), -1)
    # Small cross (black on white)
    small_cross = max(1, int(6 * legend_scale))
    cross_thickness = max(1, int(2 * legend_scale))
    cv2.line(output, (legend_x - small_cross, label_y), (legend_x + small_cross, label_y), (0, 0, 0), cross_thickness)
    cv2.line(output, (legend_x, label_y - small_cross), (legend_x, label_y + small_cross), (0, 0, 0), cross_thickness)
    cv2.putText(output, "Label", (legend_x + int(30 * legend_scale), label_y + int(10 * legend_scale)), 
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 255, 255), legend_text_thickness)
    
    # Prediction point legend
    pred_y = legend_y + 2 * legend_spacing
    cv2.circle(output, (legend_x, pred_y), legend_point_size + legend_border_size, (0, 0, 0), -1)
    cv2.circle(output, (legend_x, pred_y), legend_point_size, (255, 255, 255), -1)
    # Small triangle
    small_triangle_size = max(1, int(4 * legend_scale))
    small_triangle_points = np.array([
        [legend_x, pred_y - small_triangle_size],
        [legend_x - small_triangle_size, pred_y + int(small_triangle_size * 0.8)],
        [legend_x + small_triangle_size, pred_y + int(small_triangle_size * 0.8)]
    ], np.int32)
    cv2.fillPoly(output, [small_triangle_points], (0, 0, 0))
    cv2.putText(output, "Prediction", (legend_x + int(30 * legend_scale), pred_y + int(10 * legend_scale)), 
                cv2.FONT_HERSHEY_SIMPLEX, legend_font_scale, (255, 255, 255), legend_text_thickness)
    
    return output


def create_predictions_gif_improved(video_index=3, start_frame=630, end_frame=680, predictions_dir="3", point_size_factor=1.0, legend_scale=0.5):
    """
    Version améliorée avec de meilleurs paramètres GIF pour préserver les couleurs.
    
    Args:
        video_index (int): Index of the video (default: 3)
        start_frame (int): Starting frame number (default: 630)
        end_frame (int): Ending frame number (default: 680)
        predictions_dir (str): Directory name for predictions (default: "3")
        point_size_factor (float): Factor to scale point sizes (default: 1.0)
        legend_scale (float): Factor to scale legend size (default: 0.5)
    """
    print(f"Creating improved gif for video {video_index}, frames {start_frame} to {end_frame}")
    print(f"Using predictions from directory: {predictions_dir}")
    
    # Get video path
    video_path = get_labeled_dir() / f'{video_index}.hevc'
    
    # Initialize video
    try:
        cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    except ValueError as e:
        print(e)
        return
    
    # Validate frame range
    if start_frame >= end_frame:
        print(f"Error: Start frame ({start_frame}) must be less than end frame ({end_frame})")
        return
    
    # Load ground truth and predictions
    gt_pixels, gt_angles = load_ground_truth(video_index, frame_width, frame_height)
    pred_pixels = load_predictions(video_index, frame_width, frame_height, predictions_dir)
    
    print(f"Generating improved gif for {end_frame - start_frame + 1} frames...")
    
    # Position video at start frame
    success, first_frame = position_video_at_frame(cap, start_frame)
    if not success:
        return
    
    # Generate all frames for the gif
    frames = []
    skipped_frames = 0
    
    # Process first frame
    annotated_frame = visualize_frame_with_predictions(
        first_frame, start_frame, gt_pixels, pred_pixels, frame_width, frame_height, point_size_factor, legend_scale
    )
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frames.append(rgb_frame)
    
    # Process remaining frames sequentially
    current_frame_idx = start_frame + 1
    with tqdm(total=end_frame - start_frame, desc="Processing frames", initial=1) as pbar:
        while current_frame_idx <= end_frame:
            ret, current_frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {current_frame_idx}")
                skipped_frames += 1
                current_frame_idx += 1
                pbar.update(1)
                continue
            
            # Create annotated frame
            annotated_frame = visualize_frame_with_predictions(
                current_frame, current_frame_idx, gt_pixels, pred_pixels, frame_width, frame_height, point_size_factor, legend_scale
            )
            
            # Convert BGR to RGB for imageio
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            
            current_frame_idx += 1
            pbar.update(1)
    
    # Create output path
    output_dir = get_project_root() / "data" / "outputs"
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"predictions_video_{video_index}_frames_{start_frame}_{end_frame}_pred_{predictions_dir}_improved.gif"
    
    # Generate gif with paramètres optimisés pour la qualité
    if frames:
        print(f"Creating improved gif with {len(frames)} frames...")
        if skipped_frames > 0:
            print(f"Note: {skipped_frames} frames were skipped due to HEVC decoding errors")
        
        # Paramètres optimisés pour préserver les couleurs vives
        imageio.mimsave(
            gif_path, 
            frames, 
            duration=0.12,  # Légèrement plus lent pour mieux voir
            loop=0,
            quantizer='nq',  # Meilleure quantification des couleurs
            palettesize=256,  # Palette complète
            optimize=False,  # Désactiver l'optimisation qui peut ternir les couleurs
            subrectangles=True  # Optimisation sans perte de qualité
        )
        print(f"Improved gif saved to: {gif_path}")
    else:
        print("Error: No frames were processed successfully")
    
    # Cleanup
    cap.release()

if __name__ == "__main__":
    # Test with default parameters
    video_index = 4
    start_frame = 630
    end_frame = 680
    predictions_dir = "3_new"  # Change this to use different prediction directories (e.g., "3", "5", "vanilla", etc.)
    point_size_factor = 0.75  # Adjust this to change point sizes (e.g., 0.5 for smaller, 1.5 for larger)
    legend_scale = 0.5  # Adjust this to change legend size (e.g., 0.25 for smaller, 1.0 for larger)
    create_predictions_gif_improved(video_index, start_frame, end_frame, predictions_dir, point_size_factor, legend_scale) 