import cv2
import numpy as np
import os

# Imports absolus propres grâce à pip install -e .
from src.core.segmentation import VehicleDetector
from src.core.flow import calculate_flow, find_separation_points
from src.core.colinearity_optimization import VanishingPointEstimator
from src.utilities.ground_truth import read_ground_truth_pixels, read_ground_truth_angles
from src.core.rendering import (
    create_control_panel,
    visualize_flow_arrows,
    visualize_separation_points,
    create_flow_visualization,
    add_frame_info
)
from src.utilities.pixel_angle_converter import angles_to_pixels, pixels_to_angles
from src.utilities.paths import (
    get_project_root, 
    get_labeled_dir, 
    get_masks_dir, 
    get_pred_dir, 
    get_flows_dir, 
    ensure_dir_exists
)

# Constants
FOCAL_LENGTH = 910  # Camera focal length in pixels

class ManualSegmentationState:
    def __init__(self):
        self.drawing = False
        self.points = []
        self.current_frame = None
        self.manual_mask = None
        self.mask_path = None

class VisualizationState:
    def __init__(self, frame_width, frame_height, total_frames):
        self.paused = False
        self.viz_type = "1"  # Default visualization with arrows
        self.current_frame_number = 0
        self.frame = None
        self.flow = None
        self.combined_mask = None
        self.current_vehicle_mask = None
        self.prev_vehicle_mask = None
        self.prev_gray = None
        self.vanishing_point = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.total_frames = total_frames

def mouse_callback(event, x, y, flags, param):
    """Callback pour la segmentation manuelle"""
    state = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.points.append((x, y))
            # Dessiner la ligne en temps réel
            temp_frame = state.current_frame.copy()
            if len(state.points) > 1:
                for i in range(len(state.points) - 1):
                    cv2.line(temp_frame, state.points[i], state.points[i + 1], (0, 255, 0), 2)
            cv2.imshow('Manual Segmentation', temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        state.drawing = False
        if len(state.points) > 1:
            # Créer le masque à partir des points
            state.manual_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            points_array = np.array(state.points, dtype=np.int32)
            cv2.fillPoly(state.manual_mask, [points_array], 255)
            # Afficher le masque
            cv2.imshow('Mask', state.manual_mask)
            # Sauvegarder le masque
            cv2.imwrite(str(state.mask_path), state.manual_mask)
            print(f"Masque sauvegardé dans {state.mask_path}")

def get_video_frame_count(cap, video_path):
    """
    Compte le nombre réel de frames dans la vidéo en les lisant une par une.
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



def initialize_video(video_path):
    """
    Initialize video capture and get video properties.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (cap, frame_width, frame_height, fps, total_frames)
            - cap: VideoCapture object
            - frame_width: Width of video frames
            - frame_height: Height of video frames
            - fps: Frames per second
            - total_frames: Total number of frames in video
    """
    cap = cv2.VideoCapture(video_path)
    
    # Verify video opened correctly
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video at {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Count total frames
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
            - gt_pixels: List of ground truth pixel coordinates
            - gt_angles: List of ground truth angles
    """
    gt_pixels = read_ground_truth_pixels(video_index, FOCAL_LENGTH)
    gt_angles, _, _ = read_ground_truth_angles(video_index)
    return gt_pixels, gt_angles

def load_predictions(video_index, frame_width, frame_height, predictions_dir="3"):
    """
    Load predictions for the video and convert them to pixel coordinates.
    
    Args:
        video_index (int): Index of the video
        frame_width (int): Width of video frames
        frame_height (int): Height of video frames
        predictions_dir (str): Directory name for predictions (default: "3")
        
    Returns:
        list: List of predicted pixel coordinates
    """
    predictions_path = get_pred_dir(predictions_dir) / f"{video_index}.txt"
    pred_pixels = []
    
    if predictions_path.exists():
        with open(predictions_path, 'r') as f:
            for line in f:
                pitch, yaw = map(float, line.strip().split())
                x, y = angles_to_pixels(yaw, pitch, FOCAL_LENGTH, frame_width, frame_height)
                pred_pixels.append((x, y))
    
    return pred_pixels

def load_flows(video_index):
    """
    Load pre-computed flows for the video.
    
    Args:
        video_index (int): Index of the video
        
    Returns:
        numpy.ndarray: Array of flow fields
    """
    flows_path = get_flows_dir() / f'{video_index}.npy'
    if not flows_path.exists():
        raise ValueError(f"Flow file not found: {flows_path}")
    
    flows = np.load(flows_path)
    print(f"Loaded flows with shape: {flows.shape}")
    return flows

def setup_manual_segmentation(video_index, first_frame, frame_height, frame_width):
    """
    Set up manual segmentation interface and load/create mask.
    
    Args:
        video_index (int): Index of the video
        first_frame (numpy.ndarray): First frame of the video
        frame_height (int): Height of the video frames
        frame_width (int): Width of the video frames
        
    Returns:
        ManualSegmentationState: State object containing segmentation data
    """
    # Create masks directory if it doesn't exist and get mask path
    masks_dir = ensure_dir_exists(get_masks_dir())
    mask_path = masks_dir / f'{video_index}_mask.png'
    
    # Initialize state
    state = ManualSegmentationState()
    state.mask_path = mask_path
    
    # Create window for manual segmentation
    cv2.namedWindow('Manual Segmentation')
    cv2.setMouseCallback('Manual Segmentation', mouse_callback, state)
    
    # Check if mask already exists
    if mask_path.exists():
        print(f"Chargement du masque existant depuis {mask_path}")
        state.manual_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return state
    
    # Display first frame for manual segmentation
    state.current_frame = first_frame.copy()
    cv2.imshow('Manual Segmentation', state.current_frame)
    print("Dessinez votre segmentation sur la première frame. Appuyez sur 'q' quand vous avez terminé.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    return state

def position_video_at_frame(cap, start_frame):
    """
    Position the video at a specific frame.
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        start_frame (int): Frame number to position at
        
    Returns:
        tuple: (bool, numpy.ndarray)
            - bool: Success status
            - numpy.ndarray: First frame after positioning
    """
    print("Positionnement à la frame de départ...")
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

def handle_keyboard_input(key, state, detector, vp_estimator, gt_pixels, gt_angles, pred_pixels, show_vp=True):
    """
    Handle keyboard input and update state accordingly.
    
    Args:
        key (int): Key code from cv2.waitKey
        state (VisualizationState): Current visualization state
        detector (VehicleDetector): Vehicle detector instance
        vp_estimator (VanishingPointEstimator): Vanishing point estimator instance
        gt_pixels (list): Ground truth pixel coordinates
        gt_angles (list): Ground truth angles
        pred_pixels (list): Predicted pixel coordinates
        show_vp (bool): Whether to show vanishing point
    """
    if key == ord('q'):
        return False  # Signal to quit
    
    elif key == ord(' '):  # Space for pause/play
        state.paused = not state.paused
        create_control_panel(state.paused)
    
    elif key in [ord('1'), ord('2'), ord('3')]:  # Change visualization
        # Close debug windows if not in debug mode
        if state.viz_type == "3" and chr(key) != "3":
            cv2.destroyWindow('Previous Frame Masked')
            cv2.destroyWindow('Current Frame Masked')
        
        state.viz_type = chr(key)
        create_control_panel(state.paused)
        
        # Recreate visualization with current frame
        if state.frame is not None and state.flow is not None and state.combined_mask is not None:
            update_visualization(state, detector, vp_estimator, gt_pixels, gt_angles, pred_pixels, show_vp)
    
    return True  # Continue running

def process_frame(cap, state, detector, vp_estimator, gt_pixels, flows, show_vp=True):
    """
    Process a single frame: read, detect vehicles, load flow, estimate vanishing point.
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        state (VisualizationState): Current visualization state
        detector (VehicleDetector): Vehicle detector instance
        vp_estimator (VanishingPointEstimator): Vanishing point estimator instance
        gt_pixels (list): Ground truth pixel coordinates
        flows (numpy.ndarray): Pre-computed flow fields
        show_vp (bool): Whether to calculate and show vanishing point
    
    Returns:
        bool: True if frame was processed successfully, False if video ended
    """
    # Read frame
    ret, frame = cap.read()
    if not ret:
        return False
    
    # Update frame number
    state.current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    state.frame = frame
    
    # Detect vehicles and create mask
    state.current_vehicle_mask = detector.detect_vehicles(frame)
    state.current_vehicle_mask = detector.dilate_mask(state.current_vehicle_mask)
    
    # Combine masks
    state.combined_mask = detector.combine_masks(
        state.manual_mask, 
        state.current_vehicle_mask, 
        state.prev_vehicle_mask
    )
    
    # Load pre-computed flow
    if state.current_frame_number < len(flows):
        state.flow = flows[state.current_frame_number]
    else:
        print(f"Warning: No flow available for frame {state.current_frame_number}")
        return False
    
    # Calculate vanishing point only if needed
    if show_vp:
        # Get ground truth point for visualization if available
        ground_truth_point = None
        if state.current_frame_number < len(gt_pixels):
            ground_truth_point = gt_pixels[state.current_frame_number]
        
        # Estimate vanishing point
        state.vanishing_point = vp_estimator.estimate_vanishing_point(
            state.flow,
            visualize=False,
            ground_truth_point=ground_truth_point
        )
    else:
        # Set a dummy vanishing point to avoid errors
        state.vanishing_point = (0, 0)
    
    # Update state for next iteration
    state.prev_vehicle_mask = state.current_vehicle_mask
    
    return True

def update_visualization(state, detector, vp_estimator, gt_pixels, gt_angles, pred_pixels, show_vp=True):
    """
    Update visualization based on current state and visualization type.
    
    Args:
        state (VisualizationState): Current visualization state
        detector (VehicleDetector): Vehicle detector instance
        vp_estimator (VanishingPointEstimator): Vanishing point estimator instance
        gt_pixels (list): Ground truth pixel coordinates
        gt_angles (list): Ground truth angles
        pred_pixels (list): Predicted pixel coordinates
        show_vp (bool): Whether to show vanishing point
    """
    if state.viz_type == "1":
        output = visualize_flow_arrows(state.frame, state.flow, state.combined_mask)
        
        # Draw the legal zone for vanishing points (only if vp_estimator exists)
        if vp_estimator is not None:
            vp_estimator.draw_vanishing_point_zone(output)
        
        # Add ground truth point if available
        if state.current_frame_number < len(gt_pixels):
            gt_x, gt_y = gt_pixels[state.current_frame_number]
            gt_yaw, gt_pitch = gt_angles[state.current_frame_number]
            
            # Draw ground truth point
            cv2.circle(output, (gt_x, gt_y), 5, (0, 0, 255), -1)
            cv2.putText(output, "LABEL", (gt_x + 10, gt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add debug info
            center_x, center_y = state.frame_width // 2, state.frame_height // 2
            cv2.circle(output, (center_x, center_y), 5, (255, 0, 0), -1)
            text = f"GT: yaw={np.degrees(gt_yaw):.2f}°, pitch={np.degrees(gt_pitch):.2f}°, (x={gt_x}, y={gt_y})"
            cv2.putText(output, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add predictions if available
        if state.current_frame_number < len(pred_pixels):
            pred_x, pred_y = pred_pixels[state.current_frame_number]
            cv2.circle(output, (int(pred_x), int(pred_y)), 5, (0, 255, 255), -1)
            cv2.putText(output, "PRED", (int(pred_x) + 10, int(pred_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add vanishing point if enabled
        if show_vp:
            vp_x, vp_y = map(int, state.vanishing_point)
            cv2.circle(output, (vp_x, vp_y), 5, (0, 255, 0), -1)
            cv2.putText(output, "VP", (vp_x + 10, vp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    elif state.viz_type == "2":
        best_x, best_y = find_separation_points(state.flow, state.combined_mask)
        output = visualize_separation_points(state.frame, best_x, best_y)
        
        # Draw the legal zone for vanishing points (only if vp_estimator exists)
        if vp_estimator is not None:
            vp_estimator.draw_vanishing_point_zone(output)
        
        # Add ground truth if available
        if state.current_frame_number < len(gt_pixels):
            gt_x, gt_y = gt_pixels[state.current_frame_number]
            cv2.circle(output, (gt_x, gt_y), 5, (0, 0, 255), -1)
            cv2.putText(output, "GT", (gt_x + 10, gt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    else:  # viz_type == "3"
        output = create_flow_visualization(state.flow, state.combined_mask, state.frame)
        # Draw the legal zone for vanishing points (only if vp_estimator exists)
        if vp_estimator is not None:
            vp_estimator.draw_vanishing_point_zone(output)
    
    # Add frame info
    output = add_frame_info(
        output, 
        state.viz_type, 
        state.current_frame_number, 
        state.total_frames, 
        state.paused
    )
    
    # Show masks and visualization
    cv2.imshow('Vehicle Mask', state.current_vehicle_mask)
    cv2.imshow('Combined Mask', state.combined_mask)
    cv2.imshow('Visualization', output)

def main(video_index, start_frame, predictions_dir="3", show_vp=True):
    # Get video path
    video_path = get_labeled_dir() / f'{video_index}.hevc'
    
    # Initialize video
    try:
        cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    except ValueError as e:
        print(e)
        return
    
    print(f"Démarrage à la frame: {start_frame}")
    print(f"Utilisation des prédictions depuis: pred/{predictions_dir}/")
    print(f"Affichage du vanishing point: {'Activé' if show_vp else 'Désactivé'}")
    
    # Vérifier que la frame de départ est valide
    if start_frame >= total_frames:
        print(f"Erreur: La frame de départ {start_frame} est supérieure au nombre total de frames {total_frames}")
        return
    
    # Load ground truth, predictions and flows
    gt_pixels, gt_angles = load_ground_truth(video_index, frame_width, frame_height)
    pred_pixels = load_predictions(video_index, frame_width, frame_height, predictions_dir)
    flows = load_flows(video_index)
    
    # Position video at start frame
    success, first_frame = position_video_at_frame(cap, start_frame)
    if not success:
        return
    
    # Initialize all components
    # 1. Manual segmentation setup
    seg_state = setup_manual_segmentation(video_index, first_frame, frame_height, frame_width)
    
    # 2. Initialize visualization state
    viz_state = VisualizationState(frame_width, frame_height, total_frames)
    viz_state.current_frame_number = start_frame
    viz_state.manual_mask = seg_state.manual_mask
    
    # 3. Processing components
    detector = VehicleDetector()
    # Only create VanishingPointEstimator if needed
    if show_vp:
        vp_estimator = VanishingPointEstimator(frame_width, frame_height, FOCAL_LENGTH, use_max_distance=False, use_reoptimization=False)
    else:
        vp_estimator = None
    
    # 4. Flow tracking state
    viz_state.prev_vehicle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    
    # 5. Setup visualization windows
    cv2.namedWindow('Visualization')
    cv2.setWindowProperty('Visualization', cv2.WND_PROP_TOPMOST, 1)
    create_control_panel(viz_state.paused)
    
    # Main loop
    running = True
    while running:
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        running = handle_keyboard_input(
            key, 
            viz_state, 
            detector, 
            vp_estimator, 
            gt_pixels, 
            gt_angles, 
            pred_pixels,
            show_vp
        )
        
        if not running:
            break
        
        # Process frame if not paused
        if not viz_state.paused:
            success = process_frame(cap, viz_state, detector, vp_estimator, gt_pixels, flows, show_vp)
            if not success:
                break
        
        # Update visualization
        if viz_state.frame is not None:
            update_visualization(
                viz_state, 
                detector, 
                vp_estimator, 
                gt_pixels, 
                gt_angles, 
                pred_pixels,
                show_vp
            )
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_index = 4
    start_frame = 220
    predictions_dir = "5"  # Change this to use different prediction directories (e.g., "3", "5", "vanilla", etc.)
    show_vp = False  # Set to False to disable vanishing point calculation and display
    main(video_index, start_frame, predictions_dir, show_vp)