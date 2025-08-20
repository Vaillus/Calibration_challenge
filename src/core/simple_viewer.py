import cv2
import numpy as np

# Imports absolus propres grâce à pip install -e .
from src.utilities.pixel_angle_converter import angles_to_pixels
from src.utilities.paths import (
    get_labeled_dir, 
    get_pred_dir, 
    get_unlabeled_dir
)

# Constants
FOCAL_LENGTH = 910  # Camera focal length in pixels

class SimpleVisualizationState:
    def __init__(self, frame_width, frame_height, total_frames):
        self.paused = False
        self.current_frame_number = 0
        self.frame = None
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.total_frames = total_frames

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
    max_frames = 2000
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

def load_predictions(video_index, frame_width, frame_height, predictions_dir="3"):
    """
    Load predictions for the video and convert them to pixel coordinates.
    """
    predictions_path = get_pred_dir(predictions_dir) / f"{video_index}.txt"
    pred_pixels = []
    
    if predictions_path.exists():
        with open(predictions_path, 'r') as f:
            for line in f:
                pitch, yaw = map(float, line.strip().split())
                x, y = angles_to_pixels(pitch, yaw, FOCAL_LENGTH, frame_width, frame_height)
                pred_pixels.append((x, y))
    
    return pred_pixels

def position_video_at_frame(cap, start_frame):
    """
    Position the video at a specific frame.
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

def handle_keyboard_input(key, state):
    """
    Handle keyboard input and update state accordingly.
    """
    if key == ord('q'):
        return False  # Signal to quit
    
    elif key == ord(' '):  # Space for pause/play
        state.paused = not state.paused
        status = "PAUSE" if state.paused else "PLAY"
        print(f"Status: {status}")
    
    return True  # Continue running

def process_frame(cap, state):
    """
    Process a single frame: read frame.
    """
    # Read frame
    ret, frame = cap.read()
    if not ret:
        return False
    
    # Update frame number and state
    state.current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    state.frame = frame
    
    return True

def update_visualization(state, pred_pixels):
    """
    Update visualization with center point and predictions.
    """
    if state.frame is None:
        return
    
    # Create output frame
    output = state.frame.copy()
    
    # Draw center point
    center_x, center_y = state.frame_width // 2, state.frame_height // 2
    cv2.circle(output, (center_x, center_y), 8, (255, 0, 0), -1)
    cv2.putText(output, "CENTER", (center_x + 15, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Add predictions if available
    if state.current_frame_number < len(pred_pixels):
        pred_x, pred_y = pred_pixels[state.current_frame_number]
        cv2.circle(output, (int(pred_x), int(pred_y)), 8, (0, 255, 255), -1)
        cv2.putText(output, "PRED", (int(pred_x) + 15, int(pred_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Calculate and display distance
        distance = np.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
        cv2.putText(output, f"Distance: {distance:.1f}px", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add frame info
    status = "PAUSED" if state.paused else "PLAYING"
    frame_info = f"Frame: {state.current_frame_number}/{state.total_frames} - {status}"
    cv2.putText(output, frame_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add controls info
    cv2.putText(output, "Controls: SPACE=pause/play, Q=quit", (30, state.frame_height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show visualization
    cv2.imshow('Simple Viewer', output)

def main(video_index, start_frame=0, predictions_dir="3"):
    """
    Main function for simple video viewer.
    """
    # Get video path
    if video_index < 5:
        video_path = get_labeled_dir() / f'{video_index}.hevc'
    else:
        video_path = get_unlabeled_dir() / f'{video_index}.hevc'
    
    # Initialize video
    try:
        cap, frame_width, frame_height, fps, total_frames = initialize_video(video_path)
    except ValueError as e:
        print(e)
        return
    
    print(f"Démarrage à la frame: {start_frame}")
    print(f"Utilisation des prédictions depuis: pred/{predictions_dir}/")
    
    # Vérifier que la frame de départ est valide
    if start_frame >= total_frames:
        print(f"Erreur: La frame de départ {start_frame} est supérieure au nombre total de frames {total_frames}")
        return
    
    # Load predictions
    pred_pixels = load_predictions(video_index, frame_width, frame_height, predictions_dir)
    print(f"Chargé {len(pred_pixels)} prédictions")
    
    # Position video at start frame
    success, first_frame = position_video_at_frame(cap, start_frame)
    if not success:
        return
    
    # Initialize visualization state
    state = SimpleVisualizationState(frame_width, frame_height, total_frames)
    state.current_frame_number = start_frame
    state.frame = first_frame
    
    # Setup visualization window
    cv2.namedWindow('Simple Viewer', cv2.WINDOW_AUTOSIZE)
    
    print("Contrôles:")
    print("- ESPACE: pause/lecture")
    print("- Q: quitter")
    
    # Main loop
    running = True
    while running:
        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF
        running = handle_keyboard_input(key, state)
        
        if not running:
            break
        
        # Process frame if not paused
        if not state.paused:
            success = process_frame(cap, state)
            if not success:
                print("Fin de la vidéo")
                break
        
        # Update visualization
        update_visualization(state, pred_pixels)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_index = 7
    start_frame = 0
    predictions_dir = "5_7_smoothed"
    main(video_index, start_frame, predictions_dir)