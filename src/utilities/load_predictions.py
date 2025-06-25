from typing import Union

from src.utilities.paths import get_pred_dir
from src.utilities.pixel_angle_converter import angles_to_pixels


def load_predictions(video_index, frame_index=None, predictions_dir="5") -> Union[tuple, list]:
    """
    Load predictions for the video and convert them to pixel coordinates.
    
    Args:
        video_index (int): Index of the video
        frame_index (int): Index of the frame
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
                x, y = angles_to_pixels(pitch, yaw)
                pred_pixels.append((x, y))
    if frame_index is not None:
        return pred_pixels[frame_index]
    else:
        return pred_pixels