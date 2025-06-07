import os
from typing import List, Tuple
import cv2

# Imports absolus propres grâce à pip install -e .
from src.utilities.pixel_angle_converter import angles_to_pixels, pixels_to_angles
from src.utilities.paths import get_labeled_dir

DEFAULT_FOCAL_LENGTH = 910

def read_ground_truth_angles(video_index: int) -> Tuple[List[Tuple[float, float]], int, int]:
    """
    Reads ground truth angles for a given video.
    
    Args:
        video_index: Index of the video
        
    Returns:
        Tuple containing:
        - List of tuples (yaw, pitch) in radians
        - Video width in pixels
        - Video height in pixels
    """
    labeled_dir = get_labeled_dir()
    gt_path = labeled_dir / f"{video_index}.txt"
    video_path = labeled_dir / f"{video_index}.hevc"
    
    # Get video dimensions using OpenCV
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    #print(f"image_width: {width}, image_height: {height}")
    
    gt_angles = []
    with open(gt_path, 'r') as f:
        for line in f:
            pitch, yaw = map(float, line.strip().split())  # First number = pitch, second = yaw
            gt_angles.append((yaw, pitch))  # Stored in the order expected by angles_to_pixels
            
    return gt_angles, width, height

def read_ground_truth_pixels(video_index: int, focal_length: float=DEFAULT_FOCAL_LENGTH) -> List[Tuple[int, int]]:
    """
    Reads ground truth angles and converts them to pixels for a given video.
    
    Args:
        video_index: Index of the video
        focal_length: Focal length in pixels
        
    Returns:
        List of tuples (x, y) in pixels
    """
    gt_angles, image_width, image_height = read_ground_truth_angles(video_index)
    gt_pixels = []
    
    for yaw, pitch in gt_angles:
        x, y = angles_to_pixels(yaw, pitch, focal_length, image_width, image_height)
        gt_pixels.append((x, y))
        
    return gt_pixels

def get_frame_angles(video_index: int, frame_index: int) -> Tuple[float, float]:
    """
    Gets ground truth angles for a specific frame.
    
    Args:
        video_index: Index of the video
        frame_index: Index of the frame
        
    Returns:
        Tuple (yaw, pitch) in radians
    """
    gt_angles, _, _ = read_ground_truth_angles(video_index)
    if frame_index >= len(gt_angles):
        raise IndexError(f"Frame index {frame_index} out of range (max: {len(gt_angles)-1})")
    return gt_angles[frame_index]

def get_frame_pixel(video_index: int, frame_index: int, focal_length: float=DEFAULT_FOCAL_LENGTH) -> Tuple[int, int]:
    """
    Gets ground truth pixel coordinates for a specific frame.
    
    Args:
        video_index: Index of the video
        frame_index: Index of the frame
        focal_length: Focal length in pixels
    Returns:
        Tuple (x, y) in pixels
    """
    x, y = read_ground_truth_pixels(video_index, focal_length)[frame_index]
    return x, y

def main():
    """
    Main function to test get_frame_pixels.
    """
    # Test parameters
    video_index = 0  # Example with the first video
    frame_index = 5  # Example with the 6th frame
    focal_length = DEFAULT_FOCAL_LENGTH  # Example focal length
    
    try:
        # Test get_frame_pixels
        x, y = get_frame_pixels(video_index, frame_index, focal_length)
        print(f"Pixel coordinates for video {video_index}, frame {frame_index}:")
        print(f"x = {x}, y = {y}")
        
        # Test get_frame_angles for comparison
        yaw, pitch = get_frame_angles(video_index, frame_index)
        print(f"Corresponding angles: yaw = {yaw} rad, pitch = {pitch} rad")
        
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    main()
