"""
Utility functions for loading video frames.
"""

import cv2
from pathlib import Path
from typing import Tuple, Optional

def read_frame_at_index(video_path: Path, target_frame_idx: int) -> Tuple[bool, Optional[cv2.Mat]]:
    """
    Read a specific frame from a video by reading sequentially until reaching the target frame.
    This approach is more reliable for HEVC encoded videos where direct frame seeking might not work.
    
    Args:
        video_path: Path to the video file
        target_frame_idx: Index of the frame to read (0-based)
        
    Returns:
        Tuple of (success, frame) where:
        - success is a boolean indicating if the frame was read successfully
        - frame is the read frame as a cv2.Mat object, or None if reading failed
    """
    cap = cv2.VideoCapture(str(video_path))
    current_frame = 0
    
    while current_frame <= target_frame_idx:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return False, None
        if current_frame == target_frame_idx:
            cap.release()
            return True, frame
        current_frame += 1
    
    cap.release()
    return False, None

def read_frame_rgb(video_path: Path, target_frame_idx: int) -> Tuple[bool, Optional[cv2.Mat]]:
    """
    Read a specific frame from a video and convert it to RGB format.
    
    Args:
        video_path: Path to the video file
        target_frame_idx: Index of the frame to read (0-based)
        
    Returns:
        Tuple of (success, frame) where:
        - success is a boolean indicating if the frame was read successfully
        - frame is the read frame in RGB format as a cv2.Mat object, or None if reading failed
    """
    success, frame = read_frame_at_index(video_path, target_frame_idx)
    if not success:
        return False, None
    
    # Convert from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return True, frame_rgb 