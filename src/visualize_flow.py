import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_flow(flow, frame_idx, video_idx=0, stride=20):
    """
    Visualize optical flow field with arrows.
    
    Args:
        flow: Flow field array (H, W, 2)
        frame_idx: Index of the frame to visualize
        video_idx: Index of the video
        stride: Spacing between arrows (higher = fewer arrows)
    """
    # Load the original frame for background
    video_path = os.path.join('calib_challenge/labeled', f'{video_idx}.hevc')
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Impossible de lire la frame {frame_idx}")
        return
    
    # Convert frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original frame
    plt.imshow(frame_rgb)
    
    # Get flow field for the specified frame
    flow_field = flow[frame_idx]
    
    # Create grid for arrows
    h, w = flow_field.shape[:2]
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Get flow vectors at grid points
    fx = flow_field[y, x, 0]
    fy = flow_field[y, x, 1]
    
    # Filter out zero or near-zero flow values
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > 0.1  # Threshold for minimum flow magnitude
    
    # Plot arrows only where there is significant flow
    plt.quiver(x[mask], y[mask], fx[mask], fy[mask], 
              color='r', angles='xy', scale_units='xy', scale=1)
    
    plt.title(f'Flow Field - Video {video_idx}, Frame {frame_idx}')
    plt.axis('off')
    plt.show()

def main():
    # Load flows
    flow_file = os.path.join('calib_challenge/flows', '0.npy')
    if not os.path.exists(flow_file):
        print(f"Fichier de flows non trouvé: {flow_file}")
        return
    
    flows = np.load(flow_file)
    print(f"Shape des flows: {flows.shape}")
    
    # Visualize frame 6
    frame_idx = 285
    visualize_flow(flows, frame_idx)

if __name__ == "__main__":
    main() 