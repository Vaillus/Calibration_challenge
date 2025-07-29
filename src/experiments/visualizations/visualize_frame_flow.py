#!/usr/bin/env python3
"""
Visualisation simple d'une frame avec vecteurs de flux optique non filtrés.
Affiche l'image en fond avec tous les vecteurs de flow sans filtrage.

Usage:
    python visualize_raw_flow.py
"""

import numpy as np
import matplotlib.pyplot as plt

from src.utilities.paths import get_labeled_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.utilities.load_flows import load_flows


def visualize_raw_flow(video_idx: int = 3, frame_idx: int = 101, stride: int = 20):
    """
    Visualise une frame avec le champ de vecteurs de flow non filtré.
    
    Args:
        video_idx: Index de la vidéo
        frame_idx: Index de la frame
        stride: Espacement entre les flèches (plus élevé = moins de flèches)
    """
    print(f"📂 Chargement frame {frame_idx} - vidéo {video_idx}...")
    
    # Load RGB frame
    video_path = get_labeled_dir() / f'{video_idx}.hevc'
    _, frame_rgb = read_frame_rgb(video_path, frame_idx)
    
    # Load raw flow data (non filtré)
    flow_data = load_flows(video_idx, start_frame=frame_idx-1, end_frame=frame_idx-1)
    flow_data = flow_data[0]  # Remove batch dimension: (1, H, W, 2) -> (H, W, 2)
    
    print(f"✅ Données chargées - Flow shape: {flow_data.shape}")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display the frame as background
    plt.imshow(frame_rgb)
    
    # Create grid for arrows
    h, w = flow_data.shape[:2]
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Get flow vectors at grid points
    fx = flow_data[y, x, 0]
    fy = flow_data[y, x, 1]
    
    # Plot ALL flow vectors (no filtering)
    plt.quiver(x, y, fx, fy, 
              color='red', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.8)
    
    # plt.title(f'Champ de Flux Non Filtré - Vidéo {video_idx}, Frame {frame_idx}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualisation terminée")


def main():
    """Fonction principale avec paramètres par défaut."""
    print("🎯 VISUALISATION FLUX NON FILTRÉ")
    print("=" * 40)
    
    # Paramètres par défaut
    video_idx = 4
    frame_idx = 380
    stride = 20
    
    print(f"📊 Configuration: Vidéo {video_idx}, Frame {frame_idx}, Stride {stride}")
    
    visualize_raw_flow(video_idx=video_idx, frame_idx=frame_idx, stride=stride)


if __name__ == "__main__":
    main() 