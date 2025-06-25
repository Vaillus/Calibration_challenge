import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utilities.paths import get_labeled_dir, get_pred_dir
from src.utilities.pixel_angle_converter import angles_to_pixels

DEFAULT_FOCAL_LENGTH = 910  # Focal length in pixels
IMAGE_WIDTH = 1920  # Standard image width
IMAGE_HEIGHT = 1080  # Standard image height

def load_distances_for_video(video_id, run_name="vanilla"):
    """Load and calculate distances for a specific video."""
    # Load ground truth
    gt_path = get_labeled_dir() / f"{video_id}.txt"
    if not gt_path.exists():
        return None
    
    # Load predictions
    pred_path = get_pred_dir(run_name) / f"{video_id}.txt"
    if not pred_path.exists():
        return None
    
    # Load data
    gt_data = np.loadtxt(gt_path)  # (yaw, pitch) in radians
    pred_data = np.loadtxt(pred_path)  # (yaw, pitch) in radians
    
    # Ensure same number of frames
    min_frames = min(len(gt_data), len(pred_data))
    gt_data = gt_data[:min_frames]
    pred_data = pred_data[:min_frames]
    
    # Calculate distances for valid frames only
    valid_distances = []
    
    for i in range(min_frames):
        # Skip if prediction has NaN or if ground truth has NaN
        if np.isnan(pred_data[i]).any() or np.isnan(gt_data[i]).any():
            continue
            
        # Convert angles to pixels
        yaw_gt, pitch_gt = gt_data[i]
        x_gt, y_gt = angles_to_pixels(yaw_gt, pitch_gt, DEFAULT_FOCAL_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        yaw_pred, pitch_pred = pred_data[i]
        x_pred, y_pred = angles_to_pixels(yaw_pred, pitch_pred, DEFAULT_FOCAL_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        # Calculate distance
        distance = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
        valid_distances.append(distance)
    
    return np.array(valid_distances) if valid_distances else None

def load_all_distances(run_name="vanilla"):
    """Load distances for all videos."""
    distances_by_video = {}
    
    for video_id in range(5):  # Videos 0 to 4
        distances = load_distances_for_video(video_id, run_name)
        if distances is not None and len(distances) > 0:
            distances_by_video[video_id] = distances
            print(f"Video {video_id}: {len(distances)} valid frames, avg: {np.mean(distances):.1f}px")
        else:
            print(f"Video {video_id}: No valid data")
    
    return distances_by_video

def plot_boxplot_distributions(run_name="vanilla"):
    """Create box plot distribution by video."""
    distances_by_video = load_all_distances(run_name)
    
    if not distances_by_video:
        print("No valid data found!")
        return
    
    # Prepare data for plotting
    video_ids = sorted(distances_by_video.keys())
    distances_list = [distances_by_video[vid] for vid in video_ids]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create box plots
    bp = plt.boxplot(distances_list, labels=[str(vid) for vid in video_ids], 
                     patch_artist=True, showmeans=True, meanline=True)
    
    plt.title(f'Distribution des distances par vidéo - Run "{run_name}"', fontsize=16, pad=20)
    plt.xlabel('Vidéo', fontsize=14)
    plt.ylabel('Distance (pixels)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Color the box plots with a nice palette
    colors = plt.cm.Set2(np.linspace(0, 1, len(distances_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add some statistics as text above each box
    for i, (vid, distances) in enumerate(distances_by_video.items()):
        mean_val = np.mean(distances)
        median_val = np.median(distances)
        # Add mean value as text above the box
        plt.text(i + 1, mean_val + np.std(distances) + 5, f'μ={mean_val:.1f}px', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ DES DISTRIBUTIONS - RUN: {run_name}")
    print(f"{'='*60}")
    print(f"{'Vidéo':<8} {'N frames':<10} {'Moyenne':<10} {'Médiane':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    print(f"{'-'*70}")
    
    for vid in video_ids:
        distances = distances_by_video[vid]
        print(f"{vid:<8} {len(distances):<10} {np.mean(distances):<10.1f} {np.median(distances):<10.1f} "
              f"{np.std(distances):<10.1f} {np.min(distances):<8.1f} {np.max(distances):<8.1f}")
    
    # Global summary
    all_distances = np.concatenate(list(distances_by_video.values()))
    print(f"\n{'='*40}")
    print(f"RÉSUMÉ GLOBAL")
    print(f"{'='*40}")
    print(f"Total frames valides: {len(all_distances)}")
    print(f"Distance globale moyenne: {np.mean(all_distances):.2f} pixels")
    print(f"Distance globale médiane: {np.median(all_distances):.2f} pixels")
    print(f"Écart-type global: {np.std(all_distances):.2f} pixels")
    
    # Variability between videos
    video_means = [np.mean(distances_by_video[vid]) for vid in video_ids]
    print(f"\nVariabilité entre vidéos:")
    print(f"  Écart-type des moyennes: {np.std(video_means):.2f} pixels")
    print(f"  Range des moyennes: {np.min(video_means):.1f} - {np.max(video_means):.1f} pixels")

def main():
    """Main function to run the distribution analysis."""
    print("=== ANALYSE DES DISTRIBUTIONS DE DISTANCES PAR VIDÉO ===")
    
    # Ask for run name
    run_name = input("Entrez le nom du run à analyser (défaut: 'vanilla'): ").strip()
    if not run_name:
        run_name = "5"
    
    print(f"\nAnalyse du run: {run_name}")
    
    # Generate box plot visualization
    plot_boxplot_distributions(run_name)

if __name__ == "__main__":
    main() 