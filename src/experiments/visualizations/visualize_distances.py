import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import os
import json

from src.utilities.paths import get_labeled_dir, get_pred_dir
from src.utilities.pixel_angle_converter import angles_to_pixels

DEFAULT_FOCAL_LENGTH = 910  # Focal length in pixels
IMAGE_WIDTH = 1920  # Standard image width
IMAGE_HEIGHT = 1080  # Standard image height

def calculate_distances(video_id, run_name="vanilla"):
    """Calculate distances between ground truth and predictions for a specific video and run."""
    # Load ground truth
    gt_path = get_labeled_dir() / f"{video_id}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    # Load predictions
    pred_path = get_pred_dir(run_name) / f"{video_id}.txt"
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    # Load data
    gt_data = np.loadtxt(gt_path)  # (pitch, yaw) in radians
    pred_data = np.loadtxt(pred_path)  # (pitch, yaw) in radians
    
    # Ensure same number of frames
    min_frames = min(len(gt_data), len(pred_data))
    gt_data = gt_data[:min_frames]
    pred_data = pred_data[:min_frames]
    
    # Initialize arrays for all frames
    distances = np.zeros(min_frames)
    valid_frames = []
    
    # Process each frame
    for i in range(min_frames):
        # If prediction has NaN, distance is 0
        if np.isnan(pred_data[i]).any():
            distances[i] = 0
            continue
            
        # If ground truth has NaN, skip this frame
        if np.isnan(gt_data[i]).any():
            continue
            
        # Convert ground truth angles to pixels
        pitch_gt, yaw_gt = gt_data[i]
        x_gt, y_gt = angles_to_pixels(pitch_gt, yaw_gt)
        
        # Convert prediction angles to pixels
        pitch_pred, yaw_pred = pred_data[i]
        x_pred, y_pred = angles_to_pixels(pitch_pred, yaw_pred)
        
        # Calculate distance
        distances[i] = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
        valid_frames.append(i)
    
    if not valid_frames:
        raise ValueError(f"No valid frames found for video {video_id}")
    
    print(f"Calculated {len(valid_frames)} distances for video {video_id} (run: {run_name})")
    return distances, valid_frames

def calculate_all_distances(run_name="vanilla"):
    """Calculate distances for all videos from a specific run."""
    all_distances = {}
    all_valid_frames = {}
    total_frames = 0
    
    for video_id in range(5):  # 0 to 4
        try:
            distances, valid_frames = calculate_distances(video_id, run_name)
            all_distances[video_id] = distances
            all_valid_frames[video_id] = valid_frames
            # Count only non-zero distances
            non_zero_distances = distances[distances > 0]
            total_frames += len(non_zero_distances)
            if len(non_zero_distances) > 0:
                print(f"Video {video_id}: {len(non_zero_distances)} valid frames, avg distance: {np.mean(non_zero_distances):.2f} pixels")
            else:
                print(f"Video {video_id}: No valid frames found")
        except Exception as e:
            print(f"⚠️  Video {video_id}: {e}")
    
    print(f"\nTotal valid frames across all videos: {total_frames}")
    return all_distances, all_valid_frames

def plot_single_video_distances(video_id, run_name="vanilla"):
    """Plot distances over time for a single video."""
    distances, valid_frames = calculate_distances(video_id, run_name)
    
    plt.figure(figsize=(12, 6))
    
    # Get non-zero distances and their corresponding frames
    non_zero_mask = distances > 0
    valid_distances = distances[non_zero_mask]
    valid_frame_indices = np.arange(len(distances))[non_zero_mask]
    
    # Find continuous segments of valid frames
    if len(valid_frame_indices) > 0:
        # Find where the frame indices are not consecutive
        breaks = np.where(np.diff(valid_frame_indices) > 1)[0] + 1
        segments = np.split(np.column_stack((valid_frame_indices, valid_distances)), breaks)
        
        # Plot each segment separately with the same color
        color = plt.cm.tab10(0)  # Use first color from tab10 colormap
        for segment in segments:
            if len(segment) > 0:  # Only plot non-empty segments
                plt.plot(segment[:, 0], segment[:, 1], color=color, linewidth=0.8, alpha=0.7)
    
    plt.title(f'Video {video_id} - Run "{run_name}" - Vanishing Point Estimation Errors Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Distance (pixels)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 300)  # Set fixed Y-axis limit
    
    # Add statistics (excluding zero distances)
    if len(valid_distances) > 0:
        mean_dist = np.mean(valid_distances)
        median_dist = np.median(valid_distances)
        std_dist = np.std(valid_distances)
        
        plt.axhline(y=mean_dist, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_dist:.1f}px')
        plt.axhline(y=median_dist, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_dist:.1f}px')
        
        plt.legend()
        plt.tight_layout()
        
        plt.show()
        
        # Print statistics
        print(f"\n--- STATISTICS FOR VIDEO {video_id} (RUN: {run_name}) ---")
        print(f"Total frames: {len(distances)}")
        print(f"Valid frames (non-zero distances): {len(valid_distances)}")
        print(f"Mean distance: {mean_dist:.2f} pixels")
        print(f"Median distance: {median_dist:.2f} pixels")
        print(f"Std deviation: {std_dist:.2f} pixels")
        print(f"Min distance: {np.min(valid_distances):.2f} pixels")
        print(f"Max distance: {np.max(valid_distances):.2f} pixels")
        print(f"25th percentile: {np.percentile(valid_distances, 25):.2f} pixels")
        print(f"75th percentile: {np.percentile(valid_distances, 75):.2f} pixels")
    else:
        print(f"No valid distances found for video {video_id}")

def plot_all_videos_comparison(run_name="vanilla"):
    """Plot comparison of all videos for a specific run."""
    all_distances, all_valid_frames = calculate_all_distances(run_name)
    
    if not all_distances:
        print("No valid data found!")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    stats = []
    
    # Use the same color for all videos
    color = plt.cm.tab10(0)  # First color from tab10 colormap
    
    for i, (video_id, distances) in enumerate(all_distances.items()):
        if i < 5:  # Max 5 videos (0-4)
            ax = axes[i]
            
            # Get non-zero distances and their corresponding frames
            non_zero_mask = distances > 0
            valid_distances = distances[non_zero_mask]
            valid_frame_indices = np.arange(len(distances))[non_zero_mask]
            
            # Find continuous segments of valid frames
            if len(valid_frame_indices) > 0:
                # Find where the frame indices are not consecutive
                breaks = np.where(np.diff(valid_frame_indices) > 1)[0] + 1
                segments = np.split(np.column_stack((valid_frame_indices, valid_distances)), breaks)
                
                # Plot each segment separately with the same color
                for segment in segments:
                    if len(segment) > 0:  # Only plot non-empty segments
                        ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=0.6, alpha=0.8)
            
            ax.set_ylim(0, 300)  # Set fixed Y-axis limit
            
            # Calculate mean excluding zero distances
            if len(valid_distances) > 0:
                mean_dist = np.mean(valid_distances)
                ax.axhline(y=mean_dist, color='red', linestyle='--', alpha=0.7)
                ax.set_title(f'Video {video_id} (Mean: {mean_dist:.1f}px)')
                
                stats.append({
                    'video_id': video_id,
                    'frames': len(valid_distances),
                    'mean': mean_dist,
                    'median': np.median(valid_distances),
                    'std': np.std(valid_distances),
                    'min': np.min(valid_distances),
                    'max': np.max(valid_distances)
                })
            else:
                ax.set_title(f'Video {video_id} (No valid frames)')
                stats.append({
                    'video_id': video_id,
                    'frames': 0,
                    'mean': np.nan,
                    'median': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan
                })
            
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Distance (pixels)')
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(all_distances), 6):
        axes[i].set_visible(False)
    
    fig.suptitle(f'All Videos Comparison - Run "{run_name}"', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    print(f"\n{'='*80}")
    print(f"COMPARISON STATISTICS - RUN: {run_name}")
    print(f"{'='*80}")
    print(f"{'Video':<8} {'Frames':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print(f"{'-'*80}")
    
    for stat in stats:
        if not np.isnan(stat['mean']):
            print(f"{stat['video_id']:<8} {stat['frames']:<8} {stat['mean']:<8.1f} {stat['median']:<8.1f} {stat['std']:<8.1f} {stat['min']:<8.1f} {stat['max']:<8.1f}")
        else:
            print(f"{stat['video_id']:<8} {stat['frames']:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")

def compare_runs(run1="vanilla", run2="1"):
    """Compare two different runs."""
    print(f"📊 COMPARING RUNS: {run1} vs {run2}")
    
    data1, valid_frames1 = calculate_all_distances(run1)
    data2, valid_frames2 = calculate_all_distances(run2)
    
    if not data1 or not data2:
        print("❌ Missing data for one or both runs!")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    print(f"\n{'='*80}")
    print(f"VIDEO-BY-VIDEO COMPARISON: {run1} vs {run2}")
    print(f"{'='*80}")
    print(f"{'Video':<8} {run1+' Avg':<12} {run2+' Avg':<12} {'Improvement':<12} {'% Change':<12}")
    print(f"{'-'*80}")
    
    for video_id in range(5):
        if video_id in data1 and video_id in data2:
            if video_id < 5:  # Plot first 5 videos
                ax = axes[video_id]
                
                # Get non-zero distances and their corresponding frames for both runs
                non_zero_mask1 = data1[video_id] > 0
                valid_distances1 = data1[video_id][non_zero_mask1]
                valid_frame_indices1 = np.arange(len(data1[video_id]))[non_zero_mask1]
                
                non_zero_mask2 = data2[video_id] > 0
                valid_distances2 = data2[video_id][non_zero_mask2]
                valid_frame_indices2 = np.arange(len(data2[video_id]))[non_zero_mask2]
                
                # Plot segments for run1
                if len(valid_frame_indices1) > 0:
                    breaks1 = np.where(np.diff(valid_frame_indices1) > 1)[0] + 1
                    segments1 = np.split(np.column_stack((valid_frame_indices1, valid_distances1)), breaks1)
                    color1 = plt.cm.tab10(0)  # First color for run1
                    for segment in segments1:
                        if len(segment) > 0:
                            ax.plot(segment[:, 0], segment[:, 1], color=color1, label=f'{run1}', alpha=0.7, linewidth=0.8)
                
                # Plot segments for run2
                if len(valid_frame_indices2) > 0:
                    breaks2 = np.where(np.diff(valid_frame_indices2) > 1)[0] + 1
                    segments2 = np.split(np.column_stack((valid_frame_indices2, valid_distances2)), breaks2)
                    color2 = plt.cm.tab10(1)  # Second color for run2
                    for segment in segments2:
                        if len(segment) > 0:
                            ax.plot(segment[:, 0], segment[:, 1], color=color2, label=f'{run2}', alpha=0.7, linewidth=0.8)
                
                ax.set_ylim(0, 300)  # Set fixed Y-axis limit
                
                # Add means (excluding zero distances)
                if len(valid_distances1) > 0:
                    mean1 = np.mean(valid_distances1)
                    ax.axhline(y=mean1, color=color1, linestyle='--', alpha=0.5)
                else:
                    mean1 = np.nan
                    
                if len(valid_distances2) > 0:
                    mean2 = np.mean(valid_distances2)
                    ax.axhline(y=mean2, color=color2, linestyle='--', alpha=0.5)
                else:
                    mean2 = np.nan
                
                if not np.isnan(mean1) and not np.isnan(mean2):
                    ax.set_title(f'Video {video_id}: {mean1:.1f}px vs {mean2:.1f}px')
                else:
                    ax.set_title(f'Video {video_id}: No valid frames')
                
                ax.set_xlabel('Frame Number')
                ax.set_ylabel('Distance (pixels)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Print comparison
            if not np.isnan(mean1) and not np.isnan(mean2):
                improvement = mean1 - mean2
                pct_change = (improvement / mean1) * 100 if mean1 > 0 else 0
                print(f"{video_id:<8} {mean1:<12.2f} {mean2:<12.2f} {improvement:<+12.2f} {pct_change:<+12.1f}%")
            else:
                print(f"{video_id:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    # Hide unused subplot
    if len(data1) < 6:
        axes[5].set_visible(False)
    
    fig.suptitle(f'Run Comparison: {run1} vs {run2}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Overall comparison
    all1 = np.concatenate([d[d > 0] for d in data1.values()])
    all2 = np.concatenate([d[d > 0] for d in data2.values()])
    
    if len(all1) > 0 and len(all2) > 0:
        overall_improvement = np.mean(all1) - np.mean(all2)
        overall_pct = (overall_improvement / np.mean(all1)) * 100 if np.mean(all1) > 0 else 0
        
        print(f"\n📊 OVERALL SUMMARY:")
        print(f"  {run1} average:  {np.mean(all1):.2f}px")
        print(f"  {run2} average:  {np.mean(all2):.2f}px")
        print(f"  Overall improvement: {overall_improvement:+.2f}px ({overall_pct:+.1f}%)")
    else:
        print("\n📊 OVERALL SUMMARY: No valid frames found for comparison")

def plot_distance_distribution(run_name="vanilla", save_plot=True):
    """Plot distribution of distances across all videos for a specific run."""
    all_distances, _ = calculate_all_distances(run_name)
    
    if not all_distances:
        print("No valid data found!")
        return
    
    # Combine all non-zero distances
    combined_distances = np.concatenate([d[d > 0] for d in all_distances.values()])
    
    if len(combined_distances) == 0:
        print("No valid distances found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(combined_distances, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Distribution of All Distances - Run "{run_name}"')
    axes[0, 0].set_xlabel('Distance (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot by video
    video_distances = [d[d > 0] for d in all_distances.values()]
    video_labels = [f'Video {vid}' for vid in sorted(all_distances.keys())]
    
    axes[0, 1].boxplot(video_distances, labels=video_labels)
    axes[0, 1].set_title('Distance Distribution by Video')
    axes[0, 1].set_ylabel('Distance (pixels)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_distances = np.sort(combined_distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    
    axes[1, 0].plot(sorted_distances, cumulative)
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].set_xlabel('Distance (pixels)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add percentile lines
    for percentile in [50, 75, 90, 95]:
        value = np.percentile(combined_distances, percentile)
        axes[1, 0].axvline(x=value, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].text(value, percentile/100, f'{percentile}%: {value:.1f}px', 
                       rotation=90, verticalalignment='bottom')
    
    # Violin plot by video
    parts = axes[1, 1].violinplot(video_distances, positions=range(1, len(video_distances) + 1))
    axes[1, 1].set_title('Distance Distribution Shape by Video')
    axes[1, 1].set_xlabel('Video ID')
    axes[1, 1].set_ylabel('Distance (pixels)')
    axes[1, 1].set_xticks(range(1, len(video_distances) + 1))
    axes[1, 1].set_xticklabels([f'{vid}' for vid in sorted(all_distances.keys())])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        from src.utilities.paths import get_visualizations_dir, ensure_dir_exists
        output_dir = ensure_dir_exists(get_visualizations_dir() / "distances" / run_name)
        output_file = output_dir / "distance_distributions.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Distribution plots saved to: {output_file}")
    
    plt.show()
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS - RUN: {run_name}")
    print(f"{'='*60}")
    print(f"Total valid frames: {len(combined_distances)}")
    print(f"Mean distance: {np.mean(combined_distances):.2f} pixels")
    print(f"Median distance: {np.median(combined_distances):.2f} pixels")
    print(f"Std deviation: {np.std(combined_distances):.2f} pixels")
    print(f"Min distance: {np.min(combined_distances):.2f} pixels")
    print(f"Max distance: {np.max(combined_distances):.2f} pixels")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(combined_distances, p):.2f} pixels")

def analyze_error_patterns(run_name="vanilla"):
    """Analyze patterns in the errors for a specific run."""
    all_distances, _ = calculate_all_distances(run_name)
    
    if not all_distances:
        print("No valid data found!")
        return
    
    print(f"\n{'='*80}")
    print(f"ERROR PATTERN ANALYSIS - RUN: {run_name}")
    print(f"{'='*80}")
    
    for video_id, distances in all_distances.items():
        print(f"\n--- VIDEO {video_id} ---")
        
        # Get non-zero distances
        non_zero_distances = distances[distances > 0]
        
        if len(non_zero_distances) == 0:
            print("No valid distances found")
            continue
        
        # Find frames with very high errors
        high_error_threshold = np.percentile(non_zero_distances, 95)
        high_error_frames = np.where(distances > high_error_threshold)[0]
        
        print(f"High error threshold (95th percentile): {high_error_threshold:.1f} pixels")
        print(f"Number of high error frames: {len(high_error_frames)} ({len(high_error_frames)/len(non_zero_distances)*100:.1f}%)")
        
        if len(high_error_frames) > 0:
            print(f"High error frame indices: {high_error_frames[:10]}{'...' if len(high_error_frames) > 10 else ''}")
            print(f"Max error: {np.max(non_zero_distances):.1f} pixels at frame {np.argmax(distances)}")
        
        # Find frames with very low errors
        low_error_threshold = np.percentile(non_zero_distances, 5)
        low_error_frames = np.where(distances < low_error_threshold)[0]
        
        print(f"Low error threshold (5th percentile): {low_error_threshold:.1f} pixels")
        print(f"Number of low error frames: {len(low_error_frames)} ({len(low_error_frames)/len(non_zero_distances)*100:.1f}%)")
        
        if len(low_error_frames) > 0:
            print(f"Min error: {np.min(non_zero_distances):.1f} pixels at frame {np.argmin(distances)}")

def main(run_name="5"):
    """Main function with menu for different visualization options."""
    print("=== DISTANCE VISUALIZATION TOOL ===")
    print()
    
    while True:
        print("\nChoose visualization option:")
        print("1. Plot single video distances")
        print("2. Compare all videos (single run)")
        print("3. Plot distance distributions (single run)")
        print("4. Analyze error patterns (single run)")
        print("5. Compare two runs")
        print("6. Show all visualizations (single run)")
        print("0. Exit")
        
        choice = input("\nEnter choice (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            video_id = input("Enter video ID (0-4): ").strip()
            try:
                video_id = int(video_id)
                if 0 <= video_id <= 4:
                    plot_single_video_distances(video_id, run_name)
                else:
                    print("Invalid video ID. Please enter 0-4.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif choice == "2":
            plot_all_videos_comparison(run_name)
        elif choice == "3":
            plot_distance_distribution(run_name)
        elif choice == "4":
            analyze_error_patterns(run_name)
        elif choice == "5":
            run2 = input("Enter run number to compare with: ").strip()
            compare_runs(run_name, run2)
        elif choice == "6":
            print("Generating all visualizations...")
            plot_all_videos_comparison(run_name)
            plot_distance_distribution(run_name)
            analyze_error_patterns(run_name)
            print("All visualizations completed!")
        else:
            print("Invalid choice. Please enter 0-6.")

if __name__ == "__main__":
    main("5_4") 