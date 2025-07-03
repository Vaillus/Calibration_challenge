import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from src.utilities.paths import get_labeled_dir, get_pred_dir

def visualize_temporal_errors(test_dir):
    """
    Visualize temporal errors for all 5 videos in separate plots.
    
    Args:
        test_dir (str): Run number (e.g., '5' for pred/5/)
    """
    gt_dir = get_labeled_dir()
    pred_dir = get_pred_dir(test_dir)  # Use video_num parameter
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(10, 12))
    fig.suptitle('Temporal Error Evolution Across Videos (Relative to Zero Baseline)', fontsize=12)
    
    # Colors for different videos
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Lists to store overall errors
    zero_mses = []
    mses = []
    processed_videos = []
    
    for i in range(5):
        # Load ground truth and predictions
        gt_path = gt_dir / f'{i}.txt'
        test_path = pred_dir / f'{i}.txt'
        
        # Skip if either file is missing
        if not gt_path.exists():
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue
        if not test_path.exists():
            print(f"Warning: Prediction file not found: {test_path}")
            continue
            
        try:
            gt = np.loadtxt(gt_path)
            test = np.loadtxt(test_path)
            
            # Calculate temporal error (MSE for each frame)
            test = np.nan_to_num(test)
            
            # Calculate MSE for predictions
            pred_mse = np.nanmean((gt - test)**2, axis=1)
            # Calculate MSE for zero baseline
            zero_mse = np.nanmean((gt - np.zeros_like(gt))**2, axis=1)
            
            # Store overall errors (handle NaN values)
            if not np.isnan(pred_mse).all() and not np.isnan(zero_mse).all():
                mses.append(np.nanmean(pred_mse))
                zero_mses.append(np.nanmean(zero_mse))
                processed_videos.append(i)
                
                # Calculate percentage error relative to zero baseline
                temporal_errors = 100 * pred_mse / zero_mse
                # Replace any remaining NaN values with 0
                temporal_errors = np.nan_to_num(temporal_errors, nan=0.0)
                
                # Plot temporal error in corresponding subplot
                ax = axes[i]
                ax.plot(temporal_errors, color=colors[i], alpha=0.7)
                ax.set_title(f'Video {i}', fontsize=10)
                ax.set_xlabel('Frame Number', fontsize=9)
                ax.set_ylabel('Error (% relative to zero baseline)', fontsize=9)
                ax.set_ylim(0, 1500)
                ax.grid(True, alpha=0.3)
                
                # Improve x-axis precision with ticks every 10 frames and labels every 50 frames
                max_frames = len(temporal_errors)
                ax.set_xticks(np.arange(0, max_frames, 10))  # Minor ticks every 10 frames
                
                # Set tick labels and increase size for labeled ticks
                tick_labels = [str(i) if i % 50 == 0 else '' for i in range(0, max_frames, 10)]
                ax.set_xticklabels(tick_labels)
                
                # Adjust the size of ticks where numbers are displayed
                for tick in ax.xaxis.get_major_ticks():
                    if tick.label1.get_text() != '':
                        tick.tick1line.set_markersize(6)  # Moderate tick size
                        tick.label1.set_fontsize(8)      # Smaller font size
                
                # Also adjust y-axis tick size
                ax.tick_params(axis='y', labelsize=8)
                
                # Set y-axis to log scale for better visualization of error variations
                ax.set_yscale('log')
                ax.set_ylim(1, 10000)  # Adjust limits for log scale
            
        except Exception as e:
            print(f"Error processing video {i}: {str(e)}")
            continue
    
    if not processed_videos:
        print("No videos were successfully processed!")
        return
        
    # Calculate and print overall error (handle NaN values)
    mses = np.array(mses)
    zero_mses = np.array(zero_mses)
    valid_indices = ~(np.isnan(mses) | np.isnan(zero_mses))
    
    if np.any(valid_indices):
        percent_err_vs_all_zeros = 100 * np.mean(mses[valid_indices]) / np.mean(zero_mses[valid_indices])
        print(f'\nOverall error score: {percent_err_vs_all_zeros:.2f}% (lower is better)')
        
        # Print individual video errors
        print('\nIndividual video errors:')
        for i, (mse, zero_mse) in enumerate(zip(mses, zero_mses)):
            if not np.isnan(mse) and not np.isnan(zero_mse):
                video_err = 100 * mse / zero_mse
                print(f'Video {processed_videos[i]}: {video_err:.2f}%')
    else:
        print("\nNo valid error scores could be calculated!")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TEST_DIR = '5_4'  # This is now relative to pred directory
    visualize_temporal_errors(TEST_DIR)
