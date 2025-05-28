import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def get_project_root():
    """Get the absolute path to the project root directory."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to reach the project root
    return os.path.dirname(script_dir)

def visualize_temporal_errors(test_dir):
    """
    Visualize temporal errors for all 5 videos.
    
    Args:
        test_dir (str): Directory containing the prediction files (relative to project root)
    """
    project_root = get_project_root()
    gt_dir = os.path.join(project_root, 'labeled')
    test_dir = os.path.join(project_root, test_dir)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Colors for different videos
    colors = ['b', 'g', 'r', 'c', 'm']
    
    # Lists to store overall errors
    zero_mses = []
    mses = []
    processed_videos = []
    
    for i in range(5):
        # Load ground truth and predictions
        gt_path = os.path.join(gt_dir, f'{i}.txt')
        test_path = os.path.join(test_dir, f'{i}.txt')
        
        # Skip if either file is missing
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found: {gt_path}")
            continue
        if not os.path.exists(test_path):
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
                
                # Plot temporal error
                plt.plot(temporal_errors, label=f'Video {i}', color=colors[i], alpha=0.7)
            
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
    
    plt.title('Temporal Error Evolution Across Videos (Relative to Zero Baseline)')
    plt.xlabel('Frame Number')
    plt.ylabel('Error (% relative to zero baseline)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Improve x-axis precision with ticks every 10 frames and labels every 50 frames
    max_frames = max([len(np.loadtxt(os.path.join(test_dir, f'{i}.txt'))) for i in processed_videos])
    plt.xticks(np.arange(0, max_frames, 10))  # Minor ticks every 10 frames
    
    # Set tick labels and increase size for labeled ticks
    tick_labels = [str(i) if i % 50 == 0 else '' for i in range(0, max_frames, 10)]
    plt.gca().set_xticklabels(tick_labels)
    
    # Increase the size of ticks where numbers are displayed
    for tick in plt.gca().xaxis.get_major_ticks():
        if tick.label1.get_text() != '':
            tick.tick1line.set_markersize(8)  # Increase tick size
            tick.label1.set_fontsize(10)      # Increase font size
    
    plt.show()

if __name__ == "__main__":
    TEST_DIR = 'pred/4'  # This is now relative to project root
    visualize_temporal_errors(TEST_DIR)
