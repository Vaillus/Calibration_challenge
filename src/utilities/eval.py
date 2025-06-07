"""
Camera Calibration Challenge - Evaluation Script
================================================

This script evaluates the performance of vanishing point predictions by comparing
them with ground truth labels. It calculates MSE (Mean Squared Error) and provides
a percentage error score relative to an all-zeros baseline.

USAGE:
------
    # From another script
    from src.utilities.eval import evaluate_predictions
    
    # Evaluate specific test directory
    score = evaluate_predictions('pred/4')
    
    # Evaluate with custom ground truth directory
    score = evaluate_predictions('pred/4', gt_dir='labeled')

OUTPUT:
-------
- Returns percentage error score (lower is better)
- Prints detailed evaluation results to console

EVALUATION METRICS:
------------------
- MSE per video between predictions and ground truth
- Overall percentage error vs all-zeros baseline
- Individual video performance breakdown
"""

import numpy as np
import os
import sys

def get_mse(gt, test):
    """
    Calculate Mean Squared Error between ground truth and test predictions.
    
    Args:
        gt (np.array): Ground truth values
        test (np.array): Test predictions
        
    Returns:
        float: Mean squared error
    """
    test = np.nan_to_num(test)
    return np.mean(np.nanmean((gt - test)**2, axis=0))

def evaluate_predictions(test_dir, gt_dir='labeled'):
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        test_dir (str): Directory containing prediction files (relative to project root)
        gt_dir (str): Directory containing ground truth files (relative to project root)
        
    Returns:
        float: Percentage error score (lower is better)
    """
    # Get project root directory (calib_challenge/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    
    # Build full paths
    full_test_dir = os.path.join(project_root, test_dir)
    full_gt_dir = os.path.join(project_root, gt_dir)
    
    print(f"📊 EVALUATION REPORT")
    print(f"===================")
    print(f"Ground Truth Dir: {full_gt_dir}")
    print(f"Test Predictions Dir: {full_test_dir}")
    print()
    
    zero_mses = []
    mses = []
    
    for i in range(0, 5):
        # Load ground truth
        gt_file = os.path.join(full_gt_dir, f'{i}.txt')
        if not os.path.exists(gt_file):
            print(f"⚠️  Warning: Ground truth file {gt_file} not found")
            continue
            
        gt = np.loadtxt(gt_file)
        zero_mses.append(get_mse(gt, np.zeros_like(gt)))

        # Load test predictions
        test_file = os.path.join(full_test_dir, f'{i}.txt')
        if not os.path.exists(test_file):
            print(f"⚠️  Warning: Test file {test_file} not found")
            continue
            
        test = np.loadtxt(test_file)
        mse = get_mse(gt, test)
        mses.append(mse)
        
        print(f"📹 Video {i}:")
        print(f"   MSE: {mse:.4f}")
        print(f"   Zero baseline MSE: {zero_mses[-1]:.4f}")
        print(f"   Relative performance: {100*mse/zero_mses[-1]:.2f}%")
        print()

    if not mses or not zero_mses:
        print("❌ No valid files found for evaluation")
        return None
    
    # Calculate overall performance
    percent_err_vs_all_zeros = 100 * np.mean(mses) / np.mean(zero_mses)
    
    print(f"📈 OVERALL RESULTS")
    print(f"==================")
    print(f"Average MSE: {np.mean(mses):.4f}")
    print(f"Average Zero MSE: {np.mean(zero_mses):.4f}")
    print(f"YOUR ERROR SCORE IS {percent_err_vs_all_zeros:.2f}% (lower is better)")
    
    return percent_err_vs_all_zeros

def main():
    """
    Main function for direct script execution.
    Provides example usage of the evaluation function.
    """
    # Example: evaluate predictions in 'pred/4' directory
    score = evaluate_predictions('pred/4')
    
    if score is not None:
        print(f"\n🎯 Final Score: {score:.2f}%")
    else:
        print("\n❌ Evaluation failed")

if __name__ == "__main__":
    main() 