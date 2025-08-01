# Experiments

This directory contains all scripts used for experimentation, analysis, performance benchmarking, and generating visualizations for the project.

## Structure

### **`benchmarks/`**

This folder contains scripts for performance testing and optimization. Key benchmarks include:
- Optimizer comparisons (Adam vs. L-BFGS-B)
- Hyperparameter searches for filtering (e.g., using Bayesian optimization)
- Analysis of data loading times and quantization precision loss
- Evaluation of the frame sampling strategy

### **`visualizations/`**

This folder contains scripts to create a wide range of visualizations for analysis and documentation. Key areas include:
- **Animations**: GIFs of prediction results and core concepts (e.g., collinearity score).
- **Optimizer Analysis**: Visuals of optimization paths and parameter effects.
- **Spatial Analysis**: Heatmaps and topographic maps of collinearity scores.
- **Error Analysis**: Plots of prediction error, distances, and data distributions.
- **Frame-by-Frame Inspection**: Detailed views of optical flow, filtering, and predictions on individual frames.
