# Visualizations

This directory contains various scripts to visualize data, analyze results, and generate figures for the project documentation.

## Scripts

### Prediction and Animation Generators

- **`predictions_gif_generator.py`**: Creates GIFs to visualize the final predictions overlaid on the video frames, comparing different runs.
- **`collinearity_gif.py`**: Generates an animation showing the concept of the collinearity score for a single pixel and a moving candidate point.
- **`global_collinearity_animation.py`**: Creates a GIF illustrating how the global collinearity score changes as the candidate epipole moves across the image.

### Optimizer and Filter Analysis

- **`visualize_optimizer_single_frame.py`**: Shows the optimization path of different algorithms (e.g., Adam, L-BFGS-B) for a single frame.
- **`visualize_optimizers_distribution.py`**: Visualizes the distribution of final predictions from different optimizers.
- **`visualize_adam_early_stopping_arc5.py`**: Analyzes the effect of the early stopping criteria on the Adam optimizer's performance.
- **`compare_flow_filters.py`**: Visualizes the effect of different optical flow filtering strategies on the resulting vector field.

### Heatmaps and Spatial Analysis

- **`heatmap_visualizer.py`**: Generates and displays heatmaps of collinearity scores to identify which image regions are most informative.
- **`collinearity_topographic_map.py`**: Creates a 3D topographic map of the collinearity score landscape for a given frame.
- **`visualize_pixel_distributions.py`**: Visualizes the spatial distribution of predicted vanishing points versus the ground truth, including confidence ellipses.

### Error and Data Distribution Analysis

- **`visualize_distances.py`**: Analyzes and plots the pixel distance between predictions and ground truth, showing error over time.
- **`visualize_distance_distributions_per_video.py`**: Displays the distribution of prediction errors for each video as a boxplot.
- **`visualize_error.py`**: Provides a temporal analysis of estimation errors compared to a baseline.
- **`visualize_flow_vectors_count.py`**: Plots the number of optical flow vectors remaining after filtering for each frame.

### Single Frame Visualizers

- **`visualize_frame_flow.py`**: Displays the raw optical flow vector field for a single frame.
- **`visualize_frame_colin.py`**: Visualizes the collinearity scores of flow vectors relative to a reference point on a single frame.
- **`visualize_frame_full.py`**: A comprehensive single-frame visualizer that combines flow, collinearity, and prediction information.

### Notebooks
- **`visualize_gradient_descent.ipynb`**: Interactive analysis of the gradient descent process.
- **`arcs_1_2_visus.ipynb`**: Visualizations and analyses specific to the early stages of the project (Arcs 1 and 2).
