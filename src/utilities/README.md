# Utilities

This directory contains support tools and helper functions for the `calib_challenge` project.

## Modules

### Project Setup and Constants
- **`paths.py`**: Centralized management of all project paths.
- **`project_constants.py`**: Defines global constants used throughout the project, such as focal length.

### Data Loading
- **`load_ground_truth.py`**: Loads and converts ground truth data (angles and pixels).
- **`load_flows.py`**: Loads pre-computed optical flows (supports both `.npy` and compressed `.npz` formats).
- **`load_video_frame.py`**: Provides a reliable way to load video frames, especially for HEVC-encoded videos.
- **`load_predictions.py`**: Loads epipole predictions from output files.
- **`load_mean_point.py`**: Loads the average prediction point for a given video.

### Data Generation and Preprocessing
- **`generate_flows.py`**: Generates optical flow fields with intelligent masking of moving objects and the car hood.
- **`create_mixed_batch.py`**: Creates a balanced test dataset by sampling frames from different error deciles.
- **`convert_to_float16.py`**: Compresses optical flow fields to float16 to optimize disk space.

### Heatmaps
- **`heatmap_generator.py`**: Generates heatmaps of collinearity scores to analyze spatial patterns.
- **`heatmap_loader.py`**: Loads and processes the generated heatmaps for use in filtering.

### Evaluation and Analysis
- **`eval.py`**: Evaluates predictions and calculates final scores.
- **`filter_config_evaluator.py`**: A comprehensive tool for evaluating the performance of different filtering configurations.
- **`worst_errors.py`**: Identifies and analyzes the frames with the highest prediction errors.
- **`extract_means.py`**: Converts angle predictions to pixel coordinates and calculates their mean values.
- **`pixel_angle_converter.py`**: Converts coordinates between the camera's frame of reference (pixels) and the car's frame of reference (angles).

### Post-processing
- **`fix_predictions.py`**: Fills in missing predictions for the first frame by duplicating the prediction from the second frame.
