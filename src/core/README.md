# Core - Fundamental Modules

This directory contains the fundamental building blocks of the vanishing point estimation system.

## Modules (in execution order)

### **`simple_viewer.py`**
A lightweight, real-time viewer to play videos and overlay predictions. It allows for basic controls like play/pause and frame navigation.

### **`interactive_viewer.py`** 
A more advanced, real-time interactive viewer for debugging and analyzing the system. Launches an OpenCV interface that allows:
- Frame-by-frame navigation through videos
- Comparison of predictions vs ground truth vs estimates
- Testing different optical flow visualizations
- Manual segmentation

### **`flow.py`** 
Generation and manipulation of optical flows (dense Farnebäck calculation).

### **`segmentation.py`** 
Object detection and masking (vehicles, hood) with YOLO to filter parasitic flows.

### **`flow_filter.py`** 
Filtering and weighting of optical flows (by norm, collinearity, etc.) to select the most relevant vectors.

### **`collinearity_scorer_sample.py`**
Calculates the collinearity score for a single frame, used for standard, non-parallel processing.

### **`collinearity_scorer_batch.py`**
Calculates collinearity scores for an entire batch of frames, optimized with MLX for high-performance, parallel processing.

### **`optimizers.py`** 
Centralized optimization methods (MLX Adam + scipy L-BFGS-B) to find the optimal vanishing point by minimizing the collinearity score.

### **`predictions_from_flows.py`**
High-level module that orchestrates the entire prediction pipeline, from loading flows to running the optimization and generating final predictions.

### **`rendering.py`** 
Rendering functions for the `interactive_viewer.py` (e.g., drawing flow arrows, masks, and other visual elements).
