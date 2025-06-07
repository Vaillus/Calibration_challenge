# Core - Fundamental Modules

This directory contains the fundamental building blocks of the vanishing point estimation system.

## Modules (in execution order)

### **`interactive_viewer.py`** 
Real-time interactive viewer for debugging and analyzing the system. Launches an OpenCV interface that allows:
- Frame-by-frame navigation through videos
- Comparison of predictions vs ground truth vs estimates
- Testing different optical flow visualizations
- Manual segmentation

### **`flow.py`** 
Generation and manipulation of optical flows (dense Farnebäck calculation, separation points)

### **`segmentation.py`** 
Object detection and masking (vehicles, hood) with YOLO to filter parasitic flows

### **`flow_filter.py`** 
Filtering and weighting of optical flows (by norm, collinearity, distance to center)

### **`predictions_from_flows.py`**
Prediction generator from pre-computed optical flows.

### **`colinearity_optimization.py`** 
Vanishing point estimator (classic numpy/scipy version) with collinearity score calculation

### **`colinearity_optimization_parallel.py`** 
Vanishing point estimator (parallel MLX version) for high-performance batch processing

### **`optimizers.py`** 
Centralized optimization methods (MLX Adam + scipy L-BFGS-B) to find optimal vanishing points

### **`rendering.py`** 
Rendering functions for the interactive_viewer.py (flow arrows, separation points, masks)