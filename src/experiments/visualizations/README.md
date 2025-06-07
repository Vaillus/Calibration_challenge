# 📊 Visualizations

This directory contains visualization scripts to analyze the results of the vanishing point estimation system.

## 📁 Scripts

### 📏 **`visualize_distances.py`**
Analyzes and visualizes distances between predictions and ground truth:
- Pixel distance calculation for each video
- Temporal error visualization
- Detailed statistics (mean, median, standard deviation)
- Comparison between different runs

### 🎥 **`visualize_frame_flow.py`**
Visualizes optical flow fields on a frame:
- Display of flow vectors with arrows
- Overlay of ground truth and image center
- Filtering of low magnitude flows

### 🔍 **`visualize_frame_colin.py`**
Analyzes flow vector collinearity:
- Collinearity score calculation relative to a reference point
- Spatial visualization of scores on the frame
- Statistics on score distribution

### 📈 **`visualize_pixel_distributions.py`**
Visualizes spatial distribution of vanishing points:
- Predictions vs ground truth comparison
- Means, medians and standard deviations
- Confidence ellipses
- Reference central point

### ⚠️ **`visualize_error.py`**
Temporal analysis of estimation errors:
- Frame-by-frame error evolution
- Comparison with zero baseline
- Global and per-video statistics

### 📓 Notebooks
- `visualize_gradient_descent.ipynb`: Gradient descent analysis
- `arcs_1_2_visus.ipynb`: Visualizations specific to the early stages of the project