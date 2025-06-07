# Calibration Challenge Solution

This project is my attempt to solve the [Comma.ai Calibration Challenge](https://github.com/commaai/calib_challenge). The goal is to predict the orientation of a camera mounted on a vehicle from video sequences.

## 🎯 Challenge Objective

Predict two camera orientation angles:
- **Pitch** ($\phi_o$) : vertical angle
  - $\phi_v$ : dynamic vehicle pitch (speed bumps, braking, etc.)
  - $\phi_c$ : fixed vertical angular offset between camera and vehicle
- **Yaw** ($\theta_o$) : horizontal angle
  - $\theta_v$ : dynamic yaw (turns)
  - $\theta_c$ : fixed horizontal angular offset between camera and vehicle

## 📊 Dataset

- 10 one-minute videos
- 5 labeled videos for training
- 5 unlabeled videos for prediction
- Various environments and lighting conditions

## 🏗️ Project Structure

The project follows a modular and organized architecture:

```
calib_challenge/
├── src/
│   ├── core/           # Core modules
│   ├── production/     # Production scripts
│   ├── utilities/      # Support tools
│   └── experiments/    # Experimentation scripts
│       ├── benchmarks/     # Performance tests
│       └── visualizations/ # Visualizations
├── data/
│   ├── inputs/        # Input data
│   ├── intermediate/  # Intermediate data
│   └── outputs/       # Results
├── models/           # Model weights
└── prompts/          # Documentation and notes
```

## 🔧 Solution Approach

### 1. Optical Flow Analysis
- Dense optical flow implementation using OpenCV
- Optimized parameters for global motion detection
- Float16 quantization to reduce memory footprint

### 2. Intelligent Segmentation
- YOLOv8 for object segmentation
- Filtering of disruptive elements (vehicles, pedestrians)
- Manual mask for the vehicle hood

### 3. Vanishing Point Estimation
- Vector collinearity-based method
- Adam optimization with plateau detection
- Intelligent flow vector filtering

### 4. Performance Optimizations
- MLX-based GPU acceleration
- Precomputed and compressed flow fields
- Smart frame sampling for optimization

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Vaillus/Calibration_challenge.git
cd Calibration_challenge
```

2. Download the YOLOv8 model:
```bash
mkdir -p models
# Download yolov8x-seg.pt from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
# and place it in the models/ directory
```

3. Install dependencies:
```bash
pip install -e .
```

## 📝 Development Status

The project is under active development. Current focus areas:
- Filtering parameter optimization
- Edge case handling improvements
- Optimization method comparison (Adam vs L-BFGS-B)

## 📚 References

- [Comma.ai Calibration Challenge](https://github.com/commaai/calib_challenge)
- [OpenCV Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MLX](https://github.com/ml-explore/mlx)

