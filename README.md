# Onboard Camera Calibration in a Semi-Autonomous Car

This repository contains my solution to the [Comma.ai Camera Calibration Challenge](https://github.com/commaai/calib_challenge). It's a fascinating computer vision problem where the goal is to determine the orientation of a car's onboard camera relative to its direction of motion, using only video footage.

This project was developed iteratively, starting from a simple baseline and progressively adding layers of sophistication. The journey is documented in a detailed blog post, which you can find here:

➡️ **[Read the Full, Detailed Write-Up](./docs/en/index.md)**

<!-- --- -->

<!-- ## 🏆 Current Performance on Labeled Data: 8.58%

After several stages of optimization and post-processing on the 5 labeled videos, the model achieved an error score of **8.58%**. The evaluation metric considers 0% a perfect score and 100% the score of a naive baseline that always predicts the center of the image.

The predictions for the unlabeled test set have been generated and are ready for submission to obtain the final score on the leaderboard.

--- -->

##  odyssey: My Journey to the Solution

My approach evolved through five main "arcs," each building upon the last to improve performance.

### Arc 1: The Optical Flow Baseline (Score: 1960.20%)
I started by establishing a baseline using dense optical flow (cv2.calcOpticalFlowFarneback) to estimate the epipole (the point of convergence). The initial method, based on finding horizontal and vertical separation lines in the flow field, was intuitive but very noisy and performed poorly.

### Arc 2: Adding Segmentation to Cut Through the Noise (Score: 812.57%)
The key insight was that many errors came from non-static elements like other cars or reflections on the car's hood. I introduced two filtering steps:
1.  **YOLOv8-seg**: To detect and mask out other vehicles in the scene.
2.  **Manual Masking**: To ignore the car's hood, which was a consistent source of noise.
This step dramatically improved the signal quality and cut the error by more than half.

### Arc 3: A New Method for Epipole Estimation (Score: 168.83%)
Instead of the simplistic separation method, I reformulated the problem as an optimization task. I defined a **global collinearity score** that measures how well flow vectors align with a candidate epipole. The goal is to find the point that minimizes this score. I used the **L-BFGS-B** optimizer to find this minimum efficiently, leading to another significant performance boost.

### Arc 4: Performance Optimization and Advanced Filtering (Score: 54.32%)
To iterate faster, I needed to optimize my pipeline. This arc was all about speed and smarter data handling:
- **Pre-computation**: I generated and stored all optical flow fields to avoid recalculating them.
- **GPU Acceleration**: I rewrote the filtering logic using **Apple's MLX library** to leverage the M1 GPU, parallelizing operations on pixels and frames.
- **Intelligent Sampling**: I created a representative 100-frame sample to run experiments in seconds instead of hours.
This speed allowed me to systematically search for optimal hard filtering thresholds (based on vector norm and collinearity), leading to a much better score.

### Arc 5: The Final Polish (Score on Labeled Data: 8.58%)
The final arc focused on refining the details:
- **Sigmoidal Weighting**: I replaced "hard" filtering with a more flexible sigmoidal function, allowing for soft weighting of vectors.
- **Reference Point Optimization**: Instead of using the image center as a reference for filtering, I used the mean of a previous prediction run, which proved to be a more accurate heuristic.
- **Bayesian Search**: I used `skopt` to run a Bayesian search over the now 6-dimensional parameter space to find the optimal filter configuration.
- **Post-processing**: The final touch was to apply a **bi-directional exponential smoothing** to the time series of predictions, which dramatically reduced noise and produced the score of 8.58% on the labeled data.

---

## 🔧 Key Technologies
- **Computer Vision**: `OpenCV`
- **Object Segmentation**: `YOLOv8-seg`
- **GPU-Accelerated Computing**: `MLX` (for Apple Silicon)
- **Optimization**: `SciPy`, custom `Adam` implementation, `scikit-optimize`

---

## 🏗️ Project Structure
The project is organized as follows:
```
calib_challenge/
├── src/
│   ├── core/           # Core modules for optical flow, optimization, etc.
│   ├── production/     # Scripts for making final predictions
│   ├── utilities/      # Helper scripts and data loaders
│   └── experiments/    # Experimentation and visualization scripts
├── data/
│   ├── inputs/         # Original video and label files
│   ├── intermediate/   # Pre-computed data like flow fields
│   └── outputs/        # Final prediction files and results
├── models/             # Trained model weights (e.g., for YOLO)
└── docs/               # Full project documentation and write-up
```

---

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Vaillus/Calibration_challenge.git
    cd Calibration_challenge
    ```

2.  Download the YOLOv8 model:
    ```bash
    # Download yolov8x-seg.pt from https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
    # and place it in the models/ directory
    # Note: You might need to create the models/ directory first.
    ```

3.  Install dependencies:
    ```bash
    pip install -e .
    ```

---

## 📚 References
- [Comma.ai Calibration Challenge](https://github.com/commaai/calib_challenge)
- [OpenCV Optical Flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [MLX](https://github.com/ml-explore/mlx)
