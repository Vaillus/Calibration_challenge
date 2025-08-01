# Source Code

This directory contains all the Python source code for the project, organized into several key packages.

## Directories

### **`core/`**
Contains the fundamental modules that form the backbone of the epipole estimation pipeline. This includes scripts for optical flow generation, object segmentation, vector filtering, and the core optimization logic.

### **`experiments/`**
Houses all scripts related to experimentation, analysis, and visualization. It is split into two main subdirectories:
- **`benchmarks/`**: Scripts for performance testing, hyperparameter optimization, and comparing different algorithms (e.g., optimizers, data loaders).
- **`visualizations/`**: A comprehensive collection of scripts to generate plots, animations, and figures for data analysis and documentation.

### **`utilities/`**
A collection of helper scripts and support tools used across the project. This includes functions for data loading (ground truth, flows, predictions), path management, data generation, and evaluation.

### **`production/`**
Contains scripts for running the final, optimized pipeline to generate predictions on new data.

### **`unit_tests/`**
Includes unit tests to ensure the correctness and reliability of individual components.

### **`notebooks/`**
Jupyter notebooks used for exploratory data analysis and iterative development.
