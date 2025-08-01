# Benchmarks

This directory contains benchmark scripts to evaluate and optimize the different components of the system.

## Benchmark Scripts

### **`bayesian_search.py`**
Hyperparameter optimization framework using Bayesian search (`skopt`) to find the best filtering and processing parameters.

### **`benchmark_adam_plateau.py`**
Evaluation of early stopping parameters for the Adam optimizer to fine-tune its convergence criteria.

### **`benchmark_lbfgs_vs_adam.py`**
Performance comparison between the L-BFGS-B and Adam optimizers for epipole estimation.

###  **`benchmark_loading.py`**
Measures the time required to load pre-computed optical flow data, testing different storage formats like `.npy` and `.npz`.

### 🎯 **`benchmark_sampling_mean_representativity.py`**
Evaluates how well the mean error of a small, stratified sample of frames represents the error of the entire dataset.

### **`colinearity_score_per_norm.py`**
Analysis of flow vector collinearity, binned by vector norm, to understand the relationship between vector magnitude and its usefulness.

###  Norms Distribution **`flow_norms_distribution.py`**
Analysis and visualization of flow vector norm distributions to inform filtering strategies.

### **`perte_precision_quantization.py`**
Evaluation of precision loss (angular error) when quantizing optical flow vectors from float32 to float16.
