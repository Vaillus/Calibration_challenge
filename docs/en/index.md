---
title: "Onboard Camera Calibration in a Semi-Autonomous Car"
layout: dynamic_toc
mathjax: true
lang: en
---

# Introduction
Welcome to my blog! I recently undertook to solve Comma.ai's camera calibration challenge, a fascinating computer vision problem in the world of semi-autonomous driving. My approach ultimately allowed me to reach Xth place in the ranking, and I wanted to share this experience. This article therefore traces my journey, explaining step by step how I went about solving this technical challenge.

## Context
Comma.ai is a company that seeks to democratize autonomous driving. Where Tesla sells complete cars, comma.ai develops Openpilot: an open-source system that transforms your existing car into a semi-autonomous vehicle.

It's a sort of Android equivalent facing iOS, but for cars.

They offer a [set of public challenges](https://comma.ai/leaderboard) on their github, with a prize for the first person managing to solve the challenge, and a public ranking that is maintained over time.

One challenge in particular caught my attention, although it was published a few years ago and the prize has long since been won. This is the onboard camera calibration challenge in a semi-autonomous car.

## The Problem to Solve
In cars equipped with the Openpilot system, a dedicated comma.ai device (like the comma 3X) serves as the main camera. Unlike Tesla where cameras are fixed at precise positions in the factory, each Openpilot installation is unique: the device can be placed at different positions on the windshield, with different orientations. For the driving assistance system to work correctly, it must understand how the device and its cameras are oriented relative to the car. This is called camera calibration.

<figure>
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*_oAenBeOAbrtmOOuVVnSfg.jpeg" alt="Comma.ai device in the cockpit" style="width: 90%;" />
  <figcaption>Example of a comma.ai device positioned in a car's cockpit</figcaption>
</figure>

## The Objective
This challenge asks to develop an algorithm that, from a video taken by the comma.ai device during driving, can determine in which direction the car is moving relative to the camera's orientation.

To describe this direction of movement precisely in the camera's reference frame, I must predict two key angles for each image in the video:
- **Pitch (φ)**: The vertical angle between the camera's axis and the direction of movement. The observable pitch (φₒ) is influenced by:
	- The car's vertical movements (braking, acceleration, speed bumps)
	- The camera's fixed vertical orientation relative to the car
	- φ > 0: the car accelerates / goes over a speed bump
	- φ < 0: the car brakes / goes down a speed bump
- **Yaw (θ)**: The horizontal angle between the camera's axis and the direction of movement. The observable yaw (θₒ) is influenced by:
	- The car's trajectory (right or left turns)
	- The camera's fixed horizontal orientation relative to the car
	- θ > 0: the car turns right
	- θ < 0: the car turns left

### The Epipole
In computer vision, the point toward which the car is heading is called the "**epipole**". It's the point where the trajectories of stationary objects converge when the camera moves in a straight line.
With the given focal distance of 910 pixels, we can establish a direct relationship between the angles (pitch and yaw) and the coordinates (x,y) of this point in the image.

## Available Data
To solve this problem, I have access to 10 one-minute videos, each with approximately 1200 frames.
5 videos are labeled with the correct angles already identified, and 5 videos are unlabeled.
Each video shows different driving conditions (environment, lighting, etc.)

<figure>
  <img src="../imgs/intro/videos.gif" alt="Overview of the 9 videos in the dataset" style="width: 90%;" />
  <figcaption>Overview of 9 videos in the dataset</figcaption>
</figure>

## Evaluation Criterion
Predictions are evaluated on a scale where 0% corresponds to a perfect prediction and 100% corresponds to the score obtained by simply predicting the center of the image. The higher the score, the greater the error.

## Considered Strategies
By analyzing the challenge leaderboard, I noticed a clustering of scores around 20%. This phenomenon strongly suggests a ceiling for classic neural network (NN) approaches, probably limited by the small amount of training data available (only 5 labeled videos).

I therefore considered three main approaches:
1. **Deep Learning**: Although this approach is intuitive for computer vision problems, the data limitation made me doubt its ability to surpass the ceiling observed on the leaderboard.
2. **SLAM (Simultaneous Localization and Mapping)**: Mentioned in the problem statement as a validation method.
3. **Optical Flow**: Technique that estimates the apparent movement of objects between two consecutive images by calculating a displacement vector field. Simple and interpretable method.

My philosophy for this project was to prioritize simplicity and rapid iteration. I prefer to start by understanding the problem with more transparent methods before adding complexity if necessary.

Optical flow presents several advantages that convinced me to explore this path first:
- The method is **interpretable**: we can visualize the motion vectors and intuitively understand what's happening
- It's **relatively simple to implement** with libraries like OpenCV

Most importantly, there's a direct conceptual link between optical flow and the epipole that makes this approach particularly promising. To understand this relationship, we need to imagine what happens visually when the car moves forward in a straight line: the stationary objects in the environment (trees, buildings, signs) seem to "flow" backward in our field of vision, creating a perspective effect where all these elements appear to diverge from a central point.

This central point from which everything seems to diverge is precisely the epipole: the point toward which the car is heading. Optical flow, by calculating the apparent displacement vectors of each element between two consecutive images, mathematically captures this visual phenomenon. In theory, when the camera moves in a straight line, all optical flow vectors of stationary points in the environment point in directions that move away from the epipole.

This fundamental relationship makes optical flow a natural tool for locating the epipole.

# 1st Arc: Optical Flow
To start my exploration, I first wanted to establish a baseline with the most direct and intuitive approach possible. In this arc, I introduce optical flow - a fundamental technique that will remain at the heart of all my approaches throughout this project.

## Optical Flow Implementation
Now that the conceptual link between optical flow and epipole is established, I needed to choose how to concretely implement this approach to exploit this geometric relationship.

The `opencv` package offers two main methods for calculating optical flow:
- **cv2.calcOpticalFlowFarneback()**: called "dense", this method calculates optical flow for all pixels in the image. It's particularly suited for measuring continuous and global movements, but is more computationally expensive.
- **cv2.calcOpticalFlowPyrLK()**: called "sparse", this method only calculates optical flow for specific points previously identified (generally corners or interest points). It's faster but requires pertinent selection of points to track.

Having no strong time computation constraint and wishing to measure the global relative movement of the environment relative to the vehicle, I opted for the dense approach with Farnebäck's algorithm. This method allows me to obtain a complete vector field that represents the apparent movement between two consecutive frames.

### Visualization
I therefore implemented the calculation of dense optical flow between consecutive frames, which produces a vector field as illustrated below:

<figure>
  <img src="../imgs/1/flow_vector_example.png" alt="Example of optical flow vector field" style="width: 90%;" />
  <figcaption>Visualization of dense optical flow vector field calculated between two consecutive frames</figcaption>
</figure>

In this visualization, each arrow represents the apparent displacement of a pixel between two frames. We can observe that in a straight-line movement scenario, these vectors seem to diverge from a particular point - this should be our epipole.

## First Approach for Epipole Estimation

From the optical flow vector field obtained, I sought to locate the epipole using an intuitive method based on analyzing direction changes of the vectors.

The method relies on a fundamental observation: in straight-line movement, optical flow vectors tend to "spread out" from the point toward which the vehicle is heading (the epipole). I therefore sought to locate this convergence point by separately analyzing the horizontal and vertical components of the flow.

The idea is simple: the epipole corresponds to the place where vectors change direction, both horizontally and vertically. Vertically, above the epipole, vectors point mainly upward, while below, they point downward. Similarly horizontally, to the left of the epipole, they point left, and to the right, they point right. The intersection of these two direction change lines should therefore give an approximate estimation of the epipole.

<figure>
<div style="display: flex; justify-content: space-between;">
  <img src="../imgs/1/sep_vertical.png" style="width: 48%;" />
  <img src="../imgs/1/sep_horizontal.png" style="width: 48%;" />
</div>
  <figcaption>Vertical and horizontal separation of optical flow vectors</figcaption>
</figure>

Let's isolate the method for finding the vertical separation axis:

1. For each column $j$, I calculate the mean $m_j$ of the horizontal components of the vectors:
   $$m_j = \frac{1}{H} \sum_{i=1}^{H} x_{i,j}$$
   where $H$ is the image height and $x_{i,j}$ is the horizontal component of the vector at position $(i,j)$.

2. For each potential separation position $s$, I calculate the difference between the means on the right and left:
   $$\delta(j') = \sum_{j=j'+1}^{W} m_j - \sum_{j=1}^{j'} m_j$$
   where $W$ is the image width.

3. I select the position $j^*$ that maximizes this difference:
   $$j^* = \arg\max_j \delta(j)$$

By applying this method for both horizontal and vertical axes, I obtain the coordinates of my first epipole estimation by calculating the intersection point between the two separation lines.

## Results and Limitations
With this very simple method, I obtain my baseline - a score of **1960.20%**. If you were wondering whether it's possible to do worse than 100%, you have your answer!

Although the figures presented a bit earlier show a relatively clear separation on certain frames, this method fails completely on many other situations. In these problematic cases, the separation lines end up stuck to the image edges, producing aberrant predictions.

I notably identified several elements that handicap epipole prediction with this technique, such as the car's hood and other moving vehicles in the scene. These are precisely the problems I will address in the next section.

<figure>
  <img src="../imgs/1/final_viz.gif" alt="Example prediction with arc 1 method" style="width: 90%;" />
  <figcaption>Example prediction with arc 1 method</figcaption>
</figure>

# 2nd Arc: Segmentation
## Problems Identified with the Previous Method
With the previous method based only on optical flow, certain frames give acceptable results, but the whole is very noisy. In some cases, the algorithm fails completely as we can see in these examples:

<div style="display: flex; justify-content: space-between;">
  <img src="../imgs/2/sep_real_2.png" style="width: 48%;" />
  <img src="../imgs/2/sep_heatm_2.png" style="width: 48%;" />
</div>

**Example 1 - Moving vehicles:** In this first case, a van is overtaking the car, creating vectors oriented to the right while they are located on the left of the screen (the red spots on the left of the left figure). The vertical separation line then ends up stuck to the left extremity, producing an aberrant estimation.

<div style="display: flex; justify-content: space-between;">
  <img src="../imgs/2/sep_real_1.png" style="width: 48%;" />
  <img src="../imgs/2/sep_heatm_1.png" style="width: 48%;" />
</div>

**Example 2 - The car's hood:** We observe here that the car's hood reflects the scenery (the light blue spots at the bottom of the right figure), creating vectors that point upward while they are located at the bottom of the image. This artificially pushes the horizontal separation line downward. Although less pronounced than in the first example, this also distorts the epipole estimation.

The solution becomes obvious: we need to segment and ignore both the car's hood and the mobile elements of the scenery (other vehicles, pedestrians) likely to distort our estimation.

## Choice of Segmentation Methods
After brief research, I identified two promising algorithms for segmentation:

### SAM 2 from Meta
SAM 2 from Meta is a very general segmentation model, capable of handling almost any object. However, it's very slow and resource-demanding, making it unsuitable for real-time applications or use on personal machines.

I was initially attracted by SAM 2's capabilities, but quickly became disillusioned: even the lightest versions of the model were difficult to load on my Mac's M1 chip, and the "large" version still hadn't finished loading after 20 minutes of waiting. This hardware constraint, which I had imposed on myself for this project (using only my personal Mac), pushed me toward an alternative.

### YOLOv8-seg
This model presents several advantages:
- Fast and efficient
- Excellent detection of vehicles and other common objects
- Compatible with my hardware constraints

I therefore opted for YOLOv8-seg, which proved fast and generally efficient for vehicle detection and segmentation:

<figure>
  <img src="../imgs/2/yolo_seg.png" alt="YOLOv8-seg" style="width: 90%;" />
  <figcaption>Example of vehicle detection and segmentation with YOLOv8-seg</figcaption>
</figure>

I configured YOLO to segment only vehicles, excluding static scenery elements like traffic lights, although it's also capable of detecting them.

### Manual Hood Segmentation
A problem persisted however: YOLOv8 doesn't detect the car's hood, since it's not trained for this specific task. I therefore quickly developed a simple interface allowing me to manually segment the hood on the first frame of each video. This manual segmentation is then applied to all frames of the corresponding video.

## Results and Performance Improvement
The application of this combined segmentation (YOLOv8 for mobile objects + manual hood segmentation) reduced the error score to **812.57%**, representing approximately 60% improvement compared to the method based only on optical flow.

<figure>
  <img src="../imgs/2/final_viz.gif" alt="Example prediction with arc 2 method" style="width: 90%;" />
  <figcaption>Example prediction with arc 2 method</figcaption>
</figure>

Although this score remains very high and far from satisfactory, this improvement confirms the importance of segmentation in our approach. This preprocessing step will be preserved and used in all subsequent methods.

# 3rd Arc: New Method for Epipole Estimation
## Principle of Collinearity Score
In this new approach, I directly tackle the fundamental problem: how to find the convergence point of optical flow vectors (the epipole) more precisely?

When the vehicle in which the camera is embedded moves in a straight line, all stationary objects seem to "flow" from a unique point - the epipole. This phenomenon creates a vector field with an essential geometric property: the optical flow vectors of stationary objects point in directions that move away from the epipole.

Since the epipole's position is unknown, the approach consists of testing several "candidate points". For each candidate, we calculate a global collinearity score to evaluate its plausibility.

For each pixel p with an optical flow vector $\vec{v}(p)$, and a candidate point e, I calculate:
- The normalized optical flow vector:  
$$\hat{v}(p) = \frac{\vec{v}(p)}{\|\vec{v}(p)\|}$$
- The normalized vector going from the pixel to the candidate epipole:  
$$\hat{d}(p) = \frac{e - p}{\|e - p\|}$$    
- The individual collinearity score for this pixel:
$$s(p) = \hat{v}(p) \cdot \hat{d}(p)$$

This dot product equals 1 if the vectors are perfectly aligned, 0 if they are perpendicular, and -1 if they point in opposite directions.

<figure>
  <img src="../imgs/3/collinearity_concept.gif" alt="Collinearity score" style="width: 90%;" />
  <figcaption>Example of collinearity score calculation for a pixel</figcaption>
</figure>

The global collinearity score for a candidate point e is then the average of these individual scores:
$$S(e) = \frac{1}{|P|} \sum_{p \in P} s(p)$$
Where P is the set of pixels in the image.

<figure>
  <img src="../imgs/3/global_collinearity_score.gif" alt="Global collinearity score" style="width: 90%;" />
  <figcaption>Example of global collinearity score calculation for a candidate point</figcaption>
</figure>

Since optical flow vectors should point in the direction opposite to the vector going from the pixel to the epipole (for stationary objects), the objective is to find the point $e^*$ that minimizes this score:
$$e^* = \arg\min_{e} S(e)$$

This method transforms our problem into an optimization task: find the point that minimizes the global collinearity score.

## Optimization Approach
To optimize my estimation of the epipole position on the image, I therefore need to minimize the global collinearity score.

### Preliminary Vector Filtering
During my initial observations of optical flow vector fields, I noticed an important phenomenon: large amplitude vectors generally point in the direction opposite to the epipole in a coherent manner, while very small norm vectors present much more noisy behavior.

To improve the robustness of my estimation, I therefore decided to apply simple filtering: eliminate all vectors whose norm is less than $10^{-2}$. This threshold was chosen somewhat arbitrarily from my observations. I decided to look at this more closely in the next iteration of the project.

### Optimization Method

By visualizing the **global collinearity score** (defined previously) for different candidate positions, I observe that the objective function to minimize presents a generally **convex** form, making it ideal for optimization.

<figure>
  <img src="../imgs/3/convex_function.png" alt="Global collinearity score" style="width: 90%;" />
  <figcaption>Visualization of the global collinearity score for candidate positions covering the entire image. The "bowl" shape is characteristic of a convex function.</figcaption>
</figure>

This property comforted me in choosing a **gradient-based optimization method** to find the minimum efficiently.

Although classic gradient descent is an option, I chose a more sophisticated approach: the **L-BFGS-B** algorithm (Limited-memory BFGS with Bounds). This is a **quasi-Newton** method that offers an excellent performance/cost compromise:
It converges in much fewer iterations than simple gradient descent for this type of problem, although each iteration is more complex.

## Optical Flow Parameters
Optical flow vectors constitute the raw data used by all subsequent optimization methods. It therefore seemed essential to optimize their quality to improve the entire epipole estimation process.

For this, I optimized the parameters of OpenCV's `cv2.calcOpticalFlowFarneback` function with a fairly simple grid search. The objective was to find parameters that improve the global collinearity score for the points provided in the labels.

After a series of tests, I identified a satisfactory configuration that improves the criterion without excessively increasing computation time. The main adjustments concern the scale pyramid (`pyr_scale`, `levels`), the analysis window size (`winsize`) and polynomial smoothing parameters.

## Results of this New Approach
This method gives a performance of **168.83%**, which represents approximately 80% improvement compared to the previous iteration.

This significant improvement validates the combined approach: new collinearity criterion, gradient descent optimization method, and optimized optical flow parameterization.

<figure markdown>
  <img src="../imgs/3/final_viz.gif" alt="Prediction GIF" style="width: 90%;" />
  <figcaption>Example prediction with arc 3 method</figcaption>
</figure>

That's progress!

# 4th Arc: Vector Filtering
## Rethinking Filtering
In the previous iteration of the project, I arbitrarily chose to filter vectors and keep only those whose norm is greater than $10^{-2}$ because I observed that smaller vectors tended to be more noisy.
My intuition led me to think that digging in this direction would lead to significant performance improvement.

### Considered Filtering Criteria
I identified three promising criteria for improving vector selection:
1) **Vector norms**: Based on the observation mentioned previously - low amplitude vectors are indeed more noisy.
2) **Collinearity score with the center**: Even among large vectors, some point in aberrant directions that noise the estimation. Since the epipole generally remains close to the image center, a vector that doesn't "point" roughly toward the direction opposite to the center has little chance of being informative. The collinearity score with the center therefore constitutes a good proxy for identifying useful vectors.
3) **Distance to center**: Intuition that there might exist a correlation between a vector's distance to the center and its ability to contribute positively to epipole estimation.

For this first exploration, I retained the first two criteria - they seemed more promising and two parameters already constitute a good starting point.

### Filtering Strategies
I then considered two main approaches:

**Adaptive vs general filtering**: Use machine learning to find specific parameters for each frame, or find parameters that work well on average across all frames. With little training data available, I opted for simplicity: general parameters.

**Hard filtering vs weighting**: Hard filtering completely eliminates certain vectors according to binary criteria (ex: all those with norm > 0.01), while weighting assigns them variable weights (ex: linear coefficient relative to their collinearity score with the center). I chose to start with hard filtering first - simpler to implement and interpret.

### Retained Approach
My strategy is therefore clear: hard filtering with general parameters, applied on vector norms and their collinearity score with the center. The next step consists of finding the optimal values of these parameters.

## Acceleration of Epipole Estimation
### Motivation
I did preliminary tests on individual frames and observed that optimal parameters for filtering differ from one frame to another. I therefore need to be able to evaluate the impact of tested filters on all frames to find parameters that work well on average.

**The problem**: Evaluating the epipole on all frames of the 5 videos takes 1 to 2 hours. If I want to evaluate the impact of different filtering parameter sets, I can only test 12 to 24 combinations per day by running my computer continuously. For systematic exploration requiring hundreds of evaluations, this isn't a feasible solution.

**The objective**: Reduce evaluation time to ~10 seconds to allow exploration of a large parameter space with advanced search methods.

I therefore need to find a way to drastically accelerate the evaluation of filtering parameters' impact on epipole estimation.

### Module-by-Module Optimization
#### Return to the Prediction Pipeline
To find the epipole on a frame, my current method follows a sequential pipeline in three modules:

<figure>
  <img src="../imgs/4/pipeline.png" alt="Pipeline" style="width: 90%;" />
  <figcaption>Epipole prediction pipeline</figcaption>
</figure>

**Module 1: Optical Flow Generation** - From two consecutive frames, I calculate the optical flow vector field with OpenCV's Farnebäck algorithm, then apply segmentation to eliminate vectors corresponding to moving vehicles and the car's hood.

**Module 2: Vector Filtering** - I apply selection criteria (norm threshold, collinearity score) to keep only the most informative vectors for epipole estimation.

**Module 3: Optimization** - I apply gradient descent to minimize the collinearity score presented in the third arc, which gives me the final coordinates of our epipole estimation for a frame.

In this section, we'll follow my path aimed at accelerating each of these three modules.

#### Module 1: Optical Flow Vector Field Generation
**Pre-computation of Vector Fields**
The most obvious optimization consists of pre-computing all optical flow vector fields rather than recalculating them at each evaluation of filtering parameters.

**Storage Challenges**
This strategy quickly confronted me with a practical problem: the 5 generated files each weighed 10 GB (default float32 encoding), completely saturating my Mac's memory.

**Solution: Quantization and Compression**
I therefore explored vector quantization to reduce memory footprint. To evaluate the impact of this quantization, I calculated the angular error introduced by going from float32 to float16 on all vectors, then analyzed this error as a function of vector sizes.

The result is reassuring: only vectors belonging to the smallest deciles of the norm distribution (i.e., very low amplitude vectors) present an angular error greater than 1°. However, these low amplitude vectors are precisely those that are the most noisy according to my observations, and therefore less informative for epipole estimation. The error introduced by quantization should therefore be negligible compared to the noise already present in these vectors.

By combining this quantization with .npz compression, I obtained 5 files of ~3 GB each, a 70% reduction in storage space. However, I finally kept the float32 versions to prioritize performance and used the .npy format to avoid decompression time.

**Result**
This optimization completely eliminates the computation time of module 1 during parameter testing, allowing direct transition to filtering pre-computed vectors.

#### Module 2: Filtering
**The Performance Challenge** Now that I had these pre-computed optical flow tensors (one tensor per video, dimensions: n_frames × height × width × 2), I needed to parallelize vector filtering. These operations, initially processed frame by frame, represented a major bottleneck in my pipeline when applied to thousands of frames.

**Hardware Constraints and Technological Choices** Working on a Mac M1 Pro, I was confronted with a known limitation: JAX (an accelerated numerical computation library) doesn't efficiently exploit this architecture's GPU. Two options were available to me:
- Look for an external server for my computations
- Find a library better adapted to my chip

Out of curiosity and to test my hardware's capability limits, I opted for the second option. My research led me to **MLX**, the library developed by Apple specifically for their chips. The first tests were convincing: a notable acceleration compared to sequential calculations.

**Parallelization Strategy for Filtering** I implemented two-level parallelization specifically to optimize vector filtering:
- **Parallelization on pixels**: The collinearity score calculation, which initially took 1 second per frame in sequential mode, was reduced to 3 milliseconds thanks to vectorized calculation on GPU. This 300-factor improvement transformed a critical operation into quasi-instantaneous computation.
- **Parallelization on frames**: By processing videos in batches adapted to GPU capacity, I could calculate collinearity scores on all frames of a video in 3-4 seconds versus several minutes previously.

Great victory on accelerating processing speed on this module.

#### Module 3: Optimization
**A Critical Module for Evaluation**
This third module occupies a particular position in my pipeline: being last, it's the one that produces the final epipole prediction. Consequently, it's also at this level that I evaluate the impact of each filtering parameter set on result quality. To efficiently optimize my parameters, I must be able to quickly test thousands of different combinations.

**The L-BFGS-B Bottleneck**
Until now, I used the L-BFGS-B algorithm from `scipy.optimize` for epipole estimation. This method worked well individually, but presented a major problem for my optimization objective: **it's not parallelizable**. Evaluating the impact of different filtering parameters on hundreds of frames would require sequential calculations lasting hours.

Faced with this limitation, I briefly considered a radical alternative: completely abandon gradient descent and directly evaluate filtering quality via collinearity score at the label level. Although seductive by its ease of parallelization, this approach presented too important a conceptual risk - nothing guaranteed that good local collinearity scores would lead to correct convergence of global optimization.

**The Adam Bet**
I therefore opted for implementing Adam from scratch, motivated by two main factors: my **familiarity with this algorithm** (easily interpretable and debuggable) and its demonstrated **robustness** on a wide range of problems. My hope was to be able to parallelize this implementation with MLX.

**Parallelization Failure and Pragmatic Choice**
This strategy ran into MLX's technical limitations: scalarity constraint for gradient calculation, absence of native Jacobian, and inefficiency of automatic vectorization. I briefly considered implementing the Jacobian calculation myself, but for time concerns, I resigned myself to maintaining a sequential approach.

However, my preliminary tests on a few individual frames showed that Adam converged satisfactorily with a generally convex cost function. I therefore decided to keep this implementation despite the absence of parallelization, making module 3 my main bottleneck in the optimization pipeline.

**Parameterization and Optimized Early Stopping**

Having failed to parallelize gradient descent, my main objective was to **reduce computation time** of the optimization module. Starting from a simple approach of **50 fixed iterations**, I implemented **early stopping based on plateau detection**: optimization stops when the cost function doesn't improve by more than 1e-4 for 3 consecutive iterations.

This modification effectively **reduced computation time by 3 to 5 times**. But unexpectedly, I also observed a **performance improvement** in terms of error relative to labels.

By analyzing this surprising phenomenon, I understood that converging to the bottom of the convex function often moved the prediction away from the image center. In cases where the optimization direction isn't perfectly aligned with the real epipole, stopping prematurely avoids moving too far away and keeps the prediction closer to the label.

This premature stopping strategy therefore proved doubly beneficial: computation acceleration AND precision improvement.

<figure>
  <img src="../imgs/4/optimizer_comp.png" alt="Prediction GIF" style="width: 90%;" />
  <figcaption>Trajectories of different optimizers for one frame</figcaption>
</figure>

At this stage, I prioritized exploration speed: quickly validate that the approach worked before refining details. This pragmatic strategy proved sufficient to move to the next step, where more rigorous evaluation would become necessary.

#### Conclusion
This approach gave mixed results. Work on **module 1** effectively reduced computation times significantly by eliminating redundant recalculations. On the other hand, gains obtained on **module 2** (filtering parallelization) are largely diminished by the bottleneck encountered in **module 3**: the impossibility of parallelizing gradient descents forces sequential processing that limits the impact of previous optimizations.

Nevertheless, at this stage evaluating a parameter set still requires 2 to 3 minutes of computation on all frames and isn't sufficient to quickly test a large number of parameter combinations. To efficiently explore the parameter space without waiting days, I needed to be more creative and find an alternative approach to the problem.

### Intelligent Sampling Strategy
To accelerate parameter space exploration, I opted for a strategic sampling approach. The key was to build a sampled subset as representative as possible of the original set for a minimal number of frames. For this, I proceeded in three steps:

1. **Error characterization**: I first calculated prediction errors on all frames using simple filtering parameters
2. **Performance stratification**: I then obtained the distribution of these errors for each video and organized frames into deciles according to their error level
3. **Balanced sampling**: I selected 2 frames per decile and per video, thus obtaining a sample of 100 frames uniformly covering the difficulty spectrum

This strategy guarantees that the sample contains both "easy" frames (low error) and "difficult" frames (high error) despite its small size.

**Result**
Evaluating filtering parameters on these 100 carefully selected frames now takes only 2-3 seconds, finally allowing efficient exploration of the parameter space.

## Parameter Search and Results
### Parameter Search Strategy
Once I had a sufficiently fast evaluation method, I needed to choose a strategy to explore the filtering parameter space. Several approaches were available to me:

1. **Manual exploration**: Intuitive search based on result observation
2. **Exhaustive search**: Systematic sweep of a restricted parameter space
3. **Bayesian optimization**: Probabilistic approach to guide the search
4. **Metaheuristics**: Genetic algorithms or other evolutionary methods

I prioritized the first two options for their **implementation simplicity**, the fact that they are potentially sufficient to obtain good results at this stage of the project, and especially because **the search space is limited to only two parameters** (collinearity threshold and norm threshold), making exhaustive exploration and visual analysis perfectly feasible.

**My strategy**: start with a broad systematic sweep (approach 2), then manually refine the zone identified as promising (approach 1). This combination, although simple and subject to missing local minima, seemed to be a good compromise.

### Results
Parameter space exploration allowed identifying the following values for filtering parameters:
- **Collinearity threshold**: 0.96
- **Norm threshold**: 13

<figure>
  <img src="../imgs/4/effet_filtrage.png" alt="Filter comparison" style="width: 90%;" />
  <figcaption>Comparison of optical flow vector fields for different filtering parameters</figcaption>
</figure>

These parameters produced a **score of 54.32%**, representing a significant 60% improvement compared to the previous iteration. This performance marks entry into an acceptable result range, while maintaining substantial improvement potential for future optimizations.

<figure markdown>
  <img src="../imgs/4/final_viz.gif" alt="Prediction GIF" style="width: 90%;" />
  <figcaption>Example prediction with arc 4 method</figcaption>
</figure>

# 5th Arc: Filtering Improvement and Post-processing
## Part 1: Pipeline Improvements
### Filter
#### Sigmoid
In the previous arc, I had opted for "hard" binary filtering on optical flow vectors: vectors whose norm was less than 13 were simply eliminated from the calculation. Although this approach demonstrated its effectiveness by producing satisfactory results, it limited the flexibility of filtering methods I could test.

This rigidity led me to explore a more nuanced approach combining filtering and weighting. The objective was to find a mathematical function general enough to express a wide range of filtering strategies, while maintaining a manageable number of parameters for optimization.

The sigmoid function emerged as the ideal solution:
$$
sig(x, θ, α) = \frac{1}{1+e^{-α(x-θ)}}
$$
where:
- θ: sigmoid threshold/center
- α: steepness/slope of transition

<figure>
  <img src="../imgs/5/sigmoid.png" alt="Sigmoid function" style="width: 90%;" />
  <figcaption>Sigmoid function</figcaption>
</figure>

This formulation presents several decisive advantages.
On one hand, it naturally encompasses extreme cases: a very high parameter $k$ reproduces classic binary filtering (e.g., the "hard" filter mentioned previously is expressed by $sig(x,13,\infty)$), while a low value of $k$ generates linear weighting.
On the other hand, it's limited to only two parameters to optimize, thus preserving search space tractability.

<figure>
  <img src="../imgs/5/sigmoids.gif" alt="Sigmoid functions" style="width: 90%;" />
  <figcaption>Sigmoid function for different threshold and steepness parameters</figcaption>
</figure>

This generality allows unified exploration of different filtering strategies, considerably simplifying the experimentation process.

**Mathematical Formulation of Weighted Filtering**

For each optical flow vector $v_i$, I evaluate two characteristics:
1. **The vector's norm**: $\|v_i\|$ 
2. **The collinearity score with the reference point**: $c_i$

Each vector $v_i$ is then transformed according to:

$$v'_i = v_i \cdot sig(\|v_i\|, θ_{norm}, α_{norm}) \cdot sig(c_i, θ_{col}, α_{col})$$

This approach unifies hard filtering and soft weighting in a coherent mathematical framework with only 4 parameters to optimize.

#### Improvement of Collinearity Criterion

To recall, in the previous arc, I calculated for each optical flow vector a collinearity score with the image center as reference point to ensure that retained vectors point roughly in the direction opposite to the image center.

Using the image center as reference point for calculating collinearity score in vector filtering presents a conceptual limitation: **this reference point isn't optimal**. Taking the average of predictions from a previous experiment as reference point should naturally improve filtering, since this point will be closer to the actual predictions to be made.

**The problem with the fixed center:**
Beyond this general consideration, I observed a specific problematic case: when the real epipole is far from the image center, vectors located between the epipole and the center can be incorrectly filtered. These vectors correctly point in the direction opposite to the epipole, but get a bad collinearity score with the center and are therefore excluded.

**The solution:**
Rather than systematically using the image center as reference point, I give myself the **possibility of using the average point of predictions from a previous generation** for each video. The method consists of:

1. Performing a first pass of predictions (regardless of the reference point used for collinearity calculation)
2. Calculating the average point of all epipole estimations obtained for each video
3. Using these new average points as reference for calculating collinearity score during vector filtering in a new experiment

This approach allows in principle to improve vector filtering, since the reference point is closer to the actual predictions to be made.

#### Collinearity Heatmap

At this stage of the project, a question intrigued me: **do all pixels in the image contribute equally to estimation quality?** My intuition was that certain zones might systematically provide more informative vectors - perhaps due to scene geometry or recurring movement patterns.

To explore this hypothesis, I decided to create a "heat map" (heatmap) showing which image regions historically produce the best collinearity scores. The objective was twofold: first understand if privileged zones actually exist, then potentially exploit this information to refine filtering by giving more weight to the most informative regions.

But a methodological question arose: **relative to which reference point should these collinearity scores be calculated?** This question naturally led me to explore two complementary approaches.

##### Two Spatial Analysis Strategies
**Approach 1: Absolute heatmap (fixed coordinates)**
In this first approach, I calculate the average collinearity of vectors from each pixel relative to the **fixed image center** (width/2, height/2).

For each absolute position `(x,y)` in the image:
- I calculate the collinearity score relative to the fixed center for all vectors located at this position
- I average these scores across all frames and all videos
- Result: "The pixel located at absolute position `(x,y)` has on average a collinearity score of X"

**Approach 2: Relative heatmap (coordinates centered on epipole)**
The first approach had an obvious limitation: it analyzed patterns relative to the fixed image center, while the average epipole can vary from one video to another. In this second approach, I calculate collinearity scores relative to the **average epipole of each video** rather than the fixed center.

For each position relative to the epipole:
- I transform coordinates: `relative_position = (x,y) - video_average_epipole`
- I calculate collinearity relative to the specific epipole of each video
- I average scores by **relative position**, not by absolute position
- Result: "A pixel located 50px to the right of the epipole (regardless of where this epipole is in the image) has on average a collinearity score of Y"

This approach aimed to discover recurring geometric patterns around the epipole - for example, "are vectors located 100 pixels bottom-right of the epipole systematically more informative?" - independently of the epipole's absolute position in each video.

##### Obtained Results
**Absolute heatmaps per video (Approach 1):**
In the following image we can see the average collinearity scores per video in absolute coordinates:

<figure>
  <img src="../imgs/5/abs_heatm_per_video.png" alt="Absolute heatmap per video" style="width: 90%;" />
  <figcaption>Absolute heatmap per video</figcaption>
</figure>

Observations:
- Patterns repeat: at the bottom of each image, a white region corresponds to the car hood mask where scores weren't calculated
- Yellow rays are clearly distinguished under the central point of images 0, 1 and 4
- Images 2 and 3 are different: image 2 presents a white region corresponding to a masked vehicle throughout the video, image 3 has no clear pattern

**Global absolute heatmap (Approach 1):**
In the following image, we can see the average collinearity scores per absolute pixel, across all videos:

<figure>
  <img src="../imgs/5/abs_heatm_global.png" alt="Global absolute heatmap" style="width: 90%;" />
  <figcaption>Global absolute heatmap</figcaption>
</figure>

A clear pattern similar to that appearing on images 0, 1 and 4 appears on the global image.

**Global relative heatmap (Approach 2):**
I also implemented and tested this approach to see if it would reveal more informative patterns than the absolute approach:

<figure>
  <img src="../imgs/5/rel_heatm_global.png" alt="Global relative heatmap" style="width: 90%;" />
  <figcaption>Global relative heatmap</figcaption>
</figure>

We observe that patterns are much less clear than in the absolute approach. Consequently, I decided to keep the global absolute heatmap approach for the continuation.

##### Using Absolute Heatmap for Filtering
I implemented the use of this absolute heatmap as coefficient mask during vector filtering. The idea is to give more importance to vectors located in regions that historically have good collinearity scores, and less importance to those in less informative regions.

Concretely, I introduced a **weighting parameter** between 0 and 1 that controls the heatmap's influence: 0 completely ignores the heatmap, 1 gives it maximum influence. This allows testing different degrees of spatial information exploitation in filtering.

### Optimizer
By analyzing estimations obtained at the end of the fourth arc, I observed something: on frames where the gap between the predicted point and the labeled point was greatest, the error notably came from the optimizer that couldn't reach the minimum of the collinearity function. My stopping criterion based on cost function improvement (stopping when improvement becomes less than 1e-4 for 3 consecutive iterations) was simply too restrictive.

Rather than trying to find the right threshold value for the stopping criterion, I decided to change the stopping criterion: stop optimization when the prediction hasn't moved more than 1 pixel during the last 5 iterations.

This approach presents several advantages:
- Alignment with objective: What ultimately matters is precise estimation to the nearest pixel.
- Preserved efficiency: Average computation time remains similar to the previous approach.
- In all observed cases, the prediction is very close to the global minimum.

## Part 2: Post-processing
Since the 3rd arc of this project, I observe that my epipole predictions, although improved at each iteration, remain very noisy and would probably benefit from post-processing smoothing. I decided to keep this optimization for the end of the project as "icing on the cake" for a small performance bonus.

#### Identification of Valid Frames

A fundamental problem arises during epipole estimation: **on certain frames, the vehicle is stationary or moving too slowly** to allow reliable estimation. In these situations, optical flow vectors have very small norms and are particularly noisy. With my current filtering criteria (norm threshold ≥ 13), the quasi-totality of these vectors are eliminated, not leaving enough information for optimization.

Faced with this lack of information, **my algorithm falls back on a default prediction: the image center**. This avoids aberrant predictions but doesn't reflect the vehicle's real movement. This approach is actually confirmed by analysis of labels provided in the problem: certain frames are associated with NaN values, corresponding precisely to situations where the vehicle is too slow to allow reliable epipole estimation.

**Definition of valid frames and notation**
For smoothing average calculations, I define the set of **valid frames** $V$ as the set of frames where the vehicle has sufficiently high speed to produce reliable epipole estimation (i.e., different from the image center).

To clarify smoothing calculations, I note $p_1, p_2, \dots, p_{\|V\|}$ the sequence of predictions from valid frames **ordered temporally**. Thus, $p_i$ corresponds to the $i$-th valid prediction in chronological order, and $p_{i-1}$ designates the valid prediction that immediately precedes it. These ordered predictions are the only ones used in smoothing calculations, thus avoiding artificially biasing results toward the screen center.

#### Implemented Smoothing Methods

I explored three smoothing approaches, from simplest to most sophisticated:

**1. Simple average**
The most direct method consists of calculating the arithmetic mean of all valid frame predictions, then assigning this average value to each valid frame:
$$\bar{p} := \frac{\sum_{i=1}^{|V|}{p_i}}{|V|}$$
$$\forall i \in \{1, ..., |V|\} : \tilde{p_i} = \bar{p}$$

This approach gives the same weight to all valid frames, independently of their temporal position, and assigns the same smoothed prediction to all these frames.

**2. Exponential average**
This method weights observations decreasingly over time, giving more importance to recent predictions:
$$\tilde{p}_1 = p_1 \quad \text{(first valid prediction)}$$
$$\forall i \in \{2, ..., |V|\} : \tilde{p}_i := \alpha \cdot p_i + (1- \alpha) \cdot \tilde{p}_{i-1}$$

where $\alpha \in [0,1]$ controls adaptation speed: a value close to 1 favors recent observations, while a value close to 0 maintains longer memory of past predictions.

**3. Bi-directional exponential average**
This approach combines the advantages of exponential smoothing in both temporal directions. For each valid prediction $p_i$, I calculate two separate exponential smoothings:

- A **forward** smoothing: $\tilde{p}_{i}^{forward}$ calculated by applying the exponential average method from position 1 to position $i$:
$$\tilde{p}_1^{forward} = p_1$$
$$\forall j \in \{2, ..., i\} : \tilde{p}_j^{forward} := \alpha \cdot p_j + (1- \alpha) \cdot \tilde{p}_{j-1}^{forward}$$

- A **backward** smoothing: $\tilde{p}_{i}^{backward}$ calculated by applying the exponential average method in reverse, from position $\|V\|$ to position $i$:

$$\tilde{p}_{|V|}^{backward} = p_{|V|}$$

$$ \tilde{p}_j^{backward} := \alpha \cdot p_j + (1- \alpha) \cdot \tilde{p}_{j+1}^{backward} \forall j \in \{|V|-1, ..., i\}$$

The final prediction combines these two estimations:
$$\forall i \in \{1, ..., |V|\} : \tilde{p_i}^{bi} := \frac{\tilde{p}_{i}^{forward} + \tilde{p}_{i}^{backward}}{2}$$

This method takes advantage of complete temporal information from the valid prediction sequence: each smoothed prediction benefits from both past and future context, which should theoretically produce more robust smoothing than previous approaches.

**Important note:** This bi-directional approach is obviously not applicable in a production context where epipole estimation must be done in real-time, since it requires knowing the entire future sequence. However, in the context of this project where we have complete videos and no real-time constraint, this method can potentially improve final results.

- [ ] Give the effect on results from the previous arc.

## Part 3: Optimal Parameter Search

### Parameter Space

At this stage of the project, my filtering pipeline has considerably evolved since the two-parameter method of the previous arc. The system now combines three successive filters:

**1. Vector norm filtering** (2 parameters): sigmoidal weighting with threshold and slope

**2. Collinearity filtering** (2 parameters + 1 binary): sigmoidal weighting with threshold and slope, and binary parameter determining choice of reference point (image center vs average estimated epipole of the video)

**3. Spatial weighting** (1 parameter): heatmap influence coefficient (0 to 1)

**Resulting parameter space:**
The parameter space now counts **6 total dimensions**, composed of 5 continuous parameters and 1 binary parameter. This increased complexity requires more sophisticated optimization methods than the search method used in the previous arc.

### Choice of Optimal Parameter Search Method

Faced with this intermediate-sized parameter space, I chose Bayesian search for finding optimal values for several reasons.

**Relevant to my case**: Evaluation remains relatively time-expensive, preventing me from exhaustively testing millions of parameters, and I expected non-linear interactions between parameters.

**Implementation simplicity**: Bayesian search with `skopt.gp_minimize()` offers an easily usable interface compared to metaheuristics that require a bit more parameterization.

**Exploration control**: The possibility of adjusting exploration/exploitation balance via acquisition functions allowed adapting strategy according to progress: broad exploration at the beginning, then exploitation of accumulated information at the end of the search.

### Search Process
#### Phase 1: Bayesian Exploration
I start with Bayesian search by fixing one parameter: the reference point used to calculate collinearity scores for filtering remains the image center. We therefore explore in the space of the five remaining parameters.

After about 1000 evaluations with an exploration then exploitation strategy, I identify a promising region in the parameter space.

At this stage, an important observation emerges: the collinearity heatmap doesn't bring significant improvement (we'll keep this parameter fixed at 0)

#### Phase 2: Local Refinement
To more exhaustively explore the identified region, I switch to local grid searches. I focus on the 4 remaining parameters: the two sigmoidal coefficients (slope and threshold) for norm and collinearity filters. I alternately optimize each filter to progressively refine parameters.

The identified optimal parameters are:
- **Norm filter**: α = 180, θ = 8
- **Collinearity filter**: α = 152, θ = 1.245
- **Reference point**: image center
- **Heatmap**: coefficient = 0.0

This gives sigmoidal filters that look like this:

<figure>
  <img src="../imgs/5/sigmoids_opti.png" alt="Sigmoid functions" style="width: 90%;" />
  <figcaption>Sigmoid functions for optimal parameters of norm and collinearity filters</figcaption>
</figure>

We observe that we stayed on a "hard" filter for the norm filter. On the other hand, the collinearity filter looks more like an exponential taking off around 0.975.

This configuration reaches a performance of **39.47%**.

**Phase 3: Reference Point Optimization**
Once my optimal parameter set is found for the reference point fixed at the image center, I allow myself to use average points of predictions from the previous experiment as new reference points for calculating collinearity scores.
Using these average reference points, performance improves significantly to **29.77%**.

#### Phase 4: Post-processing and Smoothing
I then apply different smoothing methods to predictions, with hyperparameter optimization.
In the table below, we can observe performances obtained for the two previous experiments: the one using reference point at image center and the one using average points of predictions from the previous experiment.

For each experiment, we give the raw score, then the smoothed score with simple average and bi-directional exponential average.

For exponential averages, I optimized parameter α by one-dimensional search. I noted optimal values in the last column.

| Experiment | Raw Score | Simple Average | Bi-directional Exponential Average | α |
|------------|-------------|------------|----------------|----------------|
| ref point: center | 39.47% | 18.29% | 17.63% | 0.01 |
| ref point: average | 29.77% | 10.77% | **8.58%** | 0.05 |

We observe that smoothing predictions with bi-directional exponential average is systematically the best option.

**Final result: 8.58%** with bi-directional exponential smoothing method.

<figure markdown>
  <div style="display: flex; justify-content: space-between; align-items: flex-start;">
    <div style="width: 48%; text-align: center;">
      <span style="font-weight: bold;">Reference point: Image center</span><br>
      <img src="../imgs/5/final_viz_center.gif" alt="Prediction GIF - Center" style="width: 100%;">
    </div>
    <div style="width: 48%; text-align: center;">
      <span style="font-weight: bold;">Reference point: Previous experiment average</span><br>
      <img src="../imgs/5/final_viz_mean.gif" alt="Prediction GIF - Average" style="width: 100%;">
    </div>
  </div>
  <figcaption style="text-align: center; margin-bottom: 10px;">
    <strong>Comparison of final results</strong>
  </figcaption>
</figure>

In the figure above, visually, I tend to prefer the method using reference point at image center because it seems more reactive to vehicle direction changes although the error relative to the label is higher.
I therefore suspect that labels aren't perfectly reliable because too centered around their average value.

#### Auxiliary observation: Why constrain reference point to center

A legitimate question arises: why not let Bayesian search simultaneously optimize all parameters, including choice of reference point for collinearity scores?

I actually tested this unconstrained approach, but it revealed problematic optimizer behavior. The best pre-smoothing scores systematically used average points from previous experiments as reference points for collinearity calculation.

A clear trend then emerged: the filter became extremely restrictive on collinearity scores, keeping only vectors pointing almost exactly toward the reference point. In parallel, the vector norm filter became much more permissive, because drastic filtering on collinearity had already eliminated noise from small vectors.

This strategy produced estimations systematically close to average points from previous experiments, but paradoxically less performant than the constrained search method I finally retained after smoothing.

**The optimization trap**: These solutions artificially improve the score by reducing variance around a reference point potentially distant from true epipoles. The optimizer solves the wrong problem - rather than precisely locating the epipole, it seeks to make predictions as close as possible to this reference point.

**My separation strategy**: By first constraining the reference point to the center, I force the algorithm to find parameters that produce a signal that is noisy but unbiased. Smoothing can then effectively reduce this variance without introducing systematic bias. This approach separates **bias reduction** (parameter optimization) from **variance reduction** (post-processing), avoiding suboptimal solutions from joint optimization.

## Auxiliary Explorations
### Impact of Hard Filtering on Vector Count

During my tests with binary filtering from the previous arc (fixed thresholds of norm ≥ 13 and collinearity ≥ 0.96), I wanted to verify a hypothesis: could the number of vectors remaining after filtering be correlated with prediction quality?

To explore this question, I visualized two metrics across all frames:
- The number of vectors kept after hard filtering
- The distance between predicted epipole and label

**Main observations:**
- Certain videos (0 and 1, best performances) show a relatively low but constant number of vectors
- Other videos present a very variable number of vectors from one frame to another
- In videos with variable numbers, best performances seem to coincide with frames having the most vectors

However, no clear rule emerged from this analysis. Expected correlations didn't materialize in an exploitable way to improve filtering.

# Arc final : Optimisation avancée et conclusions

## Submitting Results
I first visualized the output predictions from my filtering and smoothing on the five test videos. Satisfied with the results, I submitted them to Comma.ai, confident that my solution should perform below the 15% mark, my initial goal.
However, I noted that there were several sharp turns in this dataset, while there was only one in the training data. I hoped this wouldn't impact my results too much. Visually, it seemed to work well.
In any case, there was only one way to find out: submit my results.
After a few days of waiting, I received my model's score: 30%. Ouch, cold shower! I wasn't expecting to see such a performance drop...

"Damn those turns!" I immediately thought. Then another unpleasant idea came to mind. Such a difference between my score on training data and the score obtained on test data didn't leave my old data scientist reflexes indifferent: I had overfitted!

The question had actually been floating in my mind for several arcs: was it relevant to create an evaluation set for the algorithm I was designing? Could my parameter search for filtering and smoothing be considered a form of learning?
Given that I was looking for global parameters working for all frames in the dataset, not frame-specific parameters, I thought this should limit overfitting risks. But I apparently was wrong.

<!-- Whatever the case, part of me felt we were starting to reach the limits of general parameters. But I couldn't stop there with my current method. -->
I needed to create training and evaluation sets that would allow me to improve my score on the test set!

## Dataset Design

### Video Segment Selection for Dataset Construction
After a quick look at frames where my error was highest, it was obvious there was a strong correlation between turns and high error.

To analyze this phenomenon, I developed a method based on deviation from the median point:

1. **Reference point calculation**: For each video, I calculated the median of coordinates (x, y) of points, separately for predictions and labels
2. **Deviation measurement**: For each frame, I calculated the Euclidean distance between that frame's point and the corresponding median point

This approach allows visualizing how points deviate from their "typical" position throughout each video. More interestingly, it allows visual detection of turns: when the point moves away from the median point (which is probably close to the straight-line trajectory), it indicates a turning moment.

In the visualizations below:
- The **blue line** represents vertical shifts: above zero = upward shift, below = downward shift
- The **red line** represents horizontal shifts: above zero = right turn, below = left turn

<figure markdown>
  <div style="display: flex; justify-content: space-between; align-items: flex-start;">
    <div style="width: 48%; text-align: center;">
      <!-- <span style="font-weight: bold;">Reference point: Image center</span><br> -->
      <img src="../imgs/outro/pred.png" alt="Prediction GIF - Center" style="width: 100%;">
    </div>
    <div style="width: 48%; text-align: center;">
      <!-- <span style="font-weight: bold;">Reference point: Previous experiment average</span><br> -->
      <img src="../imgs/outro/label.png" alt="Prediction GIF - Average" style="width: 100%;">
    </div>
  </div>
  <figcaption style="text-align: center; margin-bottom: 10px;">
    <strong>Deviation from median point in pixels: predictions (left) and labels (right)</strong>
  </figcaption>
</figure>

In the figure below, we can observe the Euclidean distance between prediction and label for each frame of each video.

<figure markdown>
  <img src="../imgs/outro/distances.png" alt="Error distance distribution">
  <figcaption>Distance in pixels between prediction and label for each frame of each video</figcaption>
</figure>

It's evident from these figures that turns are the frames where error is highest. I therefore decided to create training and evaluation sets that contain turning frames.

### Dataset Construction

Now that I had identified the problematic passages, I needed to build strategic training and validation sets.

**Selection principle**: I isolated approximately 2300 frames of interest distributed in segments of 100 to 400 frames, ensuring to include:
- Segments with pronounced turns (high error zones)
- Straight segments (to avoid overfitting on turns)
- Special cases like poorly smoothed "speed bumps"

**Sampling method**: For each selected segment, I applied the same decile sampling strategy as before: division into 10 deciles and sampling a fixed number of frames per decile. This approach guarantees balanced representation of each segment.

**Final distribution**:
- **Training set**: 300 frames from segments with difficult turns and problematic smoothing zones
- **Validation set**: 100 frames including straight segments and turns not detected by labels

This more targeted approach allowed me to increase dataset sizes compared to the previous 100 samples, while focusing on critical use cases.

## Experiments with New Datasets

### Bayesian Search for Filtering Parameters

With my new training and validation sets, I relaunched a Bayesian search to optimize filtering parameters. I included my best parameters from previous arcs as starting points to guide the search. I also performed a local search around these parameters to finely explore the nearby space. Surprising result: despite exploring hundreds of combinations, no search direction significantly improved performance on training or validation sets!

### Smoothing Parameter Optimization

I also tested new smoothing parameters on these datasets. Identical result: the optimal parameter was the same as in my previous selection. No improvement was brought by using separate training and validation sets.

### Assessment and Insights

**Major conclusion**: The division into training/validation sets didn't bring significant improvement.

**Key observation**: By analyzing errors more closely, I identified that:
- Errors are massively concentrated in turns
- My method predicts larger deviations than labels during turns
- Smoothing reduces turn amplitude but spreads them temporally

**Final hypothesis**: The real problem isn't parameter overfitting, but a fundamental weakness of my method for predicting vehicle direction in turns. Test set turns are probably more difficult to predict than training ones.

# Assessment and Perspectives

Analysis of results reveals that the developed method, despite successive optimizations, presents a systematic weakness in turns. This weakness doesn't stem from a parameterization or overfitting problem, but from a fundamental conceptual confusion in the initial approach.

## The Focus of Expansion, My Method's Hidden Objective

My approach, based on minimizing optical flow collinearity score, was actually designed to find a Focus of Expansion (FoE). This phenomenon corresponds to the unique point on the image from which all apparent scene movement seems to diverge.

However, the Focus of Expansion only appears under a very strict condition: pure translational movement.

## Optical Flow "Contamination" in Turns

This condition explains the algorithm's failure in turns. A turn is a compound movement, combining translation with rotation. This rotation component "contaminates" the optical flow vector field:
- Translation alone creates radial flow moving away from the FoE.
- Rotation superimposes rotational flow that wraps vectors around a center.

The combination of both breaks the simple divergence pattern. Consequently, in a turn, the Focus of Expansion as a unique convergence point no longer exists.

## The Distinction with the Epipole

The geometric concept that remains valid under all circumstances is the Epipole. It's the projection of one camera's center onto the other's plane. Unlike the FoE, its existence is guaranteed whether the movement contains rotation or not.

## Final Diagnosis

My error was therefore developing a method that estimates the Focus of Expansion while thinking I was estimating the Epipole.
- In straight lines, both concepts coincide, explaining the method's good performance.
- In turns, the algorithm sought a Focus of Expansion that no longer existed geometrically, inevitably leading to unstable and erroneous estimation.

This limitation is therefore fundamental to the chosen approach and cannot be resolved by simple parameter refinement.

## Envisioned Solution: Camera Motion Decomposition

A robust approach consists of separating translation and rotation components of camera movement. This method relies on the following steps:

- Interest point detection on an image
- Tracking these points in the next image
- Fundamental matrix calculation from correspondences
- Essential matrix derivation from camera parameters
- Pure translation movement extraction, isolated from rotation

This approach should be robust to turns, and most necessary components are already available in OpenCV.

## Lesson Learned

My biggest strategic error was keeping smoothing for the end. I saw it as an "easy win," the cherry on top, and therefore underestimated its impact. This was a double error.

- I underestimated its gain. Smoothing wasn't a small bonus, but a major improvement. Being quick to implement, I should have started with it to obtain solid results very early.
- I misunderstood its role. More importantly, smoothing wasn't just a simple optimization; it was my best diagnostic tool. By removing "noise" from my predictions, it would have immediately highlighted my method's fundamental weakness in turns, saving me from spending days perfecting a limited approach.

The lesson is clear: always implement high-impact, low-effort solutions first. Not only for performance, but especially because they clarify the real problem to solve and help determine if more complex efforts are justified.