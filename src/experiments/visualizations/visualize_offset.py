import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

from src.utilities.paths import get_pred_dir, get_means_dir
from src.utilities.load_ground_truth import read_ground_truth_pixels
from src.utilities.load_predictions import load_predictions
from src.utilities.project_constants import get_project_constants


"""
This script is used to visualize the offset of the predictions from the mean point.
"""



def get_offsets(run_name:str, video_id:int) -> Tuple[List[int], List[int], List[int]]:
    predictions = load_predictions(predictions_dir=run_name, video_index=video_id)
    # mean_point = load_mean_point("5_6",video_id)

    xs = np.array([predictions[i][0] for i in range(len(predictions))], dtype=np.float16)
    ys = np.array([predictions[i][1] for i in range(len(predictions))], dtype=np.float16)
    consts = get_project_constants()
    wid = consts["frame_width"]
    hei = consts["frame_height"]
    center = (wid //2, hei//2)
    # for i in range(len(predictions)):
    #     if xs[i] == center[0]:
    #         xs[i] = np.nan
    #     xs[i] = [x[i] if x[i] != center[0] else np.nan for i in range(len(xs))]

    x_mean = np.median([x for x in xs if x != center[0]])
    y_mean = np.median([y for y in ys if y != center[1]])

    x_diff = [xs[i] - x_mean for i in range(len(predictions))]
    y_diff = [ys[i] - y_mean for i in range(len(predictions))]
    dist = [math.sqrt(x_diff[i]**2 + y_diff[i]**2) for i in range(len(predictions))]

    return x_diff, y_diff, dist

def get_label_offset(video_id:int) -> Tuple[List[int], List[int], List[int]]:
    labels = read_ground_truth_pixels(video_index=video_id)
    # mean_point = load_mean_point("5_6",video_id)

    xs = np.array([labels[i][0] for i in range(len(labels))], dtype=np.float16)
    ys = np.array([labels[i][1] for i in range(len(labels))], dtype=np.float16)
    consts = get_project_constants()
    wid = consts["frame_width"]
    hei = consts["frame_height"]
    center = (wid //2, hei//2)
    # for i in range(len(predictions)):
    #     if xs[i] == center[0]:
    #         xs[i] = np.nan
    #     xs[i] = [x[i] if x[i] != center[0] else np.nan for i in range(len(xs))]

    x_mean = np.median([x for x in xs if x != center[0]])
    y_mean = np.median([y for y in ys if y != center[1]])

    x_diff = [xs[i] - x_mean for i in range(len(labels))]
    y_diff = [ys[i] - y_mean for i in range(len(labels))]
    dist = [math.sqrt(x_diff[i]**2 + y_diff[i]**2) for i in range(len(labels))]

    return x_diff, y_diff, dist

def plot_offsets_single_vid(run_name, video_id):
    x_diff, y_diff, dist = get_offsets(run_name, video_id)

    plt.plot(x_diff, color='red', label="horizontal difference")
    plt.plot(y_diff, color='blue', label="vertical difference")
    plt.plot(dist, color='yellow', label="distance")

    plt.legend()
    plt.show()

def plot_offsets_multi_vids(run_name:str, type:Optional[str] = None, video_ids:Optional[List[int]] = None):
    if type is None:
        type = "pred"
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    axes[-1].set_visible(False)  # Hide the last subplot (position (1,2))
    axes = axes.flatten()
    linewidth = 0.7

    if video_ids is None:
        video_ids = range(5)
    for i, video_id in enumerate(video_ids):
        ax = axes[i]
        if type == "pred":
            x_diff, y_diff, dist = get_offsets(run_name, video_id)
        elif type == "label":
            x_diff, y_diff, dist = get_label_offset(video_id)
        else:
            raise ValueError(f"{type} is not a valid value for the 'type' argument")
        ax.plot(x_diff, color='red', label="horizontal difference", linewidth = linewidth)
        ax.plot(y_diff, color='blue', label="vertical difference", linewidth = linewidth)
        # ax.plot(dist, color='green', label="distance")

        ax.legend()
        ax.set_ylim(-60, 60)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax.set_title(f"Video {video_id}")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Offset from median (pixels)")
        ax.grid(True, alpha=0.3)
    plt.show()
    


if __name__ == "__main__":
    run_name = "5_7_smoothed"
    video_id = 0
    plot_offsets_multi_vids(
        run_name, 
        "label",
        video_ids=list(range(0,5))
        )
