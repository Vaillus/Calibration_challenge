from src.utilities.paths import get_outputs_dir

def load_mean_point(run_name: str, video_index: int):
    mean_file = get_outputs_dir() / f"means/{run_name}/{video_index}.txt"
    with open(mean_file, 'r') as f:
        line = f.readline().strip()
        x, y = map(float, line.split())
        return x, y