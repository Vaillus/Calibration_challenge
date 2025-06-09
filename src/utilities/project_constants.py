import json

from src.utilities.paths import get_inputs_dir

def get_project_constants():
    """
    Get the project constants from the inputs directory.
    
    Returns:
        dict: The project constants (focal_length, frame_width, frame_height)
    """
    project_constants_path = get_inputs_dir() / "project_constants.json"
    with open(project_constants_path, "r") as f:
        return json.load(f)
    
