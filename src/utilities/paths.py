"""Gestionnaire de chemins pour le projet calib_challenge."""
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Retourne la racine du projet."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Retourne le dossier data/."""
    return get_project_root() / "data"


def get_inputs_dir() -> Path:
    """Retourne le dossier data/inputs/."""
    return get_data_dir() / "inputs"


def get_outputs_dir() -> Path:
    """Retourne le dossier data/outputs/."""
    return get_data_dir() / "outputs"


def get_intermediate_dir() -> Path:
    """Retourne le dossier data/intermediate/."""
    return get_data_dir() / "intermediate"


def get_flows_dir() -> Path:
    """Retourne le dossier des flows précalculés."""
    return get_intermediate_dir() / "flows"


def get_pred_dir(generation_num: int) -> Path:
    """Retourne le dossier des prédictions.
    
    Args:
        generation_num: Numéro de génération (0-7)
        
    Returns:
        Path vers data/outputs/predictions/{generation_num}/
    """
    return get_outputs_dir() / "pred" / str(generation_num)


def get_means_dir(video_num: Optional[int] = None) -> Path:
    """Retourne le dossier des moyennes.
    
    Args:
        video_num: Numéro de vidéo optionnel (ex: 5 pour means/5/)
        
    Returns:
        Path vers data/outputs/means/ ou data/outputs/means/{video_num}/
    """
    base = get_outputs_dir() / "means"
    if video_num is not None:
        return base / str(video_num)
    return base


def get_distances_dir(video_num: Optional[int] = None) -> Path:
    """Retourne le dossier des distances.
    
    Args:
        video_num: Numéro de vidéo optionnel
        
    Returns:
        Path vers data/outputs/distances/ ou data/outputs/distances/{video_num}/
    """
    base = get_outputs_dir() / "distances"
    if video_num is not None:
        return base / str(video_num)
    return base


def get_visualizations_dir() -> Path:
    """Retourne le dossier des visualisations."""
    return get_outputs_dir() / "visualizations"


def get_masks_dir() -> Path:
    """Retourne le dossier des masques manuels."""
    return get_intermediate_dir() / "masks"


def get_labeled_dir() -> Path:
    """Retourne le dossier des données labellisées (ground truth)."""
    return get_inputs_dir() / "labeled"


def ensure_dir_exists(path: Path) -> Path:
    """Crée le dossier s'il n'existe pas et retourne le chemin.
    
    Args:
        path: Chemin du dossier à créer
        
    Returns:
        Le même chemin, après création si nécessaire
    """
    path.mkdir(parents=True, exist_ok=True)
    return path 