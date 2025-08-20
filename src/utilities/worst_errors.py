import numpy as np
import random
from typing import List, Tuple, Optional

from src.utilities.paths import get_labeled_dir, get_pred_dir
from src.utilities.pixel_angle_converter import angles_to_pixels

DEFAULT_FOCAL_LENGTH = 910  # Focal length in pixels
IMAGE_WIDTH = 1920  # Standard image width
IMAGE_HEIGHT = 1080  # Standard image height

def load_valid_distances_single_vid(video_id:int, run_name:str="vanilla"
                                   ) -> Tuple[List[float], List[int]]:
    """ For a specific video, load the ground truth and the predictions 
    from a given run.
    Identify the "valid" frames, i.e. frames where the ground truth and 
    predicted pixel are not NaN.
    For each valid frame, compute the distance between the ground truth 
    and predicted pixel.
    Return a tuple of two lists:
    - distances: array of distances for the valid frames
    - valid_frame_indices: array of frame indices for the valid frames"""
    
    # Load ground truth
    gt_path = get_labeled_dir() / f"{video_id}.txt"
    if not gt_path.exists():
        return None, None
    # Load predictions
    pred_path = get_pred_dir(run_name) / f"{video_id}.txt"
    if not pred_path.exists():
        return None, None
    # Load data
    gt_data = np.loadtxt(gt_path)  # (pitch, yaw) in radians
    pred_data = np.loadtxt(pred_path)  # (pitch, yaw) in radians
    # Ensure same number of frames
    min_frames = min(len(gt_data), len(pred_data))
    gt_data = gt_data[:min_frames]
    pred_data = pred_data[:min_frames]
    # Calculate distances for valid frames only
    valid_distances = []
    valid_frame_indices = []
    
    for i in range(min_frames):
        # Skip if prediction has NaN or if ground truth has NaN
        if np.isnan(pred_data[i]).any() or np.isnan(gt_data[i]).any():
            continue
            
        # Convert angles to pixels
        pitch_gt, yaw_gt = gt_data[i]
        x_gt, y_gt = angles_to_pixels(pitch_gt, yaw_gt)

        pitch_pred, yaw_pred = pred_data[i]
        x_pred, y_pred = angles_to_pixels(pitch_pred, yaw_pred)
        
        # Calculate distance
        distance = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
        valid_distances.append(distance)
        valid_frame_indices.append(i)
    
    if valid_distances:
        return valid_distances, valid_frame_indices
    else:
        return None, None
    
def load_valid_distances_mult_vids(
        video_ids:Optional[List[int]] = None, 
        run_name:str="vanilla"
    ) -> List[Tuple[float, int, int]]:
    """
    For all videos, load the ground truth and the predictions 
    from a given run.
    Identify the "valid" frames, i.e. frames where the ground truth and 
    predicted pixel are not NaN.
    For each valid frame, compute the distance between the ground truth 
    and predicted pixel.
    Return a list of tuples (distance, video_id, frame_id).
    """
    if video_ids is None:
        video_ids = range(5)

    all_distances_with_coords = []
    for vid_id in video_ids:
        distances, frame_indices = load_valid_distances_single_vid(vid_id, run_name)
        
        if distances is not None and len(distances) > 0:
            for dist, frame_id in zip(distances, frame_indices):
                all_distances_with_coords.append((dist, vid_id, frame_id))
    
    if not all_distances_with_coords:
        print("❌ Aucune distance calculée!")
        return []
    
    return all_distances_with_coords

def get_worst_errors_global(run_name="vanilla", k=4) -> List[Tuple[int, int]]:
    """
    Récupère directement les k pires erreurs globales sans sauvegarde.
    
    Returns:
        Liste de tuples (video_id, frame_id)
    """
    all_errors = []
    
    for video_id in range(5):  # Videos 0 to 4
        distances, frame_indices = load_valid_distances_single_vid(video_id, run_name)
        
        if distances is not None and len(distances) > 0:
            for dist, frame_id in zip(distances, frame_indices):
                all_errors.append((dist, video_id, frame_id))
    
    if not all_errors:
        return []
    
    # Trier par distance décroissante et prendre les k premiers
    all_errors.sort(key=lambda x: x[0], reverse=True)
    worst_errors = all_errors[:k]
    
    # Retourner seulement les coordonnées (video_id, frame_id)
    return [(int(video_id), int(frame_id)) for _, video_id, frame_id in worst_errors]

def get_worst_errors_per_video(run_name="vanilla", k_per_video=2) -> List[Tuple[int, int]]:
    """
    Récupère directement les k pires erreurs par vidéo sans sauvegarde.
    
    Returns:
        Liste de tuples (video_id, frame_id)
    """
    all_coordinates = []
    
    for video_id in range(5):  # Videos 0 to 4
        distances, frame_indices = load_valid_distances_single_vid(video_id, run_name)
        
        if distances is not None and len(distances) > 0:
            # Créer liste des erreurs pour cette vidéo
            video_errors = [(dist, video_id, frame_id) for dist, frame_id in zip(distances, frame_indices)]
            
            # Trier par distance décroissante et prendre les k_per_video premiers
            video_errors.sort(key=lambda x: x[0], reverse=True)
            worst_video_errors = video_errors[:k_per_video]
            
            # Ajouter les coordonnées
            for _, vid_id, frame_id in worst_video_errors:
                all_coordinates.append((int(vid_id), int(frame_id)))
    
    return all_coordinates

def select_random_frames_from_7th_decile(run_name="5", n_frames=10) -> List[Tuple[int, int]]:
    """
    Sélectionne n frames aléatoires du 7ème décile de la distribution des distances.
    
    Args:
        run_name: Nom du run à analyser
        n_frames: Nombre de frames à sélectionner
        
    Returns:
        List[Tuple[int, int]]: Liste de (video_id, frame_id)
    """
    print(f"🎲 Sélection de {n_frames} frames aléatoires du 7ème décile...")
    
    # 1. Calculer toutes les distances avec coordonnées
    all_distances_with_coords = []
    
    for video_id in range(5):
        distances, frame_indices = load_valid_distances_single_vid(video_id, run_name)
        
        if distances is not None and len(distances) > 0:
            for dist, frame_id in zip(distances, frame_indices):
                all_distances_with_coords.append((dist, video_id, frame_id))
    
    if not all_distances_with_coords:
        print("❌ Aucune distance calculée!")
        return []
    
    # 2. Calculer les percentiles
    distances = [d[0] for d in all_distances_with_coords]
    p70 = np.percentile(distances, 70)  # 7ème décile début
    p80 = np.percentile(distances, 80)  # 7ème décile fin
    
    print(f"📊 7ème décile: {p70:.1f} - {p80:.1f} pixels")
    
    # 3. Filtrer les frames dans le 7ème décile
    frames_in_7th_decile = [
        (int(video_id), int(frame_id)) 
        for dist, video_id, frame_id in all_distances_with_coords
        if p70 <= dist <= p80
    ]
    
    print(f"🎯 {len(frames_in_7th_decile)} frames dans le 7ème décile")
    
    if len(frames_in_7th_decile) < n_frames:
        print(f"⚠️  Seulement {len(frames_in_7th_decile)} frames disponibles")
        return frames_in_7th_decile
    
    # 4. Sélectionner aléatoirement
    selected_frames = random.sample(frames_in_7th_decile, n_frames)
    
    print(f"✅ {n_frames} frames sélectionnés aléatoirement:")
    for i, (vid, frame) in enumerate(selected_frames, 1):
        print(f"   {i}. Video {vid}, Frame {frame}")
    
    return [(int(vid), int(frame)) for vid, frame in selected_frames]


def select_frames_from_decile(
        distances_with_ids:List[Tuple[float, int, int]],
        distances:List[float],
        decile:int=7, 
        n_frames:int=10,
        seed:int=None,
        verbose:bool=False
    ) -> List[Tuple[int, int]]:
    """
    En entrée, on a une liste de distances avec coordonnées (video_id, frame_id, distance).
    On sélectionne n_frames frames aléatoirement dans le décile spécifié de cette distribution.
    On retourne la liste des couples (video_id, frame_id) de ces frames.
    
    Args:
        distances_with_ids: Liste de distances avec coordonnées (distance, video_id, frame_id)
        distances: Liste des distances
        decile: Décile à analyser (1-10)
        n_frames: Nombre de frames à sélectionner
        seed: Seed pour la génération des nombres aléatoires
        
    Returns:
        List[Tuple[int, int]]: Liste de (video_id, frame_id)
    """
    if not (1 <= decile <= 10):
        raise ValueError("Le décile doit être entre 1 et 10")
    
    # 1. Calculer les percentiles pour le décile demandé
    p_start = (decile - 1) * 10  # Début du décile (ex: 60% pour le 7ème)
    p_end = decile * 10           # Fin du décile (ex: 70% pour le 7ème)
    percentile_start = np.percentile(distances, p_start)
    percentile_end = np.percentile(distances, p_end)
    
    # 2. Filtrer les frames dans le décile demandé
    # On note qu'on exclue la frame 0 la prédiction associée n'est qu'un dupliqué de la frame 1
    # On exclue aussi les frames supérieures à 1193 car certaines vidéos ne contiennent pas
    # de frames après la frame 1193.
    frames_in_decile = [
        (int(vid_id), int(frame_id)) 
        for dist, vid_id, frame_id in distances_with_ids
        if (percentile_start <= dist <= percentile_end) and (0 < frame_id <1193)
    ]
    
    # print(f"🎯 {len(frames_in_decile)} frames dans le {decile}ème décile")
    
    if len(frames_in_decile) < n_frames:
        if verbose:
            print(f"⚠️  Seulement {len(frames_in_decile)} frames disponibles")
        return frames_in_decile
    
    # 4. Sélectionner aléatoirement
    if seed is not None:
        random.seed(seed)
    selected_frames = random.sample(frames_in_decile, n_frames)
    
    # print(f"✅ {n_frames} frames sélectionnés aléatoirement:")
    # for i, (vid, frame) in enumerate(selected_frames, 1):
        # print(f"   {i}. Video {vid}, Frame {frame}")
    
    return selected_frames


def select_frames_from_all_deciles(
        run_name:str="vanilla", 
        n_frames_per_decile:int=5, 
        video_ids:Optional[List[int]] = None, 
        seed:int=None
    ) -> List[Tuple[int, int]]:
    """
    Une distribution des distances est calculée pour la run spécifiée, 
    soit pour la vidéo spécifiée, soit pour toutes les vidéos.
    n_frames_per_decile frames sont échantillonnés aléatoirement dans 
    chacun des 10 déciles de cette distribution.
    Pour chaque frame, on retourne le couple (video_id, frame_id).
    
    Args:
        run_name: Nom du run à analyser
        n_frames_per_decile: Nombre de frames à sélectionner par décile
        video_ids: Si spécifié, ne considère que ces vidéos. Si None, analyse toutes les vidéos.
        seed: Seed pour la génération des nombres aléatoires
        
    Returns:
        List[Tuple[int, int]]: Liste de (video_id, frame_id)
    """
    # 1. Calculer toutes les distances avec coordonnées
    distances_with_ids = load_valid_distances_mult_vids(
        video_ids=video_ids, run_name=run_name
    )
    distances = [d[0] for d in distances_with_ids]

    all_frames = []
    
    for decile in range(1, 11):  # Déciles 1 à 10
        frames = select_frames_from_decile(
            distances_with_ids=distances_with_ids,
            distances=distances,
            decile=decile,
            n_frames=n_frames_per_decile,
            seed=seed
        )
        all_frames.extend(frames)
    
    return all_frames 