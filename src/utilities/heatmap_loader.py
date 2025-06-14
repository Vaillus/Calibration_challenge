#!/usr/bin/env python3
"""
Chargement de heatmaps de collinéarité.

Ce module contient toutes les fonctions pour charger les heatmaps de collinéarité,
que ce soit les heatmaps individuelles ou globales, absolues ou relatives.

Usage:
    from src.utilities.heatmap_loader import load_global_absolute_heatmap
    
    heatmap, metadata = load_global_absolute_heatmap("unfiltered")
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from src.utilities.paths import get_data_dir


def list_available_heatmaps() -> List[str]:
    """
    Liste les heatmaps disponibles dans la structure de dossiers.
    
    Returns:
        Liste des configurations disponibles
    """
    data_dir = get_data_dir()
    heatmaps_dir = data_dir / "intermediate" / "heatmaps"
    
    if not heatmaps_dir.exists():
        return []
    
    # Lister les dossiers de configuration qui contiennent un dossier individual/
    configurations = []
    for item in heatmaps_dir.iterdir():
        if item.is_dir():
            individual_dir = item / "individual"
            if individual_dir.exists() and individual_dir.is_dir():
                # Vérifier qu'il y a au moins un fichier heatmap dans individual/
                heatmap_files = list(individual_dir.glob("video*_heatmap.npy"))
                if heatmap_files:
                    configurations.append(item.name)
    
    return sorted(configurations)


def load_individual_heatmap(video_id: int, config: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge une heatmap individuelle et ses statistiques.
    
    Args:
        video_id: ID de la vidéo
        config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
        
    Returns:
        Tuple (heatmap, stats) ou (None, None) si erreur
    """
    try:
        data_dir = get_data_dir()
        config_dir = data_dir / "intermediate" / "heatmaps" / config / "individual"
        
        # Noms de fichiers
        heatmap_filename = f"video{video_id}_heatmap.npy"
        stats_filename = f"video{video_id}_stats.json"
        
        heatmap_path = config_dir / heatmap_filename
        stats_path = config_dir / stats_filename
        
        # Vérifier l'existence des fichiers
        if not heatmap_path.exists():
            print(f"❌ Fichier heatmap introuvable: {heatmap_path}")
            return None, None
            
        if not stats_path.exists():
            print(f"❌ Fichier stats introuvable: {stats_path}")
            return None, None
        
        # Charger la heatmap
        heatmap = np.load(heatmap_path)
        
        # Charger les stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        return heatmap, stats
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None


def load_global_heatmap(config: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge une heatmap globale précédemment sauvegardée (absolue ou relative).
    
    Args:
        config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
        
    Returns:
        Tuple (global_heatmap, metadata) ou (None, None) si erreur
    """
    data_dir = get_data_dir()
    global_dir = data_dir / "intermediate" / "heatmaps" / config / "global"
    
    heatmap_filename = "global_heatmap.npy"
    metadata_filename = "global_heatmap_metadata.json"
    
    heatmap_path = global_dir / heatmap_filename
    metadata_path = global_dir / metadata_filename
    
    try:
        if not heatmap_path.exists() or not metadata_path.exists():
            return None, None
        
        # Charger la heatmap
        global_heatmap = np.load(heatmap_path)
        
        # Charger les métadonnées
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Reconvertir les coordonnées en arrays numpy si c'est une heatmap relative
        if metadata.get('type') == 'relative' and 'unified_space' in metadata:
            if 'coordinates' in metadata['unified_space']:
                metadata['unified_space']['coordinates']['x'] = np.array(metadata['unified_space']['coordinates']['x'])
                metadata['unified_space']['coordinates']['y'] = np.array(metadata['unified_space']['coordinates']['y'])
        
        return global_heatmap, metadata
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None


def load_global_absolute_heatmap(config: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Charge spécifiquement une heatmap globale absolue.
    
    Args:
        config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
        
    Returns:
        Tuple (global_absolute_heatmap, metadata) ou (None, None) si erreur
    """
    global_heatmap, metadata = load_global_heatmap(config)
    
    if global_heatmap is None or metadata is None:
        return None, None
    
    # Vérifier que c'est bien une heatmap absolue
    if metadata.get('type') != 'absolute':
        print(f"⚠️  La heatmap de configuration '{config}' n'est pas de type absolue")
        return None, None
    
    return global_heatmap, metadata


def load_mean_pixel_for_video(video_id: int) -> Optional[Tuple[float, float]]:
    """
    Charge la moyenne des pixels pour une vidéo depuis le fichier txt correspondant.
    
    Args:
        video_id: ID de la vidéo
        
    Returns:
        Tuple (x, y) des coordonnées moyennes, ou None si erreur
    """
    try:
        from src.utilities.paths import get_outputs_dir
        outputs_dir = get_outputs_dir()
        mean_file = outputs_dir / "means" / "5" / f"{video_id}.txt"
        
        if not mean_file.exists():
            print(f"❌ Fichier moyenne introuvable: {mean_file}")
            return None
            
        with open(mean_file, 'r') as f:
            line = f.readline().strip()
            if not line:
                print(f"❌ Fichier moyenne vide: {mean_file}")
                return None
                
            parts = line.split()
            if len(parts) != 2:
                print(f"❌ Format incorrect dans {mean_file}: {line}")
                return None
                
            x, y = float(parts[0]), float(parts[1])
            return (x, y)
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la moyenne: {e}")
        return None


def get_heatmap_info(config: str) -> Optional[Dict]:
    """
    Récupère les informations d'une configuration de heatmap sans charger les données.
    
    Args:
        config: Configuration (ex: "filtered_thresh13.0", "unfiltered")
        
    Returns:
        Dictionnaire avec les métadonnées ou None si erreur
    """
    data_dir = get_data_dir()
    
    # Vérifier le fichier de résumé
    summary_path = data_dir / "intermediate" / "heatmaps" / config / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            return summary
        except Exception as e:
            print(f"❌ Erreur lors du chargement du résumé: {e}")
    
    # Vérifier les métadonnées globales
    metadata_path = data_dir / "intermediate" / "heatmaps" / config / "global" / "global_heatmap_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"❌ Erreur lors du chargement des métadonnées: {e}")
    
    return None


def check_heatmap_compatibility(config: str, video_ids: List[int]) -> Dict[str, bool]:
    """
    Vérifie la compatibilité d'une configuration avec une liste de vidéos.
    
    Args:
        config: Configuration à vérifier
        video_ids: Liste des IDs de vidéos
        
    Returns:
        Dictionnaire {video_id: disponible}
    """
    data_dir = get_data_dir()
    config_dir = data_dir / "intermediate" / "heatmaps" / config / "individual"
    
    compatibility = {}
    
    for video_id in video_ids:
        heatmap_filename = f"video{video_id}_heatmap.npy"
        stats_filename = f"video{video_id}_stats.json"
        
        heatmap_path = config_dir / heatmap_filename
        stats_path = config_dir / stats_filename
        
        compatibility[video_id] = heatmap_path.exists() and stats_path.exists()
    
    return compatibility 