#!/usr/bin/env python3
"""
Script simple pour charger les flows depuis le dossier calib_challenge/flows/

Usage:
    from load_flows import load_flows
    
    # Charger flows pour vidéo 0 (format original .npy)
    flows = load_flows(0)
    
    # Charger flows pour vidéo 1 (format compressé .npz)
    flows = load_flows(1, use_compressed=True)
    
    # Charger un flow spécifique (frame 100 de vidéo 0)
    single_flow = load_single_flow(0, 100)
    
    # Lister tous les fichiers de flows disponibles
    list_available_flows()
"""

import numpy as np
import os
from pathlib import Path
from typing import Optional, Union, Tuple, List

from src.utilities.paths import get_flows_dir

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def list_available_flows() -> List[dict]:
    """
    Liste tous les fichiers de flows disponibles.
    
    Returns:
        List[dict]: Liste des flows disponibles avec leurs informations
    """
    flows_dir = get_flows_dir()
    flows_info = []
    
    if not flows_dir.exists():
        print(f"❌ Dossier flows non trouvé: {flows_dir}")
        return flows_info
    
    print(f"📁 Flows disponibles dans: {flows_dir}")
    print("=" * 70)
    
    # Chercher les fichiers .npy (format original)
    for npy_file in sorted(flows_dir.glob("*.npy")):
        if npy_file.name.endswith("_labels.npy"):
            continue  # Skip label files
            
        size_gb = npy_file.stat().st_size / (1024**3)
        
        # Extraire l'index de la vidéo
        try:
            video_index = int(npy_file.stem)
            flows_info.append({
                'video_index': video_index,
                'format': 'original',
                'file': npy_file,
                'size_gb': size_gb
            })
            print(f"🎬 Vidéo {video_index:2d} | Format: NPY     | Taille: {size_gb:.1f} GB | {npy_file.name}")
        except ValueError:
            # Nom de fichier qui n'est pas un index de vidéo simple
            print(f"📄 Autre fichier: {npy_file.name} | Taille: {size_gb:.1f} GB")
    
    # Chercher les fichiers .npz (format compressé)
    for npz_file in sorted(flows_dir.glob("*_float16.npz")):
        size_gb = npz_file.stat().st_size / (1024**3)
        
        # Extraire l'index de la vidéo
        try:
            video_index = int(npz_file.stem.split("_")[0])
            flows_info.append({
                'video_index': video_index,
                'format': 'compressed',
                'file': npz_file,
                'size_gb': size_gb
            })
            print(f"🎬 Vidéo {video_index:2d} | Format: NPZ     | Taille: {size_gb:.1f} GB | {npz_file.name}")
        except (ValueError, IndexError):
            print(f"📄 Autre fichier: {npz_file.name} | Taille: {size_gb:.1f} GB")
    
    # Fichiers spéciaux
    special_files = ["mixed_batch_flows.npz", "first_frame_float16.npz"]
    for special_file in special_files:
        special_path = flows_dir / special_file
        if special_path.exists():
            size_mb = special_path.stat().st_size / (1024**2)
            print(f"⭐ Fichier spécial: {special_file} | Taille: {size_mb:.1f} MB")
    
    print("=" * 70)
    return flows_info

def load_flows(video_index: int, use_compressed: bool = False, verbose: bool = True, 
               start_frame: Optional[int] = None, end_frame: Optional[int] = None, 
               return_mlx: bool = False):
    """
    Charge les flows pour une vidéo donnée.
    
    Args:
        video_index (int): Index de la vidéo (0, 1, 2, 3, 4)
        use_compressed (bool): Si True, utilise le format .npz compressé, sinon .npy
        verbose (bool): Si True, affiche des informations de chargement
        start_frame (Optional[int]): Frame de début (0 par défaut)
        end_frame (Optional[int]): Frame de fin (dernière frame par défaut)
        return_mlx (bool): Si True, retourne un MLX array au lieu de NumPy
        
    Returns:
        Union[np.ndarray, mx.array]: Array des flows de forme (n_frames, height, width, 2) ou None si erreur
    """
    flows_dir = get_flows_dir()
    
    if use_compressed:
        flow_file = flows_dir / f"{video_index}_float16.npz"
        file_type = "NPZ (compressé)"
    else:
        flow_file = flows_dir / f"{video_index}.npy"
        file_type = "NPY (original)"
    
    if not flow_file.exists():
        print(f"❌ Fichier de flows non trouvé: {flow_file}")
        return None
    
    if verbose:
        size_gb = flow_file.stat().st_size / (1024**3)
        print(f"📂 Chargement flows vidéo {video_index} ({file_type}) - {size_gb:.1f} GB")
    
    try:
        if use_compressed:
            with np.load(flow_file) as data:
                flows = data['flow']
                if verbose:
                    print(f"✅ Flows chargés depuis NPZ: {flows.shape} ({flows.dtype})")
        else:
            flows = np.load(flow_file)
            if verbose:
                print(f"✅ Flows chargés depuis NPY: {flows.shape} ({flows.dtype})")
        
        # Gestion des plages de frames
        if start_frame is not None or end_frame is not None:
            start = start_frame if start_frame is not None else 0
            end = end_frame if end_frame is not None else len(flows) - 1
            flows = flows[start:end+1]
            if verbose:
                print(f"   🎯 Frames {start} à {end} ({end-start+1} frames)")
        
        # Conversion MLX si demandée
        if return_mlx:
            if not MLX_AVAILABLE:
                print("⚠️ MLX non disponible, retour NumPy array")
                return flows
            
            if verbose:
                print(f"   🔄 Conversion en MLX array...")
            flows_mlx = mx.array(flows, dtype=mx.float32)
            mx.eval(flows_mlx)
            if verbose:
                print(f"   ✅ MLX array: {flows_mlx.shape}")
            return flows_mlx
        
        return flows
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

def load_single_flow(video_index: int, frame_index: int, use_compressed: bool = False) -> Optional[np.ndarray]:
    """
    Charge un flow spécifique (une frame) pour une vidéo donnée.
    
    Args:
        video_index (int): Index de la vidéo
        frame_index (int): Index de la frame
        use_compressed (bool): Si True, utilise le format .npz compressé
        
    Returns:
        np.ndarray: Flow de forme (height, width, 2) ou None si erreur
    """
    flows_dir = get_flows_dir()
    
    if use_compressed:
        flow_file = flows_dir / f"{video_index}_float16.npz"
    else:
        flow_file = flows_dir / f"{video_index}.npy"
    
    if not flow_file.exists():
        print(f"❌ Fichier de flows non trouvé: {flow_file}")
        return None
    
    try:
        if use_compressed:
            with np.load(flow_file) as data:
                flows = data['flow']
                if frame_index >= len(flows):
                    print(f"❌ Frame index {frame_index} trop grand, max: {len(flows)-1}")
                    return None
                single_flow = flows[frame_index]
        else:
            # Pour les gros fichiers .npy, on peut utiliser mmap pour éviter de charger tout
            flows = np.load(flow_file, mmap_mode='r')
            if frame_index >= len(flows):
                print(f"❌ Frame index {frame_index} trop grand, max: {len(flows)-1}")
                return None
            single_flow = flows[frame_index].copy()  # Copy pour détacher du mmap
        
        print(f"✅ Flow chargé - Vidéo {video_index}, Frame {frame_index}: {single_flow.shape}")
        return single_flow
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

def get_flow_info(video_index: int, use_compressed: bool = False) -> Optional[dict]:
    """
    Obtient des informations sur les flows d'une vidéo sans les charger.
    
    Args:
        video_index (int): Index de la vidéo
        use_compressed (bool): Si True, vérifie le format .npz compressé
        
    Returns:
        dict: Informations sur les flows ou None si erreur
    """
    flows_dir = get_flows_dir()
    
    if use_compressed:
        flow_file = flows_dir / f"{video_index}_float16.npz"
    else:
        flow_file = flows_dir / f"{video_index}.npy"
    
    if not flow_file.exists():
        return None
    
    try:
        size_gb = flow_file.stat().st_size / (1024**3)
        
        if use_compressed:
            with np.load(flow_file) as data:
                shape = data['flow'].shape
                dtype = data['flow'].dtype
        else:
            # Juste charger la metadata sans charger le fichier entier
            flows = np.load(flow_file, mmap_mode='r')
            shape = flows.shape
            dtype = flows.dtype
        
        return {
            'video_index': video_index,
            'file_path': flow_file,
            'shape': shape,
            'dtype': str(dtype),
            'size_gb': size_gb,
            'n_frames': shape[0],
            'height': shape[1],
            'width': shape[2],
            'compressed': use_compressed
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des métadonnées: {e}")
        return None

def load_mixed_batch() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Charge le batch mixte (flows et labels).
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (flows, labels) ou None si erreur
    """
    flows_dir = get_flows_dir()
    
    flows_file = flows_dir / "mixed_batch_flows.npz"
    labels_file = flows_dir / "mixed_batch_labels.npy"
    
    if not flows_file.exists():
        print(f"❌ Fichier mixed batch flows non trouvé: {flows_file}")
        return None
    
    if not labels_file.exists():
        print(f"❌ Fichier mixed batch labels non trouvé: {labels_file}")
        return None
    
    try:
        print("📂 Chargement du batch mixte...")
        
        with np.load(flows_file) as data:
            flows = data['flows']
        
        labels = np.load(labels_file)
        
        print(f"✅ Batch mixte chargé:")
        print(f"   Flows: {flows.shape} ({flows.dtype})")
        print(f"   Labels: {labels.shape} ({labels.dtype})")
        
        return flows, labels
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du batch mixte: {e}")
        return None

if __name__ == "__main__":
    print("🔍 SCRIPT DE CHARGEMENT DES FLOWS")
    print("=" * 50)
    
    # Lister tous les flows disponibles
    list_available_flows()
    
    print("\n📊 EXEMPLES D'UTILISATION:")
    print("-" * 30)
    
    # Exemple 1: Charger flows complets
    print("\n1️⃣ Chargement flows complets (vidéo 0, format compressé):")
    flows = load_flows(0, use_compressed=True)
    if flows is not None:
        print(f"   Flows chargés: {flows.shape}")
        print(f"   Premier flow: {flows[0].shape}")
    
    # Exemple 2: Charger un flow spécifique
    print("\n2️⃣ Chargement d'un flow spécifique (vidéo 0, frame 100):")
    single_flow = load_single_flow(0, 100, use_compressed=True)
    if single_flow is not None:
        print(f"   Flow spécifique: {single_flow.shape}")
        print(f"   Valeurs: min={single_flow.min():.3f}, max={single_flow.max():.3f}")
    
    # Exemple 3: Infos sans chargement
    print("\n3️⃣ Informations sans chargement:")
    info = get_flow_info(0, use_compressed=True)
    if info:
        print(f"   Vidéo {info['video_index']}: {info['n_frames']} frames")
        print(f"   Résolution: {info['height']}x{info['width']}")
        print(f"   Type: {info['dtype']}, Taille: {info['size_gb']:.1f} GB") 