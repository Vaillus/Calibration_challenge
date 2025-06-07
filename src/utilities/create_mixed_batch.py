#!/usr/bin/env python3
"""
Création d'un dataset de test représentatif pour l'optimisation des paramètres de filtrage.

PROBLÈME :
Pour optimiser les paramètres de filtrage des flow vectors, il faut tester de nombreuses
combinaisons de paramètres. Tester sur l'ensemble des frames (~1200 par vidéo) prendrait
trop de temps (2-3 minutes par combinaison de paramètres).

SOLUTION :
Ce script crée un dataset de test plus petit mais représentatif en :
1. Calculant les erreurs de prédiction sur l'ensemble des frames avec des paramètres par défaut
2. Échantillonnant stratégiquement des frames dans chaque décile de la distribution d'erreurs
3. Créant un dataset de test de 100 frames (2 frames × 10 déciles × 5 vidéos)

BÉNÉFICES :
- Évaluation rapide : ~2-3 secondes par combinaison de paramètres
- Représentativité : couvre toute la gamme des erreurs possibles
- Équilibre : même nombre de frames par vidéo et par niveau d'erreur

USAGE :
    python create_mixed_batch.py

Le script génère deux fichiers dans le dossier 'data/intermediate/mixed_batch/' :
- 'mixed_batch_flows.npz' : contient les flow vectors des frames sélectionnées
- 'mixed_batch_labels.npy' : contient les labels correspondants

Author: Hugo Vaillaud
Date: 2024-05-28
"""

import numpy as np
import mlx.core as mx
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ground_truth import read_ground_truth_pixels

def load_distances(video_id):
    """Load distances for a specific video."""
    distances_file = Path(f"data/outputs/distances/video_{video_id}_distances.npy")
    if not distances_file.exists():
        raise FileNotFoundError(f"Distance file not found: {distances_file}")
    
    distances = np.load(distances_file)
    return distances

def load_video_flows(video_id):
    """Load flow data for a specific video."""
    npz_path = Path(f"data/intermediate/flows/{video_id}_float16.npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Flow file not found: {npz_path}")
    
    print(f"Loading flows for video {video_id}...")
    with np.load(npz_path) as data:
        flows_data_f16 = mx.array(data['flow'])
    mx.eval(flows_data_f16)
    
    # Convert to float32
    flows_data = flows_data_f16.astype(mx.float32)
    mx.eval(flows_data)
    
    del flows_data_f16
    return flows_data

def load_video_labels(video_id, total_frames):
    """Load ground truth labels for a specific video."""
    print(f"Loading labels for video {video_id}...")
    # Read ground truth starting from index 1 (flow offset)
    gt_pixels = mx.array(read_ground_truth_pixels(video_id)[1:total_frames+1])
    mx.eval(gt_pixels)
    return gt_pixels

def select_frames_by_deciles(distances, frames_per_decile=2, random_seed=42):
    """
    Select frames from all 10 deciles.
    
    Args:
        distances: Array of distances for all frames
        frames_per_decile: Number of frames to select per decile
        random_seed: Random seed for reproducible selection
        
    Returns:
        list: Selected frame indices for each decile
    """
    np.random.seed(random_seed)
    
    # Calculate all 10 decile boundaries
    decile_boundaries = [np.percentile(distances, i * 10) for i in range(11)]  # 0%, 10%, 20%, ..., 100%
    
    selected_indices = []
    decile_info = []
    
    for decile in range(10):
        lower_bound = decile_boundaries[decile]
        upper_bound = decile_boundaries[decile + 1]
        
        # Find frames in this decile
        if decile == 9:  # Last decile includes the maximum
            decile_mask = (distances >= lower_bound) & (distances <= upper_bound)
        else:
            decile_mask = (distances >= lower_bound) & (distances < upper_bound)
        
        decile_indices = np.where(decile_mask)[0]
        
        print(f"  Decile {decile+1}: {lower_bound:.2f}-{upper_bound:.2f} pixels ({len(decile_indices)} frames)")
        
        # Randomly sample from this decile
        if len(decile_indices) >= frames_per_decile:
            selected = np.random.choice(decile_indices, size=frames_per_decile, replace=False)
        else:
            print(f"    ⚠️  Only {len(decile_indices)} frames available, taking all")
            selected = decile_indices
        
        selected_indices.extend(selected)
        decile_info.extend([decile + 1] * len(selected))  # Store decile number (1-10)
    
    return np.array(selected_indices), decile_info

def create_mixed_batch(frames_per_decile=2, random_seed=42):
    """
    Create a mixed batch with frames from all deciles across videos 0-4.
    
    Args:
        frames_per_decile: Number of frames to select per decile
        random_seed: Random seed for reproducible selection
        
    Returns:
        dict: Contains flows, labels, frame_info, and statistics
    """
    print(f"Creating mixed batch:")
    print(f"  {frames_per_decile} frames per decile × 10 deciles = {frames_per_decile * 10} frames per video")
    print(f"  5 videos (0-4) total = {frames_per_decile * 10 * 5} frames")
    print(f"  Random seed: {random_seed}")
    print()
    
    all_flows = []
    all_labels = []
    frame_info = []
    
    for video_id in range(5):  # Videos 0-4
        print(f"Processing video {video_id}...")
        
        try:
            # Load distances, flows, and labels
            distances = load_distances(video_id)
            flows = load_video_flows(video_id)
            labels = load_video_labels(video_id, len(distances))
            
            # Select frame indices from all deciles
            selected_indices, decile_info = select_frames_by_deciles(
                distances, frames_per_decile=frames_per_decile, random_seed=random_seed + video_id
            )
            
            selected_distances = distances[selected_indices]
            
            print(f"  Selected {len(selected_indices)} frames across all deciles")
            print(f"    Distance range: {selected_distances.min():.1f}-{selected_distances.max():.1f} pixels")
            
            # Extract selected flows and labels
            selected_indices_mx = mx.array(selected_indices)
            selected_flows = flows[selected_indices_mx]
            selected_labels = labels[selected_indices_mx]
            mx.eval(selected_flows)
            mx.eval(selected_labels)
            
            # Store flows and labels
            all_flows.append(selected_flows)
            all_labels.append(selected_labels)
            
            # Store frame information
            for i, (frame_idx, distance, decile) in enumerate(zip(selected_indices, selected_distances, decile_info)):
                frame_info.append({
                    'video_id': video_id,
                    'original_frame_idx': int(frame_idx),
                    'label_idx': int(frame_idx + 1),  # +1 offset for labels
                    'distance': float(distance),
                    'decile': decile,
                    'batch_idx': len(frame_info)  # Position in final batch
                })
            
            print(f"  ✅ Video {video_id} completed")
            
        except FileNotFoundError as e:
            print(f"  ❌ Video {video_id} failed: {e}")
            continue
        
        print()
    
    # Combine all flows and labels into single batches
    if all_flows and all_labels:
        combined_flows = mx.concatenate(all_flows, axis=0)
        combined_labels = mx.concatenate(all_labels, axis=0)
        mx.eval(combined_flows)
        mx.eval(combined_labels)
        
        print(f"Final batch created:")
        print(f"  Flows shape: {combined_flows.shape}")
        print(f"  Labels shape: {combined_labels.shape}")
        print(f"  Total frames: {len(frame_info)}")
        
        # Statistics by decile
        decile_stats = {}
        for decile in range(1, 11):
            decile_frames = [info for info in frame_info if info['decile'] == decile]
            if decile_frames:
                decile_distances = [info['distance'] for info in decile_frames]
                decile_stats[decile] = {
                    'count': len(decile_frames),
                    'avg_distance': np.mean(decile_distances),
                    'distance_range': (np.min(decile_distances), np.max(decile_distances))
                }
        
        print(f"  Decile distribution:")
        for decile, stats in decile_stats.items():
            print(f"    Decile {decile}: {stats['count']} frames, avg: {stats['avg_distance']:.1f}px")
        
        return {
            'flows': combined_flows,
            'labels': combined_labels,
            'frame_info': frame_info,
            'stats': {
                'total_frames': len(frame_info),
                'frames_per_decile': frames_per_decile,
                'decile_stats': decile_stats,
                'overall_distance_range': (
                    min(info['distance'] for info in frame_info),
                    max(info['distance'] for info in frame_info)
                )
            }
        }
    else:
        print("❌ No flows or labels could be loaded!")
        return None

def save_mixed_batch(batch_data, output_dir="data/intermediate/mixed_batch"):
    """Save only the flows and labels to the flows directory."""
    output_path = Path(output_dir)
    
    if batch_data is None:
        print("No batch data to save!")
        return
    
    # Save flows as NPZ (compressed)
    flows_file = output_path / "mixed_batch_flows.npz"
    flows_np = np.array(batch_data['flows'])
    np.savez_compressed(flows_file, flows=flows_np)
    print(f"Flows saved to: {flows_file}")
    
    # Save labels as NPY
    labels_file = output_path / "mixed_batch_labels.npy"
    labels_np = np.array(batch_data['labels'])
    np.save(labels_file, labels_np)
    print(f"Labels saved to: {labels_file}")
    
    print(f"\nMixed batch files saved in: {output_path}")

def main():
    """Main function to create and save mixed batch."""
    print("=== MIXED BATCH CREATOR ===")
    print("Creating batch with frames from all 10 deciles")
    print("2 frames per decile × 10 deciles × 5 videos = 100 frames total")
    print()
    
    # Create mixed batch
    batch_data = create_mixed_batch(
        frames_per_decile=2,
        random_seed=42
    )
    
    if batch_data:
        # Save to files
        save_mixed_batch(batch_data)
        
        print(f"\n{'='*60}")
        print("MIXED BATCH CREATION COMPLETED!")
        print(f"{'='*60}")
        print(f"✅ {batch_data['stats']['total_frames']} frames ready for analysis")
        print(f"✅ Balanced representation across all error deciles")
        print(f"✅ Both flows and labels included with proper indexing")
        print(f"✅ Files saved in calib_challenge/flows/")
    else:
        print("❌ Failed to create mixed batch")

if __name__ == "__main__":
    main() 