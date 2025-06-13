#!/usr/bin/env python3
"""
Analyse du nombre de flow vectors par frame après filtrage.

Ce script prend tous les flow vectors de toutes les vidéos, applique un filtre 
avec un seuil minimum de norme égal à 13, compte le nombre de flow vectors 
non nuls dans chaque frame, et affiche un graphique pour chaque vidéo montrant 
l'évolution du nombre de flow vectors au cours du temps.

Usage:
    python analyse_flow_vectors_count.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import time
import mlx.core as mx

from src.utilities.load_flows import load_flows
from src.utilities.paths import get_outputs_dir
from src.core.flow_filter import FlowFilterBatch

def count_non_zero_flow_vectors_with_filter_mlx(flows_mx: mx.array, min_threshold: float = 13.0) -> mx.array:
    """
    Compte le nombre de flow vectors non nuls par frame après application du FlowFilterBatch.
    Travaille uniquement avec des MLX arrays pour optimiser la mémoire.
    
    Args:
        flows_mx: MLX array des flow vectors (n_frames, height, width, 2)
        min_threshold: Seuil minimum de norme pour considérer un vecteur comme valide
        
    Returns:
        MLX array du nombre de vecteurs valides par frame (n_frames,)
    """
    print(f"   🔢 Application du FlowFilterBatch avec seuil {min_threshold}...")
    
    # Configuration du filtre avec seuil minimum de norme
    filter_config = {
        'filtering': {
            'norm': {
                'is_used': True,
                'min_threshold': min_threshold
            }
        }
    }
    
    # Créer le filtre
    flow_filter = FlowFilterBatch(filter_config)
    
    # Appliquer le filtrage directement sur l'array MLX
    filtered_flows_mx = flow_filter.filter(flows_mx)
    mx.eval(filtered_flows_mx)
    
    # Compter les vecteurs non nuls directement en MLX
    # Le filtre met les vecteurs filtrés à zéro
    # On compte les pixels où au moins une composante (x ou y) est non nulle
    non_zero_x = filtered_flows_mx[..., 0] != 0
    non_zero_y = filtered_flows_mx[..., 1] != 0
    non_zero_vectors = non_zero_x | non_zero_y  # Union logique
    
    # Compter par frame
    counts_per_frame = mx.sum(non_zero_vectors, axis=(1, 2))
    mx.eval(counts_per_frame)
    
    # Nettoyage mémoire
    del filtered_flows_mx, non_zero_x, non_zero_y, non_zero_vectors
    
    print(f"   ✅ Filtrage terminé. Max: {mx.max(counts_per_frame)}, Min: {mx.min(counts_per_frame)}")
    
    return counts_per_frame

def analyze_single_video(video_id: int, min_threshold: float = 13.0, chunk_size: int = 100) -> Tuple[mx.array, Dict]:
    """
    Analyse une seule vidéo par chunks pour éviter les problèmes de mémoire GPU.
    Optimisé pour travailler uniquement avec MLX arrays.
    
    Args:
        video_id: ID de la vidéo à analyser
        min_threshold: Seuil minimum de norme
        chunk_size: Taille des chunks pour éviter le manque de mémoire GPU
        
    Returns:
        Tuple (counts_per_frame, stats_dict)
    """
    print(f"\n📹 Analyse de la vidéo {video_id}...")
    
    # Charger les flow vectors directement en MLX
    flows_mx = load_flows(video_id, use_compressed=False, verbose=True, return_mlx=True)
    
    if flows_mx is None:
        print(f"❌ Impossible de charger les flows pour la vidéo {video_id}")
        return None, None
    
    total_frames = flows_mx.shape[0]
    print(f"   🔄 Traitement par chunks de {chunk_size} frames (total: {total_frames})")
    
    # Traiter par chunks pour éviter les problèmes de mémoire
    all_counts = []
    
    for start_idx in range(0, total_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_frames = end_idx - start_idx
        
        print(f"      📦 Chunk {start_idx}-{end_idx-1} ({chunk_frames} frames)")
        
        # Extraire le chunk
        flows_chunk = flows_mx[start_idx:end_idx]
        
        # Compter les vecteurs valides pour ce chunk
        counts_chunk = count_non_zero_flow_vectors_with_filter_mlx(flows_chunk, min_threshold)
        
        # Stocker les résultats
        all_counts.append(counts_chunk)
        
        # Nettoyer la mémoire du chunk
        del flows_chunk, counts_chunk
    
    # Nettoyer flows_mx maintenant qu'on n'en a plus besoin
    del flows_mx
    
    # Concaténer tous les résultats
    counts_mx = mx.concatenate(all_counts, axis=0)
    mx.eval(counts_mx)
    
    # Nettoyer la liste des chunks
    del all_counts
    
    # Calculer des statistiques directement en MLX
    total_frames = counts_mx.shape[0]
    mean_count = float(mx.mean(counts_mx))
    std_count = float(mx.std(counts_mx))
    min_count = int(mx.min(counts_mx))
    max_count = int(mx.max(counts_mx))
    
    # Pour la médiane, on doit trier (pas disponible en MLX natif, utiliser numpy temporairement)
    counts_np_temp = np.array(counts_mx)
    median_count = float(np.median(counts_np_temp))
    del counts_np_temp  # Supprimer immédiatement
    
    stats = {
        'video_id': video_id,
        'total_frames': total_frames,
        'mean_count': mean_count,
        'std_count': std_count,
        'min_count': min_count,
        'max_count': max_count,
        'median_count': median_count
    }
    
    print(f"   📊 Statistiques vidéo {video_id}:")
    print(f"      - Frames totales: {stats['total_frames']}")
    print(f"      - Moyenne: {stats['mean_count']:.1f} vecteurs/frame")
    print(f"      - Médiane: {stats['median_count']:.1f} vecteurs/frame")
    print(f"      - Min: {stats['min_count']} vecteurs/frame")
    print(f"      - Max: {stats['max_count']} vecteurs/frame")
    print(f"      - Écart-type: {stats['std_count']:.1f}")
    
    return counts_mx, stats

def plot_flow_counts_all_videos(all_counts: Dict[int, mx.array], 
                               all_stats: Dict[int, Dict],
                               min_threshold: float = 13.0):
    """
    Crée des graphiques pour visualiser le nombre de flow vectors par frame pour toutes les vidéos.
    
    Args:
        all_counts: Dictionnaire {video_id: counts_per_frame}
        all_stats: Dictionnaire {video_id: stats_dict}
        min_threshold: Seuil utilisé pour le filtrage
    """
    print(f"\n📊 Création des graphiques...")
    
    # Configuration des graphiques
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Nombre de flow vectors par frame (seuil minimum: {min_threshold})', fontsize=16)
    
    # Aplanir les axes pour itération plus facile
    axes_flat = axes.flatten()
    
    # Couleurs pour chaque vidéo
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Calculer l'échelle Y globale pour toutes les vidéos
    global_max = max(stats['max_count'] for stats in all_stats.values())
    y_limit = global_max * 1.05  # Ajouter 5% de marge
    
    print(f"   📏 Échelle Y commune: 0 à {y_limit:.0f}")
    
    for i, video_id in enumerate(sorted(all_counts.keys())):
        ax = axes_flat[i]
        counts_mx = all_counts[video_id]
        stats = all_stats[video_id]
        
        # Convertir MLX en numpy seulement pour matplotlib (temporairement)
        counts_np = np.array(counts_mx)
        
        # Créer l'axe temporel (numéros de frame)
        frames = np.arange(len(counts_np))
        
        # Tracer la courbe principale
        ax.plot(frames, counts_np, color=colors[i], linewidth=1, alpha=0.8, label=f'Vidéo {video_id}')
        
        # Nettoyer immédiatement
        del counts_np
        
        # Ajouter la ligne de moyenne
        ax.axhline(y=stats['mean_count'], color=colors[i], linestyle='--', alpha=0.6, 
                  label=f'Moyenne: {stats["mean_count"]:.0f}')
        
        # Configuration de l'axe
        ax.set_title(f'Vidéo {video_id} - {stats["total_frames"]} frames')
        ax.set_xlabel('Numéro de frame')
        ax.set_ylabel('Nombre de flow vectors')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        
        # Utiliser la même échelle Y pour toutes les vidéos
        ax.set_ylim(0, y_limit)
        
        # Ajouter des statistiques en texte
        stats_text = f"Min: {stats['min_count']}\nMax: {stats['max_count']}\nStd: {stats['std_count']:.0f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Masquer le dernier subplot s'il n'est pas utilisé
    if len(all_counts) < 6:
        axes_flat[-1].set_visible(False)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    output_dir = get_outputs_dir()
    output_file = output_dir / f"flow_vectors_count_threshold_{min_threshold}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   💾 Graphique sauvegardé: {output_file}")
    
    plt.show()

def main():
    """Fonction principale du script."""
    print("🚀 ANALYSE DU NOMBRE DE FLOW VECTORS PAR FRAME")
    print("=" * 60)
    
    min_threshold = 13.0
    video_ids = [0, 1, 2, 3, 4]  # Les 5 vidéos labellisées
    
    print(f"⚙️  Configuration:")
    print(f"   - Seuil minimum de norme: {min_threshold}")
    print(f"   - Vidéos à analyser: {video_ids}")
    
    # Analyser toutes les vidéos
    all_counts = {}
    all_stats = {}
    total_start_time = time.time()
    
    for video_id in video_ids:
        counts, stats = analyze_single_video(video_id, min_threshold)
        if counts is not None and stats is not None:
            all_counts[video_id] = counts
            all_stats[video_id] = stats
        else:
            print(f"⚠️  Vidéo {video_id} ignorée due à une erreur de chargement")
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️  Temps total d'analyse: {total_time:.2f}s")
    
    if not all_counts:
        print("❌ Aucune vidéo n'a pu être analysée!")
        return
    
    # Afficher un résumé global
    print(f"\n📋 RÉSUMÉ GLOBAL:")
    print("=" * 50)
    total_frames = sum(stats['total_frames'] for stats in all_stats.values())
    all_means = [stats['mean_count'] for stats in all_stats.values()]
    overall_mean = np.mean(all_means)
    
    print(f"   - Vidéos analysées: {len(all_counts)}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Moyenne globale: {overall_mean:.1f} vecteurs/frame")
    print(f"   - Seuil appliqué: {min_threshold}")
    
    # Créer le graphique principal
    plot_flow_counts_all_videos(all_counts, all_stats, min_threshold)
    
    print(f"\n✅ Analyse terminée! Le graphique a été sauvegardé dans le dossier outputs.")

if __name__ == "__main__":
    main() 