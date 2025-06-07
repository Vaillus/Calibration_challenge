#!/usr/bin/env python3
"""
Script pour analyser et visualiser la distribution des normes des flow vectors par vidéo.

Utilise MLX pour des calculs parallélisés efficaces sur Apple Silicon.

Usage:
    python norm_distribution_by_video.py --videos 0 1 2 3 4 --max_frames 200
    python norm_distribution_by_video.py --videos 3 --max_frames 500 --detailed
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import seaborn as sns
import mlx.core as mx
import time
import gc

from src.utilities.paths import get_outputs_dir, ensure_dir_exists
from src.utilities.load_flows import load_flows as load_flows_util

def load_video_flows_mlx(video_id: int, max_frames: Optional[int] = None, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> Optional[mx.array]:
    """
    Charge les flows d'une vidéo en MLX array pour des calculs efficaces.
    
    Args:
        video_id: ID de la vidéo
        max_frames: Nombre maximum de frames à charger (depuis le début)
        start_frame: Frame de début (optionnel)
        end_frame: Frame de fin (optionnel)
        
    Returns:
        Array MLX des flows ou None si erreur
    """
    # Conversion max_frames vers end_frame si nécessaire
    if max_frames is not None and end_frame is None:
        end_frame = max_frames - 1
    
    # Utiliser la fonction centralisée avec MLX
    flows_mx = load_flows_util(video_id, use_compressed=True, verbose=True, 
                              start_frame=start_frame, end_frame=end_frame, return_mlx=True)
    if flows_mx is None:
        # Fallback vers format original
        flows_mx = load_flows_util(video_id, use_compressed=False, verbose=True,
                                  start_frame=start_frame, end_frame=end_frame, return_mlx=True)
    
    return flows_mx

def compute_norms_mlx_efficient(flows_mx: mx.array) -> mx.array:
    """
    Calcule les normes des flow vectors de manière efficace avec MLX.
    
    Args:
        flows_mx: Array MLX des flows (n_frames, h, w, 2)
        
    Returns:
        Array MLX des normes (n_frames, h, w)
    """
    print(f"   🔢 Calcul des normes MLX...")
    start_time = time.time()
    
    # Calcul vectorisé des normes
    flow_x = flows_mx[:, :, :, 0]
    flow_y = flows_mx[:, :, :, 1]
    norms = mx.sqrt(flow_x**2 + flow_y**2)
    mx.eval(norms)
    
    calc_time = time.time() - start_time
    print(f"   ✅ Normes calculées en {calc_time:.2f}s")
    
    return norms

def analyze_video_norms(video_id: int, max_frames: Optional[int] = None, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> Dict:
    """
    Analyse complète des normes pour une vidéo.
    
    Args:
        video_id: ID de la vidéo
        max_frames: Nombre maximum de frames (depuis le début)
        start_frame: Frame de début (optionnel)
        end_frame: Frame de fin (optionnel)
        
    Returns:
        Dictionnaire avec les statistiques
    """
    print(f"\n{'='*50}")
    print(f"ANALYSE VIDÉO {video_id}")
    print(f"{'='*50}")
    
    # Charger les flows
    flows_mx = load_video_flows_mlx(video_id, max_frames, start_frame, end_frame)
    if flows_mx is None:
        return None
    
    # Calculer les normes
    norms_mx = compute_norms_mlx_efficient(flows_mx)
    
    # Statistiques de base
    print(f"   📊 Calcul des statistiques...")
    flat_norms = norms_mx.flatten()
    mx.eval(flat_norms)
    
    # Convertir en numpy pour filtrage et calculs de percentiles
    flat_norms_np = np.array(flat_norms)
    
    # Exclure les normes nulles pour les statistiques
    non_zero_norms_np = flat_norms_np[flat_norms_np > 1e-10]
    
    stats = {
        'video_id': video_id,
        'total_pixels': int(flat_norms_np.shape[0]),
        'non_zero_pixels': len(non_zero_norms_np),
        'zero_percentage': float(100 * (flat_norms_np.shape[0] - len(non_zero_norms_np)) / flat_norms_np.shape[0]),
        'mean': float(np.mean(non_zero_norms_np)),
        'median': float(np.median(non_zero_norms_np)),
        'std': float(np.std(non_zero_norms_np)),
        'min': float(np.min(non_zero_norms_np)),
        'max': float(np.max(non_zero_norms_np)),
        'p25': float(np.percentile(non_zero_norms_np, 25)),
        'p75': float(np.percentile(non_zero_norms_np, 75)),
        'p90': float(np.percentile(non_zero_norms_np, 90)),
        'p95': float(np.percentile(non_zero_norms_np, 95)),
        'p97': float(np.percentile(non_zero_norms_np, 97)),
        'p98': float(np.percentile(non_zero_norms_np, 98)),
        'p99': float(np.percentile(non_zero_norms_np, 99)),
        'norms_data': non_zero_norms_np  # Pour les plots
    }
    
    # Affichage des stats
    print(f"   • Total pixels: {stats['total_pixels']:,}")
    print(f"   • Pixels avec mouvement: {stats['non_zero_pixels']:,} ({100-stats['zero_percentage']:.1f}%)")
    print(f"   • Moyenne: {stats['mean']:.3f}")
    print(f"   • Médiane: {stats['median']:.3f}")
    print(f"   • P90: {stats['p90']:.3f}")
    print(f"   • P95: {stats['p95']:.3f}")
    print(f"   • P97: {stats['p97']:.3f}")
    print(f"   • P98: {stats['p98']:.3f}")
    print(f"   • P99: {stats['p99']:.3f}")
    print(f"   • Max: {stats['max']:.3f}")
    
    # Nettoyage mémoire
    del flows_mx, norms_mx, flat_norms, flat_norms_np
    gc.collect()
    
    return stats

def plot_comparative_distributions(video_stats: List[Dict], detailed: bool = False):
    """
    Crée des visualisations comparatives des distributions de normes.
    
    Args:
        video_stats: Liste des statistiques par vidéo
        detailed: Si True, affichage plus détaillé
    """
    if not video_stats:
        print("❌ Aucune donnée à visualiser")
        return
    
    print(f"\n📈 Création des visualisations comparatives...")
    
    # Configuration pour de belles visualisations
    plt.style.use('default')
    sns.set_palette("husl")
    
    if detailed:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution des normes par vidéo (analyse détaillée)', fontsize=16, fontweight='bold')
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Distribution des normes par vidéo', fontsize=16, fontweight='bold')
        axes = axes.flatten()
    
    # Préparer les données
    video_ids = [s['video_id'] for s in video_stats]
    colors = plt.cm.Set1(np.linspace(0, 1, len(video_stats)))
    
    # 1. Histogrammes comparatifs
    ax1 = axes[0]
    for i, stats in enumerate(video_stats):
        # Sous-échantillonner si trop de données
        data = stats['norms_data']
        if len(data) > 50000:
            data = np.random.choice(data, 50000, replace=False)
        
        ax1.hist(data, bins=100, alpha=0.6, label=f'Vidéo {stats["video_id"]}', 
                color=colors[i], density=True)
    
    ax1.set_xlabel('Norme des flow vectors')
    ax1.set_ylabel('Densité')
    ax1.set_title('Histogrammes comparatifs (densité)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, min(20, max(s['p99'] for s in video_stats)))
    
    # 2. Histogrammes en log-scale
    ax2 = axes[1]
    for i, stats in enumerate(video_stats):
        data = stats['norms_data']
        if len(data) > 50000:
            data = np.random.choice(data, 50000, replace=False)
        
        ax2.hist(data, bins=100, alpha=0.6, label=f'Vidéo {stats["video_id"]}', 
                color=colors[i], density=True)
    
    ax2.set_xlabel('Norme des flow vectors')
    ax2.set_ylabel('Densité (log)')
    ax2.set_title('Histogrammes comparatifs (échelle log)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, min(20, max(s['p99'] for s in video_stats)))
    
    # 3. Box plots comparatifs
    ax3 = axes[2]
    box_data = []
    box_labels = []
    for stats in video_stats:
        # Sous-échantillonner pour box plot
        data = stats['norms_data']
        if len(data) > 10000:
            data = np.random.choice(data, 10000, replace=False)
        box_data.append(data)
        box_labels.append(f'V{stats["video_id"]}')
    
    bp = ax3.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Vidéos')
    ax3.set_ylabel('Norme des flow vectors')
    ax3.set_title('Box plots comparatifs')
    ax3.grid(True, alpha=0.3)
    
    # 4. Courbes de distribution cumulative
    ax4 = axes[3]
    for i, stats in enumerate(video_stats):
        data = stats['norms_data']
        if len(data) > 20000:
            data = np.random.choice(data, 20000, replace=False)
        
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
        ax4.plot(sorted_data, cumulative, label=f'Vidéo {stats["video_id"]}', 
                color=colors[i], linewidth=2)
    
    ax4.set_xlabel('Norme des flow vectors')
    ax4.set_ylabel('Pourcentage cumulé')
    ax4.set_title('Distributions cumulatives')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, min(15, max(s['p95'] for s in video_stats)))
    
    # Graphiques supplémentaires en mode détaillé
    if detailed and len(axes) > 4:
        # 5. Comparaison des percentiles
        ax5 = axes[4]
        percentiles = ['p90', 'p95', 'p97', 'p98', 'p99']
        x = np.arange(len(percentiles))
        width = 0.8 / len(video_stats)
        
        for i, stats in enumerate(video_stats):
            values = [stats[p] for p in percentiles]
            ax5.bar(x + i * width, values, width, label=f'Vidéo {stats["video_id"]}', 
                   color=colors[i], alpha=0.8)
        
        ax5.set_xlabel('Percentiles')
        ax5.set_ylabel('Norme')
        ax5.set_title('Comparaison des percentiles')
        ax5.set_xticks(x + width * (len(video_stats) - 1) / 2)
        ax5.set_xticklabels(percentiles)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Statistiques générales
        ax6 = axes[5]
        metrics = ['mean', 'std', 'zero_percentage']
        metric_labels = ['Moyenne', 'Écart-type', '% pixels nuls']
        
        x = np.arange(len(metrics))
        width = 0.8 / len(video_stats)
        
        for i, stats in enumerate(video_stats):
            values = [stats[m] for m in metrics]
            # Normaliser le pourcentage pour la visualisation
            if len(metrics) > 2:
                values[2] = values[2] / 10  # Diviser par 10 pour mise à l'échelle
            
            ax6.bar(x + i * width, values, width, label=f'Vidéo {stats["video_id"]}', 
                   color=colors[i], alpha=0.8)
        
        ax6.set_xlabel('Métriques')
        ax6.set_ylabel('Valeur')
        ax6.set_title('Statistiques générales')
        ax6.set_xticks(x + width * (len(video_stats) - 1) / 2)
        ax6.set_xticklabels(metric_labels)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder dans le dossier visualizations
    viz_dir = ensure_dir_exists(get_outputs_dir() / "visualizations")
    suffix = "_detailed" if detailed else ""
    video_list = "_".join(map(str, video_ids))
    output_path = viz_dir / f"norm_distributions_videos_{video_list}{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés: {output_path}")
    
    plt.show()

def print_comparative_summary(video_stats: List[Dict]):
    """
    Affiche un résumé comparatif des statistiques.
    
    Args:
        video_stats: Liste des statistiques par vidéo
    """
    if not video_stats:
        return
    
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ COMPARATIF DES DISTRIBUTIONS")
    print(f"{'='*80}")
    
    # Tableau comparatif
    print(f"\n📊 STATISTIQUES PAR VIDÉO:")
    print(f"{'Vidéo':>5} {'Frames':>8} {'% Mvt':>6} {'Moy':>6} {'Méd':>6} {'P90':>6} {'P95':>6} {'P97':>6} {'P98':>6} {'P99':>6} {'Max':>7}")
    print(f"{'-'*100}")
    
    for stats in video_stats:
        mvt_pct = 100 - stats['zero_percentage']
        print(f"{stats['video_id']:>5} "
              f"{stats['total_pixels']//1000000:>6}M "
              f"{mvt_pct:>5.1f}% "
              f"{stats['mean']:>6.2f} "
              f"{stats['median']:>6.2f} "
              f"{stats['p90']:>6.2f} "
              f"{stats['p95']:>6.2f} "
              f"{stats['p97']:>6.2f} "
              f"{stats['p98']:>6.2f} "
              f"{stats['p99']:>6.2f} "
              f"{stats['max']:>7.1f}")
    
    # Analyse comparative
    print(f"\n🎯 ANALYSE COMPARATIVE:")
    
    # Vidéo avec le plus de mouvement
    most_motion = max(video_stats, key=lambda x: 100 - x['zero_percentage'])
    print(f"  • Plus de mouvement: Vidéo {most_motion['video_id']} ({100-most_motion['zero_percentage']:.1f}%)")
    
    # Vidéo avec les plus grandes normes
    highest_norms = max(video_stats, key=lambda x: x['p99'])
    print(f"  • Plus grandes normes: Vidéo {highest_norms['video_id']} (P99={highest_norms['p99']:.2f})")
    
    # Vidéo la plus variable
    most_variable = max(video_stats, key=lambda x: x['std'])
    print(f"  • Plus variable: Vidéo {most_variable['video_id']} (σ={most_variable['std']:.2f})")
    
    # Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    avg_p90 = np.mean([s['p90'] for s in video_stats])
    avg_p95 = np.mean([s['p95'] for s in video_stats])
    avg_p97 = np.mean([s['p97'] for s in video_stats])
    avg_p98 = np.mean([s['p98'] for s in video_stats])
    avg_p99 = np.mean([s['p99'] for s in video_stats])
    
    print(f"  • Seuil très conservateur (P90): {avg_p90:.2f}")
    print(f"  • Seuil conservateur (P95): {avg_p95:.2f}")
    print(f"  • Seuil équilibré (P97): {avg_p97:.2f}")
    print(f"  • Seuil agressif (P98): {avg_p98:.2f}")
    print(f"  • Seuil très agressif (P99): {avg_p99:.2f}")
    
    if len(video_stats) > 1:
        norm_ranges = [s['max'] - s['min'] for s in video_stats]
        if max(norm_ranges) > 2 * min(norm_ranges):
            print(f"  • Attention: grande variabilité entre vidéos, considérer des seuils adaptatifs")

def main():
    parser = argparse.ArgumentParser(description="Analyse des distributions de normes par vidéo")
    parser.add_argument('--videos', type=int, nargs='+', default=[0, 1, 2, 3, 4], 
                       help='IDs des vidéos à analyser')
    parser.add_argument('--max_frames', type=int, default=None, 
                       help='Nombre max de frames par vidéo (depuis le début)')
    parser.add_argument('--start_frame', type=int, default=None,
                       help='Frame de début pour analyse spécifique')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='Frame de fin pour analyse spécifique')
    parser.add_argument('--detailed', action='store_true', 
                       help='Affichage détaillé avec plus de graphiques')
    parser.add_argument('--no_plot', action='store_true', 
                       help='Ne pas afficher les graphiques')
    
    args = parser.parse_args()
    
    print(f"🚀 ANALYSE DES DISTRIBUTIONS DE NORMES")
    print(f"Vidéos: {args.videos}")
    print(f"Max frames: {args.max_frames if args.max_frames else 'toutes'}")
    print(f"Mode détaillé: {'OUI' if args.detailed else 'NON'}")
    print(f"{'='*60}")
    
    # Analyser chaque vidéo
    all_stats = []
    total_start_time = time.time()
    
    for video_id in args.videos:
        stats = analyze_video_norms(video_id, args.max_frames, args.start_frame, args.end_frame)
        if stats is not None:
            all_stats.append(stats)
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️ Temps total: {total_time:.2f}s pour {len(all_stats)} vidéos")
    
    if not all_stats:
        print("❌ Aucune vidéo analysée avec succès")
        return
    
    # Afficher le résumé comparatif
    print_comparative_summary(all_stats)
    
    # Créer les visualisations
    if not args.no_plot:
        plot_comparative_distributions(all_stats, detailed=args.detailed)
    
    print(f"\n✅ Analyse terminée !")

if __name__ == "__main__":
    main() 