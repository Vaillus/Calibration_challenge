#!/usr/bin/env python3
"""
Analyse de la colinéarité des flow vectors par bins de normes.

Ce script analyse la relation entre la norme des flow vectors et leur score de colinéarité
avec le vecteur qui pointe vers le point de fuite (label).

Usage:
    python colinearity_by_norm_analysis.py --video_id 3 --max_frames 200
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import seaborn as sns
import mlx.core as mx

from src.utilities.load_ground_truth import read_ground_truth_pixels
from src.utilities.paths import get_flows_dir, get_outputs_dir
from src.core.collinearity_scorer_batch import BatchCollinearityScorer

def load_flows_and_labels(video_id: int, max_frames: Optional[int] = None) -> Tuple[Optional[mx.array], Optional[mx.array]]:
    """
    Charge les flows et les labels pour une vidéo donnée, retourne des arrays MLX.
    
    Args:
        video_id: ID de la vidéo
        max_frames: Nombre maximum de frames à charger (None = toutes)
        
    Returns:
        Tuple (flows_mx, labels_mx) ou (None, None) si erreur
    """
    # Charger les flows
    npz_path = get_flows_dir() / f"{video_id}_float16.npz"
    if npz_path.exists():
        print(f"📂 Chargement flows vidéo {video_id} (format NPZ)...")
        with np.load(npz_path) as data:
            flows_np = data['flow']
            if max_frames is not None:
                flows_np = flows_np[:max_frames]
            flows_mx = mx.array(flows_np, dtype=mx.float32)
            print(f"✅ Flows chargés: {flows_mx.shape}")
    else:
        # Essayer le format original
        npy_path = get_flows_dir() / f"{video_id}.npy"
        if npy_path.exists():
            print(f"📂 Chargement flows vidéo {video_id} (format NPY)...")
            flows_np = np.load(npy_path)
            if max_frames is not None:
                flows_np = flows_np[:max_frames]
            flows_mx = mx.array(flows_np, dtype=mx.float32)
            print(f"✅ Flows chargés: {flows_mx.shape}")
        else:
            print(f"❌ Fichier de flows non trouvé pour vidéo {video_id}")
            return None, None
    
    # Charger les labels
    try:
        print(f"📂 Chargement labels vidéo {video_id}...")
        labels_np = read_ground_truth_pixels(video_id)
        # Ajuster les indices (skiper la première frame)
        labels_np = labels_np[1:]  
        if max_frames is not None:
            labels_np = labels_np[:max_frames]
        labels_mx = mx.array(labels_np, dtype=mx.float32)
        print(f"✅ Labels chargés: {labels_mx.shape}")
        return flows_mx, labels_mx
    except Exception as e:
        print(f"❌ Erreur lors du chargement des labels: {e}")
        return None, None

def define_adaptive_norm_bins(norms: np.ndarray, focus_on_large: bool = True) -> Dict:
    """
    Définit des bins avec granularité adaptative - plus de détail sur les grands vecteurs.
    
    Args:
        norms: Array des normes (all frames flattened)
        focus_on_large: Si True, plus de granularité sur les grands vecteurs
        
    Returns:
        Dictionnaire avec les définitions des bins
    """
    print(f"📊 Définition de bins adaptatifs (focus sur les grands vecteurs)...")
    
    # Exclure les normes nulles
    non_zero_norms = norms[norms > 1e-10]
    
    # Calculer des percentiles clés
    p25 = np.percentile(non_zero_norms, 25)
    p40 = np.percentile(non_zero_norms, 40)
    p60 = np.percentile(non_zero_norms, 60)
    p75 = np.percentile(non_zero_norms, 75)
    p85 = np.percentile(non_zero_norms, 85)
    p92 = np.percentile(non_zero_norms, 92)
    p96 = np.percentile(non_zero_norms, 96)
    p99 = np.percentile(non_zero_norms, 99)
    max_norm = np.max(non_zero_norms)
    
    if focus_on_large:
        # Regrouper les petits, plus de granularité sur les grands
        bins = {
            'petits': {
                'min': 0.0,
                'max': float(p40),  # Regroupe très_petites + petites
                'percentile_min': 0,
                'percentile_max': 40,
                'count': 0
            },
            'moyens': {
                'min': float(p40),
                'max': float(p60),
                'percentile_min': 40,
                'percentile_max': 60,
                'count': 0
            },
            'grands': {
                'min': float(p60),
                'max': float(p75),
                'percentile_min': 60,
                'percentile_max': 75,
                'count': 0
            },
            'très_grands': {
                'min': float(p75),
                'max': float(p85),
                'percentile_min': 75,
                'percentile_max': 85,
                'count': 0
            },
            'ultra_grands': {
                'min': float(p85),
                'max': float(p92),
                'percentile_min': 85,
                'percentile_max': 92,
                'count': 0
            },
            'mega_grands': {
                'min': float(p92),
                'max': float(p96),
                'percentile_min': 92,
                'percentile_max': 96,
                'count': 0
            },
            'extrêmes': {
                'min': float(p96),
                'max': float(max_norm),
                'percentile_min': 96,
                'percentile_max': 100,
                'count': 0
            }
        }
    else:
        # Granularité équilibrée
        percentiles = [0, 20, 40, 60, 80, 100]
        thresholds = [0.0] + [np.percentile(non_zero_norms, p) for p in percentiles[1:]]
        bin_names = ['petits', 'moyens', 'grands', 'très_grands', 'extrêmes']
        
        bins = {}
        for i, name in enumerate(bin_names):
            bins[name] = {
                'min': float(thresholds[i]),
                'max': float(thresholds[i + 1]),
                'percentile_min': percentiles[i],
                'percentile_max': percentiles[i + 1],
                'count': 0
            }
    
    # Affichage des bins
    print("🎯 Bins définis (granularité adaptative):")
    for name, bin_info in bins.items():
        print(f"  • {name:12s}: [{bin_info['min']:6.3f}, {bin_info['max']:6.3f}] "
              f"(P{bin_info['percentile_min']:3.0f}-P{bin_info['percentile_max']:3.0f})")
    
    return bins

def compute_colinearity_by_bins_parallel(flows_mx: mx.array, labels_mx: mx.array, bins: Dict, batch_size: int = 50) -> Dict:
    """
    Calcule les scores de colinéarité pour chaque bin de normes en utilisant le traitement parallèle MLX.
    
    Args:
        flows_mx: Array MLX des flows (n_frames, h, w, 2)
        labels_mx: Array MLX des points de fuite (n_frames, 2)
        bins: Définition des bins
        batch_size: Taille des batches pour le traitement
        
    Returns:
        Dictionnaire avec les résultats par bin
    """
    print("🔢 Calcul des scores de colinéarité par bins (version parallèle)...")
    
    # Initialiser l'estimateur parallèle
    estimator = BatchCollinearityScorer(
        frame_width=int(flows_mx.shape[2]),
        frame_height=int(flows_mx.shape[1])
    )
    
    results = {}
    for bin_name in bins.keys():
        results[bin_name] = {
            'colinearity_scores': [],
            'count': 0,
            'mean_score': 0.0,
            'std_score': 0.0,
            'median_score': 0.0
        }
    
    n_frames = flows_mx.shape[0]
    print(f"   Traitement de {n_frames} frames par batches de {batch_size}...")
    
    # Calculer toutes les normes en une fois
    print("   Calcul des normes pour toutes les frames...")
    flow_x = flows_mx[:, :, :, 0]
    flow_y = flows_mx[:, :, :, 1]
    all_norms = mx.sqrt(flow_x**2 + flow_y**2)  # Shape: (n_frames, h, w)
    mx.eval(all_norms)
    
    # Traiter par batches
    for start_idx in range(0, n_frames, batch_size):
        end_idx = min(start_idx + batch_size, n_frames)
        if (end_idx - 1) % 50 == 0 or end_idx == n_frames:
            print(f"   Batch {start_idx}-{end_idx-1}")
        
        # Extraire le batch
        flows_batch = flows_mx[start_idx:end_idx]
        labels_batch = labels_mx[start_idx:end_idx]
        norms_batch = all_norms[start_idx:end_idx]
        
        # Calculer les cartes de colinéarité pour tout le batch
        colinearity_maps = []
        for i in range(flows_batch.shape[0]):
            flow_frame = flows_batch[i]
            label_point = (float(labels_batch[i, 0]), float(labels_batch[i, 1]))
            colinearity_map = estimator.compute_colinearity_map(np.array(flow_frame), label_point)
            colinearity_maps.append(colinearity_map)
        
        # Convertir en array MLX pour traitement vectorisé
        colinearity_batch = mx.stack(colinearity_maps)  # Shape: (batch_size, h, w)
        mx.eval(colinearity_batch)
        
        # Analyser par bins pour ce batch
        for bin_name, bin_info in bins.items():
            # Masque pour les pixels dans ce bin pour tout le batch
            bin_mask = (norms_batch >= bin_info['min']) & (norms_batch < bin_info['max'])
            
            # Si c'est le dernier bin, inclure la borne supérieure
            if bin_name == list(bins.keys())[-1]:
                bin_mask = (norms_batch >= bin_info['min']) & (norms_batch <= bin_info['max'])
            
            # Exclure les pixels sans mouvement
            motion_mask = norms_batch > 1e-10
            final_mask = bin_mask & motion_mask
            
            # Extraire les scores de colinéarité pour ce bin
            if mx.sum(final_mask) > 0:
                # Convertir en numpy pour l'indexage booléen
                final_mask_np = np.array(final_mask)
                colinearity_batch_np = np.array(colinearity_batch)
                bin_scores_np = colinearity_batch_np[final_mask_np]
                results[bin_name]['colinearity_scores'].extend(bin_scores_np.tolist())
                results[bin_name]['count'] += len(bin_scores_np)
    
    # Calculer les statistiques finales
    print("📊 Calcul des statistiques par bin...")
    for bin_name, result in results.items():
        if len(result['colinearity_scores']) > 0:
            scores = np.array(result['colinearity_scores'])
            result['mean_score'] = float(np.mean(scores))
            result['std_score'] = float(np.std(scores))
            result['median_score'] = float(np.median(scores))
            result['count'] = len(scores)
            
            # Ajouts pour analyse plus poussée
            result['percentiles'] = {
                'p10': float(np.percentile(scores, 10)),
                'p25': float(np.percentile(scores, 25)),
                'p75': float(np.percentile(scores, 75)),
                'p90': float(np.percentile(scores, 90))
            }
        else:
            print(f"⚠️ Aucun pixel trouvé pour le bin {bin_name}")
    
    return results

def analyze_colinearity_patterns(bins: Dict, results: Dict) -> Dict:
    """
    Analyse les patterns dans les scores de colinéarité.
    
    Args:
        bins: Définition des bins
        results: Résultats par bin
        
    Returns:
        Dictionnaire avec l'analyse des patterns
    """
    print("🔍 Analyse des patterns de colinéarité...")
    
    analysis = {
        'bin_summary': [],
        'best_bin': None,
        'worst_bin': None,
        'correlation_norm_colinearity': 0.0,
        'recommendations': []
    }
    
    # Créer un résumé par bin
    valid_bins = []
    for bin_name, bin_info in bins.items():
        if bin_name in results and results[bin_name]['count'] > 0:
            result = results[bin_name]
            summary = {
                'name': bin_name,
                'norm_range': (bin_info['min'], bin_info['max']),
                'count': result['count'],
                'mean_score': result['mean_score'],
                'median_score': result['median_score'],
                'std_score': result['std_score']
            }
            analysis['bin_summary'].append(summary)
            valid_bins.append(summary)
    
    if len(valid_bins) == 0:
        print("⚠️ Aucun bin valide trouvé")
        return analysis
    
    # Trouver le meilleur et le pire bin
    analysis['best_bin'] = max(valid_bins, key=lambda x: x['mean_score'])
    analysis['worst_bin'] = min(valid_bins, key=lambda x: x['mean_score'])
    
    # Analyser la corrélation entre norme et colinéarité
    if len(valid_bins) >= 3:
        norm_centers = [(b['norm_range'][0] + b['norm_range'][1]) / 2 for b in valid_bins]
        mean_scores = [b['mean_score'] for b in valid_bins]
        correlation = np.corrcoef(norm_centers, mean_scores)[0, 1]
        analysis['correlation_norm_colinearity'] = float(correlation)
    
    # Générer des recommandations
    if analysis['correlation_norm_colinearity'] > 0.5:
        analysis['recommendations'].append("Les vecteurs de grande norme sont mieux alignés")
    elif analysis['correlation_norm_colinearity'] < -0.5:
        analysis['recommendations'].append("Les vecteurs de petite norme sont mieux alignés")
    else:
        analysis['recommendations'].append("Pas de corrélation claire entre norme et alignement")
    
    if analysis['best_bin']['mean_score'] > 0.7:
        analysis['recommendations'].append(f"Excellent alignement dans le bin '{analysis['best_bin']['name']}'")
    elif analysis['best_bin']['mean_score'] < 0.3:
        analysis['recommendations'].append("Alignement général faible - vérifier la qualité des données")
    
    return analysis

def visualize_results(bins: Dict, results: Dict, analysis: Dict, video_id: int):
    """
    Visualise les résultats de l'analyse.
    
    Args:
        bins: Définition des bins
        results: Résultats par bin
        analysis: Analyse des patterns
        video_id: ID de la vidéo
    """
    print("📈 Création des visualisations...")
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Analyse colinéarité par normes - Vidéo {video_id}', fontsize=16, fontweight='bold')
    
    # Préparer les données pour les graphiques
    valid_bins = [s for s in analysis['bin_summary'] if s['count'] > 0]
    if not valid_bins:
        print("⚠️ Pas de données valides pour la visualisation")
        return
    
    bin_names = [b['name'] for b in valid_bins]
    mean_scores = [b['mean_score'] for b in valid_bins]
    counts = [b['count'] for b in valid_bins]
    norm_ranges = [f"{b['norm_range'][0]:.2f}-{b['norm_range'][1]:.2f}" for b in valid_bins]
    
    # 1. Scores moyens par bin
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(bin_names)), mean_scores, alpha=0.7)
    ax1.set_xlabel('Bins de normes')
    ax1.set_ylabel('Score de colinéarité moyen')
    ax1.set_title('Score de colinéarité moyen par bin')
    ax1.set_xticks(range(len(bin_names)))
    ax1.set_xticklabels(bin_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Colorier le meilleur bin
    if analysis['best_bin']:
        best_idx = next(i for i, b in enumerate(valid_bins) if b['name'] == analysis['best_bin']['name'])
        bars1[best_idx].set_color('green')
        bars1[best_idx].set_alpha(0.9)
    
    # 2. Distribution des comptes par bin
    ax2 = axes[0, 1]
    ax2.bar(range(len(bin_names)), counts, alpha=0.7, color='orange')
    ax2.set_xlabel('Bins de normes')
    ax2.set_ylabel('Nombre de pixels')
    ax2.set_title('Nombre de pixels par bin')
    ax2.set_xticks(range(len(bin_names)))
    ax2.set_xticklabels(bin_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plots des distributions
    ax3 = axes[1, 0]
    scores_lists = []
    box_labels = []
    for bin_name in bin_names:
        if bin_name in results and len(results[bin_name]['colinearity_scores']) > 0:
            scores = results[bin_name]['colinearity_scores']
            # Sous-échantillonner si trop de données
            if len(scores) > 10000:
                scores = np.random.choice(scores, 10000, replace=False)
            scores_lists.append(scores)
            box_labels.append(bin_name)
    
    if scores_lists:
        bp = ax3.boxplot(scores_lists, labels=box_labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax3.set_xlabel('Bins de normes')
        ax3.set_ylabel('Score de colinéarité')
        ax3.set_title('Distribution des scores par bin')
        ax3.grid(True, alpha=0.3)
    
    # 4. Corrélation norme vs colinéarité
    ax4 = axes[1, 1]
    if len(valid_bins) >= 3:
        norm_centers = [(b['norm_range'][0] + b['norm_range'][1]) / 2 for b in valid_bins]
        ax4.scatter(norm_centers, mean_scores, s=100, alpha=0.7)
        
        # Ligne de tendance
        z = np.polyfit(norm_centers, mean_scores, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(norm_centers), max(norm_centers), 100)
        ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        ax4.set_xlabel('Centre du bin (norme)')
        ax4.set_ylabel('Score de colinéarité moyen')
        ax4.set_title(f'Corrélation: r={analysis["correlation_norm_colinearity"]:.3f}')
        ax4.grid(True, alpha=0.3)
        
        # Annoter les points
        for i, name in enumerate(bin_names):
            ax4.annotate(name, (norm_centers[i], mean_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = get_outputs_dir() / f"colinearity_by_norm_analysis_video_{video_id}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Graphiques sauvegardés: {output_path}")
    
    plt.show()

def print_detailed_report(bins: Dict, results: Dict, analysis: Dict, video_id: int):
    """
    Affiche un rapport détaillé de l'analyse.
    
    Args:
        bins: Définition des bins
        results: Résultats par bin
        analysis: Analyse des patterns
        video_id: ID de la vidéo
    """
    print(f"\n{'='*80}")
    print(f"RAPPORT D'ANALYSE COLINÉARITÉ PAR NORMES - VIDÉO {video_id}")
    print(f"{'='*80}")
    
    print(f"\n📊 RÉSUMÉ PAR BIN:")
    print(f"{'Bin':12} {'Range':15} {'Count':>8} {'Mean':>6} {'Median':>6} {'Std':>6}")
    print(f"{'-'*60}")
    
    for summary in analysis['bin_summary']:
        print(f"{summary['name']:12} "
              f"[{summary['norm_range'][0]:5.2f},{summary['norm_range'][1]:5.2f}] "
              f"{summary['count']:>8,} "
              f"{summary['mean_score']:>6.3f} "
              f"{summary['median_score']:>6.3f} "
              f"{summary['std_score']:>6.3f}")
    
    print(f"\n🎯 ANALYSE DES PATTERNS:")
    if analysis['best_bin']:
        print(f"  • Meilleur alignement: {analysis['best_bin']['name']} "
              f"(score moyen: {analysis['best_bin']['mean_score']:.3f})")
    
    if analysis['worst_bin']:
        print(f"  • Pire alignement: {analysis['worst_bin']['name']} "
              f"(score moyen: {analysis['worst_bin']['mean_score']:.3f})")
    
    print(f"  • Corrélation norme ↔ colinéarité: {analysis['correlation_norm_colinearity']:.3f}")
    
    if abs(analysis['correlation_norm_colinearity']) > 0.3:
        trend = "positive" if analysis['correlation_norm_colinearity'] > 0 else "négative"
        strength = "forte" if abs(analysis['correlation_norm_colinearity']) > 0.7 else "modérée"
        print(f"    → Corrélation {strength} {trend}")
    else:
        print(f"    → Pas de corrélation significative")
    
    print(f"\n💡 RECOMMANDATIONS:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    # Recommandations pour le filtrage
    print(f"\n🔧 IMPLICATIONS POUR LE FILTRAGE:")
    if analysis['correlation_norm_colinearity'] > 0.5:
        print(f"  • Privilégier les vecteurs de grande norme dans le filtrage")
        print(f"  • Seuil recommandé: ≥ {analysis['best_bin']['norm_range'][0]:.2f}")
    elif analysis['correlation_norm_colinearity'] < -0.5:
        print(f"  • Les vecteurs de petite norme sont plus fiables")
        print(f"  • Éviter les seuils trop élevés")
    else:
        print(f"  • La norme seule n'est pas un bon critère de filtrage")
        print(f"  • Considérer d'autres critères (colinéarité directe, etc.)")

def main():
    parser = argparse.ArgumentParser(description="Analyse de la colinéarité par bins de normes")
    parser.add_argument('--video_id', type=int, default=3, help='ID de la vidéo à analyser')
    parser.add_argument('--max_frames', type=int, default=100, help='Nombre max de frames à analyser')
    parser.add_argument('--focus_large', action='store_true', default=True, help='Granularité fine sur les grands vecteurs')
    parser.add_argument('--no_plot', action='store_true', help='Ne pas afficher les graphiques')
    
    args = parser.parse_args()
    
    print(f"🚀 ANALYSE COLINÉARITÉ PAR NORMES (GRANULARITÉ ADAPTATIVE)")
    print(f"Vidéo: {args.video_id}")
    print(f"Frames: {args.max_frames if args.max_frames else 'toutes'}")
    print(f"Granularité sur grands vecteurs: {'OUI' if args.focus_large else 'NON'}")
    print(f"{'='*60}")
    
    # Charger les données (maintenant en MLX)
    flows_mx, labels_mx = load_flows_and_labels(args.video_id, args.max_frames)
    if flows_mx is None or labels_mx is None:
        return
    
    # Calculer les normes avec MLX (plus efficace)
    print("🔢 Calcul des normes des flow vectors (MLX)...")
    flow_x = flows_mx[:, :, :, 0]
    flow_y = flows_mx[:, :, :, 1]
    all_norms_mx = mx.sqrt(flow_x**2 + flow_y**2)
    mx.eval(all_norms_mx)
    
    # Convertir en numpy pour définir les bins (besoin des percentiles)
    flat_norms = np.array(all_norms_mx.flatten())
    print(f"✅ Normes calculées pour {len(flat_norms):,} pixels")
    
    # Définir les bins avec granularité adaptative
    bins = define_adaptive_norm_bins(flat_norms, focus_on_large=args.focus_large)
    
    # Calculer la colinéarité par bins (version parallèle optimisée)
    results = compute_colinearity_by_bins_parallel(flows_mx, labels_mx, bins, batch_size=50)
    
    # Analyser les patterns
    analysis = analyze_colinearity_patterns(bins, results)
    
    # Afficher le rapport
    print_detailed_report(bins, results, analysis, args.video_id)
    
    # Visualiser
    if not args.no_plot:
        visualize_results(bins, results, analysis, args.video_id)
    
    print(f"\n✅ Analyse terminée !")

if __name__ == "__main__":
    main() 