"""
Visualisation de la distribution des points en pixels pour chaque vidéo.
Compare les prédictions avec les ground truth en affichant :
- Les points individuels
- Les moyennes et médianes
- Les écarts-types sous forme d'ellipses
- Le point central de référence
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from src.utilities.paths import get_labeled_dir, get_pred_dir
from src.utilities.pixel_angle_converter import angles_to_pixels


def create_legends():
    """Crée les légendes pour les couleurs et les symboles."""
    color_legend = [
        Line2D([0], [0], color='red', label='Prédictions'),
        Line2D([0], [0], color='blue', label='Labels'),
        Line2D([0], [0], color='black', label='Centre')
    ]

    symbol_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Moyenne'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Médiane'),
        Line2D([0], [0], marker='+', color='w', markerfacecolor='black', markersize=10, label='Point central'),
        Line2D([0], [0], linestyle=':', color='gray', label='Écart-type')
    ]
    
    return color_legend + symbol_legend


def plot_video_distribution(ax, gt_pixels, pred_pixels):
    """Trace la distribution des points pour une vidéo donnée."""
    # Calcul des statistiques
    pred_mean = np.mean(pred_pixels, axis=0)
    gt_mean = np.mean(gt_pixels, axis=0)
    pred_median = np.median(pred_pixels, axis=0)
    gt_median = np.median(gt_pixels, axis=0)
    pred_std = np.std(pred_pixels, axis=0)
    gt_std = np.std(gt_pixels, axis=0)
    
    # Points individuels
    ax.scatter(pred_pixels[:, 0], pred_pixels[:, 1], c='red', alpha=0.1, s=50)
    ax.scatter(gt_pixels[:, 0], gt_pixels[:, 1], c='blue', alpha=0.1, s=50)
    
    # Moyennes (cercles)
    ax.plot(pred_mean[0], pred_mean[1], 'ro', markersize=15)
    ax.plot(gt_mean[0], gt_mean[1], 'bo', markersize=15)
    
    # Médianes (triangles)
    ax.plot(pred_median[0], pred_median[1], 'r^', markersize=15)
    ax.plot(gt_median[0], gt_median[1], 'b^', markersize=15)
    
    # Écarts-types (ellipses)
    ellipse_pred = Ellipse((pred_mean[0], pred_mean[1]), 
                          2*pred_std[0], 2*pred_std[1],
                          fill=False, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ellipse_gt = Ellipse((gt_mean[0], gt_mean[1]), 
                        2*gt_std[0], 2*gt_std[1],
                        fill=False, color='blue', linestyle=':', linewidth=2, alpha=0.8)
    ax.add_patch(ellipse_pred)
    ax.add_patch(ellipse_gt)
    
    # Point central
    center_pixels = angles_to_pixels(0, 0)
    ax.axvline(x=center_pixels[0], color='black', linestyle='--', alpha=0.7)
    ax.axhline(y=center_pixels[1], color='black', linestyle='--', alpha=0.7)
    ax.plot(center_pixels[0], center_pixels[1], 'k+', markersize=15, markeredgewidth=3)
    
    ax.grid(True)


def main():
    """Fonction principale qui crée la visualisation pour toutes les vidéos."""
    # Création de la figure avec sous-graphiques
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Distribution des points (pixels) par vidéo', fontsize=16)

    # Traitement de chaque vidéo
    for video_idx in range(5):
        try:
            # Chargement des données
            gt = np.loadtxt(get_labeled_dir() / f"{video_idx}.txt")
            pred = np.loadtxt(get_pred_dir(5) / f"{video_idx}.txt")
            
            # Conversion angles vers pixels
            pred_pixels = np.array([angles_to_pixels(yaw, pitch) for pitch, yaw in pred])
            gt_pixels = np.array([angles_to_pixels(yaw, pitch) for pitch, yaw in gt])
            
            # Position du sous-graphique
            row, col = video_idx // 3, video_idx % 3
            ax = axes[row, col]
            
            # Tracé de la distribution
            plot_video_distribution(ax, gt_pixels, pred_pixels)
            ax.set_title(f'Vidéo {video_idx}')
            
        except Exception as e:
            print(f"Video {video_idx}: ERROR - {e}")

    # Ajout des légendes au dernier sous-graphique
    axes[1, 2].legend(handles=create_legends(), fontsize=8, ncol=2)

    # Suppression du dernier sous-graphique inutilisé
    fig.delaxes(axes[1, 2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main() 