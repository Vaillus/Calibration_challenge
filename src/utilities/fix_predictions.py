"""
Script d'ajustement des prédictions pour résoudre le problème d'optical flow de la frame 0.

PROBLÈME RÉSOLU :
================
L'estimation du point de fuite utilise l'optical flow calculé entre les frames consécutives.
Pour calculer le flow entre la frame N et N-1, il faut avoir au moins 2 frames :
- Frame 0 : PAS de flow possible (pas de frame -1)
- Frame 1 : Flow calculé entre frame 1 et frame 0 ✓
- Frame 2 : Flow calculé entre frame 2 et frame 1 ✓
- etc.

CONSÉQUENCE :
============
Les prédictions commencent à la frame 1, pas à la frame 0.
Résultat : len(predictions) = len(labels) - 1

SOLUTION IMPLÉMENTÉE :
=====================
Ce script duplique la première prédiction (frame 1) et l'ajoute au début pour représenter 
la frame 0. Ainsi :
- Prédiction frame 0 = Prédiction frame 1 (dupliquée)
- Prédiction frame 1 = Prédiction originale frame 1
- etc.

Résultat final : len(predictions) = len(labels) = nombre_total_frames

USAGE :
=======
Automatiquement appelé à la fin de make_predictions.py pour ajuster toutes les prédictions.
"""

import os
import numpy as np

from src.utilities.paths import get_pred_dir, get_labeled_dir

def fix_predictions(pred_dir='pred/3', gt_dir='labeled', mode='labeled'):
    """
    Ajuste les prédictions en dupliquant la première prédiction pour représenter la frame 0 manquante.
    
    Args:
        pred_dir (str): Chemin relatif vers le dossier des prédictions depuis calib_challenge
        gt_dir (str): Chemin relatif vers le dossier des ground truth depuis calib_challenge 
                     (ignoré si mode='unlabeled')
        mode (str): 'labeled' pour vidéos 0-4 (avec GT) ou 'unlabeled' pour vidéos 5-9 (sans GT)
    
    Note:
        La frame 0 ne peut pas avoir de prédiction basée sur l'optical flow car elle nécessite
        une frame précédente. On duplique donc la prédiction de la frame 1 pour la frame 0.
        
        Mode 'labeled': Compare avec GT et ajuste si nécessaire (vidéos 0-4)
        Mode 'unlabeled': Duplique systématiquement la première prédiction (vidéos 5-9)
    """
    # Obtenir le chemin absolu du répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Remonter d'un niveau pour atteindre le répertoire calib_challenge
    base_dir = os.path.dirname(script_dir)

    pred_dir = get_pred_dir(pred_dir)
    
    # Déterminer la plage de vidéos selon le mode
    if mode == 'labeled':
        gt_dir = get_labeled_dir()
        video_range = range(5)  # Vidéos 0-4
        print(f"Mode 'labeled': traitement des vidéos 0-4 avec comparaison GT")
    elif mode == 'unlabeled':
        video_range = range(5, 10)  # Vidéos 5-9
        print(f"Mode 'unlabeled': traitement des vidéos 5-9 sans comparaison GT")
    else:
        raise ValueError(f"Mode '{mode}' non reconnu. Utilisez 'labeled' ou 'unlabeled'")
    
    # Pour chaque vidéo selon le mode
    for video_index in video_range:
        pred_file = pred_dir / f"{video_index}.txt"
        
        # Vérifier si le fichier de prédiction existe
        if not os.path.exists(pred_file):
            print(f"Fichier de prédiction manquant: {pred_file}")
            continue
        
        # Charger les prédictions
        pred = np.loadtxt(pred_file)
        
        if mode == 'labeled':
            # Mode labeled: comparer avec GT
            gt_file = gt_dir / f"{video_index}.txt"
            
            if not os.path.exists(gt_file):
                print(f"Fichier GT manquant: {gt_file}")
                continue
            
            gt = np.loadtxt(gt_file)
            
            # Vérifier si les longueurs sont différentes (typiquement pred = gt - 1)
            if len(pred) != len(gt):
                print(f"Vidéo {video_index}: Ajustement de {len(pred)} à {len(gt)} lignes")
                
                # Dupliquer la première prédiction pour représenter la frame 0
                first_pred = pred[0:1]  # Garder la forme (1, 2) pour pitch/yaw
                pred = np.vstack([first_pred, pred])  # [frame0_pred, frame1_pred, frame2_pred, ...]
                
                # Sauvegarder les prédictions ajustées
                np.savetxt(pred_file, pred, fmt='%.6f')
            else:
                print(f"Vidéo {video_index}: Déjà correct ({len(pred)} lignes)")
                
        elif mode == 'unlabeled':
            # Mode unlabeled: dupliquer systématiquement la première prédiction
            print(f"Vidéo {video_index}: Duplication de la première prédiction ({len(pred)} -> {len(pred)+1} lignes)")
            
            # Dupliquer la première prédiction pour représenter la frame 0
            first_pred = pred[0:1]  # Garder la forme (1, 2) pour pitch/yaw
            pred = np.vstack([first_pred, pred])  # [frame0_pred, frame1_pred, frame2_pred, ...]
            
            # Sauvegarder les prédictions ajustées
            np.savetxt(pred_file, pred, fmt='%.6f')
    
    print("\nAjustement terminé!")

# Si le script est exécuté directement
if __name__ == "__main__":
    pred_dir = '5_7'
    # Exemple d'utilisation pour vidéos labellisées (0-4)
    # fix_predictions(pred_dir=pred_dir, mode='labeled')
    
    # Exemple d'utilisation pour vidéos non labellisées (5-9)
    fix_predictions(pred_dir=pred_dir, mode='unlabeled')