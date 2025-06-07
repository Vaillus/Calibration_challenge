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

def fix_predictions(pred_dir='pred/3', gt_dir='labeled'):
    """
    Ajuste les prédictions pour qu'elles aient le même nombre de lignes que les ground truth
    en dupliquant la première prédiction pour représenter la frame 0 manquante.
    
    Args:
        pred_dir (str): Chemin relatif vers le dossier des prédictions depuis calib_challenge
        gt_dir (str): Chemin relatif vers le dossier des ground truth depuis calib_challenge
    
    Note:
        La frame 0 ne peut pas avoir de prédiction basée sur l'optical flow car elle nécessite
        une frame précédente. On duplique donc la prédiction de la frame 1 pour la frame 0.
    """
    # Obtenir le chemin absolu du répertoire du script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Remonter d'un niveau pour atteindre le répertoire calib_challenge
    base_dir = os.path.dirname(script_dir)
    
    # Pour chaque vidéo de 0 à 4
    for video_index in range(5):
        pred_file = os.path.join(base_dir, pred_dir, f"{video_index}.txt")
        gt_file = os.path.join(base_dir, gt_dir, f"{video_index}.txt")
        
        # Vérifier si les fichiers existent
        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            print(f"Fichiers manquants pour la vidéo {video_index}, passage à la suivante...")
            continue
        
        # Charger les prédictions et le ground truth
        pred = np.loadtxt(pred_file)
        gt = np.loadtxt(gt_file)
        
        # Vérifier si les longueurs sont différentes (typiquement pred = gt - 1)
        if len(pred) != len(gt):
            print(f"Vidéo {video_index}: Ajustement de {len(pred)} à {len(gt)} lignes")
            
            # SOLUTION : Dupliquer la première prédiction pour représenter la frame 0
            # pred[0] était la prédiction pour frame 1, elle devient aussi la prédiction pour frame 0
            first_pred = pred[0:1]  # Garder la forme (1, 2) pour yaw/pitch
            pred = np.vstack([first_pred, pred])  # [frame0_pred, frame1_pred, frame2_pred, ...]
            
            # Sauvegarder les prédictions ajustées
            np.savetxt(pred_file, pred, fmt='%.6f')
        else:
            print(f"Vidéo {video_index}: Déjà correct ({len(pred)} lignes)")
    
    print("\nAjustement terminé!")

# Si le script est exécuté directement
if __name__ == "__main__":
    pred_dir = 'pred/4/'
    fix_predictions(pred_dir=pred_dir)