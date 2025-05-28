import os
import numpy as np

def fix_predictions(pred_dir='pred/3', gt_dir='labeled'):
    """
    Ajuste les prédictions pour s'assurer qu'elles ont le même nombre de lignes que les ground truth.
    
    Args:
        pred_dir (str): Chemin relatif vers le dossier des prédictions depuis calib_challenge
        gt_dir (str): Chemin relatif vers le dossier des ground truth depuis calib_challenge
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
        
        # Vérifier si les longueurs sont différentes
        if len(pred) != len(gt):
            print(f"Vidéo {video_index}: Ajustement de {len(pred)} à {len(gt)} lignes")
            # Ajouter la première prédiction au début
            first_pred = pred[0:1]  # Garder la forme (1, 2)
            pred = np.vstack([first_pred, pred])
            # Sauvegarder les prédictions ajustées
            np.savetxt(pred_file, pred, fmt='%.6f')
        else:
            print(f"Vidéo {video_index}: Déjà correct ({len(pred)} lignes)")
    
    print("\nAjustement terminé!")

# Si le script est exécuté directement
if __name__ == "__main__":
    pred_dir = 'pred/4/'
    fix_predictions(pred_dir=pred_dir)