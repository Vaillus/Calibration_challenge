#!/usr/bin/env python3
"""
Script utilitaire pour extraire les moyennes des prédictions converties en pixels.

Ce script :
1. Récupère les prédictions d'angles (pitch, yaw) du dossier data/outputs/pred/5/
2. Convertit les angles en coordonnées pixels via la fonction de conversion
3. Calcule la moyenne des prédictions converties (en ignorant les NaN)
4. Sauvegarde les résultats dans data/outputs/means/5/

Usage:
    python src/utilities/extract_means.py

Note: Ce script était précédemment nommé "extract_median.py" mais calculait déjà des moyennes.
Le nom a été corrigé pour refléter le comportement réel (np.nanmean).
"""

import numpy as np
from pathlib import Path
from src.utilities.paths import get_pred_dir, get_means_dir, ensure_dir_exists
from src.utilities.pixel_angle_converter import angles_to_pixels


def load_prediction_file(file_path: Path) -> np.ndarray:
    """Charge un fichier de prédictions.
    
    Args:
        file_path: Chemin vers le fichier de prédictions
        
    Returns:
        Array numpy avec les prédictions [pitch, yaw]
    """
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            try:
                pitch, yaw = map(float, line.split())
                data.append([pitch, yaw])
            except ValueError:
                # Si on ne peut pas parser la ligne, mettre des NaN
                data.append([np.nan, np.nan])
    
    return np.array(data)


def convert_angles_to_pixels(angles_data: np.ndarray) -> np.ndarray:
    """Convertit un array d'angles en coordonnées pixels.
    
    Args:
        angles_data: Array numpy avec les angles [pitch, yaw]
        
    Returns:
        Array numpy avec les coordonnées pixels [x, y]
    """
    pixels_data = []
    
    for pitch, yaw in angles_data:
        if np.isnan(pitch) or np.isnan(yaw):
            pixels_data.append([np.nan, np.nan])
        else:
            x, y = angles_to_pixels(pitch, yaw)
            pixels_data.append([x, y])
    
    return np.array(pixels_data)


def process_video_predictions(video_num: int = 5) -> None:
    """Traite les prédictions d'une vidéo spécifique.
    
    Args:
        video_num: Numéro de la vidéo à traiter (défaut: 5)
    """
    # Utilisation des fonctions de chemins centralisées
    pred_dir = get_pred_dir(video_num)
    output_dir = ensure_dir_exists(get_means_dir(video_num))
    
    # Fichiers à traiter (0 à 4)
    file_numbers = range(5)
    
    for file_num in file_numbers:
        print(f"Traitement du fichier {file_num}...")
        
        # Chemin du fichier de prédiction
        pred_file = pred_dir / f"{file_num}.txt"
        
        if not pred_file.exists():
            print(f"Fichier de prédiction {pred_file} non trouvé, passage au suivant.")
            continue
        
        # Charger les prédictions
        pred_data = load_prediction_file(pred_file)
        print(f"  Lignes totales: {len(pred_data)}")
        
        # Convertir les angles en pixels
        pixels_predictions = convert_angles_to_pixels(pred_data)
        
        # Calculer la moyenne (en ignorant les NaN s'il y en a)
        mean_x = np.nanmean(pixels_predictions[:, 0])
        mean_y = np.nanmean(pixels_predictions[:, 1])
        
        print(f"  Moyenne en pixels: ({mean_x:.2f}, {mean_y:.2f})")
        
        # Sauvegarder les résultats
        output_file = output_dir / f"{file_num}.txt"
        with open(output_file, 'w') as f:
            f.write(f"{mean_x} {mean_y}\n")
        
        # Sauvegarder aussi les prédictions converties en pixels pour référence
        pixels_output_file = output_dir / f"{file_num}_pixels_predictions.txt"
        with open(pixels_output_file, 'w') as f:
            for x, y in pixels_predictions:
                if np.isnan(x) or np.isnan(y):
                    f.write("nan nan\n")
                else:
                    f.write(f"{x} {y}\n")
        
        print(f"  Résultats sauvegardés dans {output_dir}")


def main():
    """Fonction principale."""
    process_video_predictions(video_num=5)


if __name__ == "__main__":
    main() 