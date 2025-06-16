import numpy as np
from pathlib import Path
import json

from src.utilities.paths import get_labeled_dir, get_pred_dir, get_visualizations_dir, ensure_dir_exists
from src.utilities.pixel_angle_converter import angles_to_pixels

DEFAULT_FOCAL_LENGTH = 910  # Focal length in pixels
IMAGE_WIDTH = 1920  # Standard image width
IMAGE_HEIGHT = 1080  # Standard image height

def load_distances_with_frame_info(video_id, run_name="vanilla"):
    """Load and calculate distances for a specific video, keeping track of frame indices."""
    # Load ground truth
    gt_path = get_labeled_dir() / f"{video_id}.txt"
    if not gt_path.exists():
        return None, None
    
    # Load predictions
    pred_path = get_pred_dir(run_name) / f"{video_id}.txt"
    if not pred_path.exists():
        return None, None
    
    # Load data
    gt_data = np.loadtxt(gt_path)  # (pitch, yaw) in radians
    pred_data = np.loadtxt(pred_path)  # (pitch, yaw) in radians
    
    # Ensure same number of frames
    min_frames = min(len(gt_data), len(pred_data))
    gt_data = gt_data[:min_frames]
    pred_data = pred_data[:min_frames]
    
    # Calculate distances for valid frames only
    valid_distances = []
    valid_frame_indices = []
    
    for i in range(min_frames):
        # Skip if prediction has NaN or if ground truth has NaN
        if np.isnan(pred_data[i]).any() or np.isnan(gt_data[i]).any():
            continue
            
        # Convert angles to pixels
        pitch_gt, yaw_gt = gt_data[i]
        x_gt, y_gt = angles_to_pixels(yaw_gt, pitch_gt, DEFAULT_FOCAL_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        pitch_pred, yaw_pred = pred_data[i]
        x_pred, y_pred = angles_to_pixels(yaw_pred, pitch_pred, DEFAULT_FOCAL_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        # Calculate distance
        distance = np.sqrt((x_gt - x_pred)**2 + (y_gt - y_pred)**2)
        valid_distances.append(distance)
        valid_frame_indices.append(i)
    
    if valid_distances:
        return np.array(valid_distances), np.array(valid_frame_indices)
    else:
        return None, None

def save_worst_errors_global(run_name="vanilla", k=4, file_format="json"):
    """
    Sauvegarde les k pires erreurs globales (toutes vidéos confondues).
    """
    print(f"\n💾 SAUVEGARDE DES {k} PIRES ERREURS GLOBALES - RUN: {run_name}")
    print(f"{'='*60}")
    
    # Récupérer toutes les erreurs avec coordonnées
    all_errors = []
    
    for video_id in range(5):  # Videos 0 to 4
        distances, frame_indices = load_distances_with_frame_info(video_id, run_name)
        
        if distances is not None and len(distances) > 0:
            for dist, frame_id in zip(distances, frame_indices):
                all_errors.append((dist, video_id, frame_id))
            print(f"Video {video_id}: {len(distances)} frames valides chargés")
        else:
            print(f"Video {video_id}: Aucune donnée valide")
    
    if not all_errors:
        print("❌ Aucune erreur trouvée!")
        return None
    
    # Trier par distance décroissante et prendre les k premiers
    all_errors.sort(key=lambda x: x[0], reverse=True)
    worst_errors = all_errors[:k]
    
    # Préparer les données pour la sauvegarde
    error_data = []
    for i, (distance, video_id, frame_id) in enumerate(worst_errors):
        error_data.append({
            "rank": i + 1,
            "video_id": int(video_id),
            "frame_id": int(frame_id),
            "distance_pixels": float(distance)
        })
    
    # Créer le dossier de sortie et sauvegarder
    output_file = _save_to_file(error_data, run_name, k, file_format, "global")
    
    # Afficher aperçu
    _show_summary(error_data, output_file, run_name, k, "globales")
    
    return output_file

def save_worst_errors_per_video(run_name="vanilla", k_per_video=2, file_format="json"):
    """
    Sauvegarde les k pires erreurs par vidéo.
    """
    print(f"\n💾 SAUVEGARDE DES {k_per_video} PIRES ERREURS PAR VIDÉO - RUN: {run_name}")
    print(f"{'='*60}")
    
    all_error_data = []
    
    for video_id in range(5):  # Videos 0 to 4
        distances, frame_indices = load_distances_with_frame_info(video_id, run_name)
        
        if distances is not None and len(distances) > 0:
            # Créer liste des erreurs pour cette vidéo
            video_errors = [(dist, video_id, frame_id) for dist, frame_id in zip(distances, frame_indices)]
            
            # Trier par distance décroissante et prendre les k_per_video premiers
            video_errors.sort(key=lambda x: x[0], reverse=True)
            worst_video_errors = video_errors[:k_per_video]
            
            print(f"Video {video_id}: {len(distances)} frames valides, {len(worst_video_errors)} pires erreurs sélectionnées")
            
            # Ajouter à la liste globale avec rang par vidéo
            for i, (distance, vid_id, frame_id) in enumerate(worst_video_errors):
                all_error_data.append({
                    "video_id": int(vid_id),
                    "rank_in_video": i + 1,
                    "frame_id": int(frame_id),
                    "distance_pixels": float(distance)
                })
        else:
            print(f"Video {video_id}: Aucune donnée valide")
    
    if not all_error_data:
        print("❌ Aucune erreur trouvée!")
        return None
    
    # Créer le dossier de sortie et sauvegarder
    output_file = _save_to_file(all_error_data, run_name, k_per_video, file_format, "per_video")
    
    # Afficher aperçu
    _show_summary_per_video(all_error_data, output_file, run_name, k_per_video)
    
    return output_file

def _save_to_file(error_data, run_name, k, file_format, mode):
    """Fonction helper pour sauvegarder dans un fichier."""
    # Créer le dossier de sortie
    output_dir = ensure_dir_exists(get_visualizations_dir() / "worst_errors")
    
    # Nom du fichier
    if file_format.lower() == "json":
        filename = f"worst_{k}_{mode}_errors_{run_name}.json"
        output_file = output_dir / filename
        
        # Sauvegarder en JSON
        save_data = {
            "run_name": run_name,
            "mode": mode,
            "num_errors": k,
            "total_entries": len(error_data),
            "worst_errors": error_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Données sauvegardées en JSON: {output_file}")
        
    else:  # format txt
        filename = f"worst_{k}_{mode}_errors_{run_name}.txt"
        output_file = output_dir / filename
        
        # Headers différents selon le mode
        if mode == "global":
            header = "rang,video_id,frame_id,distance_pixels"
        else:
            header = "video_id,rang_dans_video,frame_id,distance_pixels"
        
        # Sauvegarder en TXT
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Pires erreurs {mode} pour le run: {run_name}\n")
            f.write(f"# Nombre d'erreurs par élément: {k}\n")
            f.write(f"# Format: {header}\n")
            f.write("#\n")
            f.write(f"{header}\n")
            
            for error in error_data:
                if mode == "global":
                    f.write(f"{error['rank']},{error['video_id']},{error['frame_id']},{error['distance_pixels']:.2f}\n")
                else:
                    f.write(f"{error['video_id']},{error['rank_in_video']},{error['frame_id']},{error['distance_pixels']:.2f}\n")
        
        print(f"✅ Données sauvegardées en TXT: {output_file}")
    
    return output_file

def _show_summary(error_data, output_file, run_name, k, error_type):
    """Afficher résumé pour erreurs globales."""
    print(f"\n📋 RÉSUMÉ DE LA SAUVEGARDE:")
    print(f"   • Fichier créé: {output_file.name}")
    print(f"   • Dossier: {output_file.parent}")
    print(f"   • Nombre d'erreurs {error_type}: {k}")
    print(f"   • Run analysé: {run_name}")
    
    print(f"\n🔍 APERÇU DES DONNÉES SAUVEGARDÉES:")
    print(f"{'Rang':<6} {'Vidéo':<8} {'Frame':<8} {'Distance':<12}")
    print(f"{'-'*36}")
    for error in error_data:
        print(f"{error['rank']:<6} {error['video_id']:<8} {error['frame_id']:<8} {error['distance_pixels']:<12.1f}")

def _show_summary_per_video(error_data, output_file, run_name, k_per_video):
    """Afficher résumé pour erreurs par vidéo."""
    print(f"\n📋 RÉSUMÉ DE LA SAUVEGARDE:")
    print(f"   • Fichier créé: {output_file.name}")
    print(f"   • Dossier: {output_file.parent}")
    print(f"   • Erreurs par vidéo: {k_per_video}")
    print(f"   • Run analysé: {run_name}")
    print(f"   • Total d'entrées: {len(error_data)}")
    
    print(f"\n🔍 APERÇU DES DONNÉES SAUVEGARDÉES:")
    print(f"{'Vidéo':<8} {'Rang':<6} {'Frame':<8} {'Distance':<12}")
    print(f"{'-'*36}")
    for error in error_data:
        print(f"{error['video_id']:<8} {error['rank_in_video']:<6} {error['frame_id']:<8} {error['distance_pixels']:<12.1f}")

def main():
    """Interface principale pour sauvegarder les pires erreurs."""
    print("=== SAUVEGARDE DES PIRES ERREURS ===")
    
    # Demander le run
    run_name = input("Entrez le nom du run à analyser (défaut: 'vanilla'): ").strip()
    if not run_name:
        run_name = "vanilla"
    
    # Demander le mode
    print("\nMode de sauvegarde:")
    print("1. Pires erreurs globales (toutes vidéos confondues)")
    print("2. Pires erreurs par vidéo")
    
    mode_choice = input("Votre choix (1 ou 2): ").strip()
    
    if mode_choice == "1":
        # Mode global
        k_input = input("Combien de pires erreurs globales ? (défaut: 4): ").strip()
        try:
            k = int(k_input) if k_input else 4
            k = max(1, k)
        except ValueError:
            k = 4
            print("⚠️  Valeur invalide, utilisation de 4 par défaut")
        
        # Format de fichier
        format_choice = input("Format de fichier (json/txt, défaut: json): ").strip().lower()
        if format_choice not in ['json', 'txt']:
            format_choice = 'json'
        
        # Sauvegarder
        output_file = save_worst_errors_global(run_name, k, format_choice)
        
    elif mode_choice == "2":
        # Mode par vidéo
        k_input = input("Combien de pires erreurs par vidéo ? (défaut: 2): ").strip()
        try:
            k_per_video = int(k_input) if k_input else 2
            k_per_video = max(1, k_per_video)
        except ValueError:
            k_per_video = 2
            print("⚠️  Valeur invalide, utilisation de 2 par défaut")
        
        # Format de fichier
        format_choice = input("Format de fichier (json/txt, défaut: json): ").strip().lower()
        if format_choice not in ['json', 'txt']:
            format_choice = 'json'
        
        # Sauvegarder
        output_file = save_worst_errors_per_video(run_name, k_per_video, format_choice)
        
    else:
        print("❌ Choix invalide!")
        return
    
    if output_file:
        print(f"\n🎉 Sauvegarde terminée avec succès!")
        print(f"📁 Fichier: {output_file}")
    else:
        print("\n❌ Échec de la sauvegarde!")

if __name__ == "__main__":
    main() 