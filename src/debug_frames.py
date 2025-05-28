import cv2
import numpy as np
from calib_challenge.src.segmentation import VehicleDetector
from flow import calculate_flow, find_separation_points
from visualization import (
    visualize_flow_arrows,
    visualize_separation_points,
    create_flow_visualization,
    add_frame_info
)

def debug_frames(video_path, frame1_num=184, frame2_num=185):
    """
    Débogue les frames spécifiées avec tous les types de visualisation.
    
    Args:
        video_path: Chemin vers la vidéo
        frame1_num: Numéro de la première frame (frame précédente)
        frame2_num: Numéro de la deuxième frame (frame courante)
    """
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)
    
    # Vérifier que la vidéo est bien ouverte
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la vidéo")
        return
    
    # Lire les frames spécifiées
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame1_num)
    ret, frame1 = cap.read()
    if not ret:
        print(f"Erreur: Impossible de lire la frame {frame1_num}")
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame2_num)
    ret, frame2 = cap.read()
    if not ret:
        print(f"Erreur: Impossible de lire la frame {frame2_num}")
        return
    
    # Initialiser le détecteur de véhicules
    detector = VehicleDetector()
    
    # Convertir les frames en niveaux de gris
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Détecter les véhicules et créer les masques
    vehicle_mask1 = detector.detect_vehicles(frame1)
    vehicle_mask2 = detector.detect_vehicles(frame2)
    vehicle_mask1 = detector.dilate_mask(vehicle_mask1)
    vehicle_mask2 = detector.dilate_mask(vehicle_mask2)
    
    # Créer un masque manuel vide de la même taille que les masques de véhicules
    manual_mask = np.zeros_like(vehicle_mask1)
    
    # Combiner les masques
    combined_mask = detector.combine_masks(manual_mask, vehicle_mask2, vehicle_mask1)
    
    # Calculer le flow optique
    flow, prev_gray_masked, gray_masked = calculate_flow(gray1, gray2, combined_mask)
    
    # Créer les visualisations pour chaque mode
    viz_types = ["1", "2", "3"]
    viz_names = {
        "1": "Flow avec fleches",
        "2": "Points de separation",
        "3": "Debug visualisation"
    }
    
    for viz_type in viz_types:
        if viz_type == "1":
            output = visualize_flow_arrows(frame2, flow, combined_mask)
        elif viz_type == "2":
            best_x, best_y, x_accuracy, y_accuracy = find_separation_points(flow, combined_mask)
            output = visualize_separation_points(frame2, best_x, best_y, x_accuracy, y_accuracy)
            print(f"\nPoints de séparation pour {viz_names[viz_type]}:")
            print(f"X: {best_x}, Y: {best_y}")
            print(f"Accuracy X: {x_accuracy:.2f}, Accuracy Y: {y_accuracy:.2f}")
        else:
            output = create_flow_visualization(flow, combined_mask, frame2)
            cv2.imshow('Previous Frame Masked', prev_gray_masked)
            cv2.imshow('Current Frame Masked', gray_masked)
        
        # Ajouter les informations sur la frame
        output = add_frame_info(output, viz_type, frame2_num, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), True)
        
        # Afficher les résultats
        cv2.imshow(f'Visualization {viz_names[viz_type]}', output)
    
    # Afficher les masques
    cv2.imshow('Vehicle Mask 1', vehicle_mask1)
    cv2.imshow('Vehicle Mask 2', vehicle_mask2)
    cv2.imshow('Combined Mask', combined_mask)
    
    # Afficher les frames originales
    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    
    # Attendre une touche
    cv2.waitKey(0)
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'calib_challenge/labeled/4.hevc'
    debug_frames(video_path) 