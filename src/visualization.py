import cv2
import numpy as np

def create_control_panel(paused):
    """Crée un panneau de contrôle avec des boutons pour changer de visualisation"""
    # Créer une fenêtre pour les contrôles
    cv2.namedWindow('Controls')
    
    # Créer une image noire pour le panneau de contrôle
    control_panel = np.zeros((150, 300, 3), dtype=np.uint8)
    
    # Ajouter du texte pour les instructions
    cv2.putText(control_panel, "Space: Pause/Play", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(control_panel, "1: Flow avec fleches", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(control_panel, "2: Points de separation", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(control_panel, "3: Debug visualisation", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Afficher l'état de pause
    status = "PAUSE" if paused else "PLAY"
    color = (0, 0, 255) if paused else (0, 255, 0)
    cv2.putText(control_panel, f"Status: {status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Afficher le panneau de contrôle
    cv2.imshow('Controls', control_panel)

def visualize_flow_arrows(frame, flow, combined_mask):
    """
    Visualise le flow optique avec des flèches.
    
    Args:
        frame: Image originale
        flow: Matrice de flow optique
        combined_mask: Masque combiné des zones à exclure
    
    Returns:
        Image avec les flèches de flow
    """
    flow_vis = np.zeros_like(frame)
    step = 16
    for y in range(0, frame.shape[0], step):
        for x in range(0, frame.shape[1], step):
            if combined_mask[y, x] == 0:  # Ne pas afficher les flèches dans les zones masquées
                dx, dy = flow[y, x]
                #if abs(dx) > 1 or abs(dy) > 1:  # Ne dessiner que les mouvements significatifs
                cv2.arrowedLine(flow_vis, (x, y), 
                                  (int(x + dx), int(y + dy)),
                                  (0, 255, 0), 1, tipLength=0.3)
    return cv2.addWeighted(frame, 0.7, flow_vis, 0.3, 0)

def visualize_separation_points(frame, best_x, best_y):
    """
    Visualise les points de séparation avec des lignes et des valeurs d'accuracy.
    
    Args:
        frame: Image originale
        best_x: Position X du point de séparation
        best_y: Position Y du point de séparation
        x_accuracy: Accuracy de la séparation X
        y_accuracy: Accuracy de la séparation Y
    
    Returns:
        Image avec les lignes de séparation et les valeurs d'accuracy
    """
    output = frame.copy()
    
    # Dessiner les lignes de séparation
    cv2.line(output, (best_x, 0), (best_x, frame.shape[0]), (0, 0, 255), 2)
    cv2.line(output, (0, best_y), (frame.shape[1], best_y), (255, 0, 0), 2)
    
    # Ajouter les valeurs d'accuracy
    # cv2.putText(output, f"X accuracy: {x_accuracy:.2f}", 
            #    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(output, f"Y accuracy: {y_accuracy:.2f}", 
            #    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return output

def create_flow_visualization(flow, mask, frame):
    """
    Crée une visualisation du flow optique avec code couleur.
    Rouge : zones masquées
    Vert : zones non masquées avec flow
    Bleu : zones non masquées sans flow
    """
    #! débuguer en visualisant les valeurs du flow ici.
    # Calculer la magnitude du flow
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    
    # Créer une image en couleur
    vis = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    
    # Zones masquées en rouge
    vis[mask > 0] = [0, 0, 255]
    
    # Zones non masquées avec flow en vert
    flow_mask = (magnitude > 1) & (mask == 0)
    vis[flow_mask] = [0, 255, 0]
    
    # Zones non masquées sans flow en bleu
    no_flow_mask = (magnitude <= 1) & (mask == 0)
    vis[no_flow_mask] = [255, 0, 0]
    
    return vis

def add_frame_info(output, viz_type, current_frame_number, total_frames, paused):
    """
    Ajoute les informations sur la frame (mode, numéro de frame, état de pause).
    
    Args:
        output: Image à modifier
        viz_type: Type de visualisation actuel
        current_frame_number: Numéro de la frame actuelle
        total_frames: Nombre total de frames
        paused: État de pause
    
    Returns:
        Image avec les informations ajoutées
    """
    # Dictionnaire des noms de visualisation
    viz_names = {
        "1": "Flow avec fleches",
        "2": "Points de separation",
        "3": "Debug visualisation"
    }
    
    # Ajouter le mode de visualisation
    cv2.putText(output, f"Mode: {viz_names[viz_type]}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Ajouter le numéro de frame
    cv2.putText(output, f"Frame: {current_frame_number}/{total_frames}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Ajouter l'indicateur de pause si nécessaire
    if paused:
        cv2.putText(output, "PAUSE", (output.shape[1]//2 - 50, output.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    return output 