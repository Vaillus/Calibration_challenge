import cv2
import numpy as np

def calculate_flow(prev_gray, gray, combined_mask, border_radius=50):
    """
    Calcule le flow optique entre deux frames, en excluant aussi une bande autour des masques.
    
    Args:
        prev_gray: Image précédente en niveaux de gris
        gray: Image courante en niveaux de gris
        combined_mask: Masque combiné des zones à exclure
        border_radius: Rayon de la bordure autour des masques
    
    Returns:
        tuple: (flow, prev_gray_masked, gray_masked) où:
            flow: Matrice de flow optique
            prev_gray_masked: Image précédente masquée
            gray_masked: Image courante masquée
    """
    # Appliquer les masques aux images en niveaux de gris
    prev_gray_masked = prev_gray.copy()
    gray_masked = gray.copy()
    prev_gray_masked[combined_mask > 0] = 0
    gray_masked[combined_mask > 0] = 0
    
    # Calculer le flow optique
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray_masked, gray_masked, None,
        pyr_scale=0.5,
        levels=7,
        winsize=23,
        iterations=3,
        poly_n=5,
        poly_sigma=0.8,
        flags=0
    )
    
    # Créer un masque d'exclusion élargi (bordure autour des masques)
    kernel = np.ones((border_radius*2+1, border_radius*2+1), np.uint8)
    border_mask = cv2.dilate((combined_mask > 0).astype(np.uint8), kernel, iterations=1)
    
    # Mettre à zéro le flow dans les zones masquées ET autour (bordure)
    flow[border_mask > 0] = 0
    
    return flow, prev_gray_masked, gray_masked


def find_separation_points(flow, mask):
    """
    Trouve les points de séparation optimaux dans le flow optique.
    
    Args:
        flow: Matrice de flow optique de forme (H, W, 2)
        mask: Masque binaire où 0 indique les pixels à considérer
    
    Returns:
        tuple: (x, y, x_accuracy, y_accuracy)
    """
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    valid_pixels = (mask == 0)
    
    # Calcul des moyennes des colonnes (pour X)
    column_means = np.zeros(flow_x.shape[1])
    for i in range(flow_x.shape[1]):
        valid_pixels_in_column = valid_pixels[:, i]
        if np.any(valid_pixels_in_column):
            column_means[i] = np.mean(flow_x[valid_pixels_in_column, i])
    
    # Calcul des moyennes des lignes (pour Y)
    row_means = np.zeros(flow_y.shape[0])
    for i in range(flow_y.shape[0]):
        valid_pixels_in_row = valid_pixels[i, :]
        if np.any(valid_pixels_in_row):
            row_means[i] = np.mean(flow_y[i, valid_pixels_in_row])
    
    # Trouver les indices valides
    valid_x = np.abs(column_means) > 1e-2
    valid_y = np.abs(row_means) > 1e-2
    valid_x_indices = np.where(valid_x)[0]
    valid_y_indices = np.where(valid_y)[0]
    
    # Trouver les points de séparation
    best_x = _find_best_separation(column_means, valid_x_indices)
    best_y = _find_best_separation(row_means, valid_y_indices)
    
    return best_x, best_y  

def _find_best_separation(means, valid_indices):
    """
    Trouve le meilleur point de séparation dans un tableau de moyennes.
    
    Args:
        means: Tableau 1D des moyennes (colonnes ou lignes)
        valid_indices: Indices des positions valides à considérer
    
    Returns:
        tuple: (best_pos, best_score) où:
            best_pos: Position optimale de séparation
            best_score: Score de la meilleure séparation
    """
    if len(valid_indices) == 0:
        return None, 0
    
    best_pos = None
    best_score = 0

    for idx, i in enumerate(valid_indices[:-1]):
        left = means[valid_indices[:idx+1]]
        right = means[valid_indices[idx+1:]]
        left_matches = np.sum(left < 0)
        right_matches = np.sum(right > 0)
        n = len(left) + len(right)
        score = (left_matches + right_matches) / n if n > 0 else 0
        # Proportion de négatifs à gauche
        # left_ratio = np.sum(left < 0) / len(left) if len(left) > 0 else 0
        # # Proportion de positifs à droite
        # right_ratio = np.sum(right > 0) / len(right) if len(right) > 0 else 0
        # # Score qui favorise une bonne séparation des deux côtés
        # score = left_ratio * right_ratio
        if score > best_score:
            best_score = score
            best_pos = i
    
    return best_pos