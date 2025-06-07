import numpy as np

FOCAL_LENGTH = 910
HEIGHT = 874
WIDTH = 1164

def angles_to_pixels(yaw, pitch, focal_length=FOCAL_LENGTH, image_width=WIDTH, image_height=HEIGHT):
    """
    Convertit les angles en radians en coordonnées en pixels.
    
    Les angles sont donnés dans le repère Car [Forward, Right, Down] et doivent être
    convertis dans le repère Camera [Right, Down, Forward].
    
    Args:
        yaw: Angle horizontal en radians dans le repère Car
            - yaw > 0 : virage à droite
            - yaw < 0 : virage à gauche
        pitch: Angle vertical en radians dans le repère Car
            - pitch > 0 : montée
            - pitch < 0 : descente
        focal_length: Focale en pixels
        image_width: Largeur de l'image en pixels
        image_height: Hauteur de l'image en pixels
    
    Returns:
        tuple: (x, y) où:
            x: Position x en pixels (0 = gauche, image_width = droite)
            y: Position y en pixels (0 = haut, image_height = bas)
    """
    # Dans le repère Car [Forward, Right, Down]:
    # - yaw est la rotation autour de l'axe Down
    # - pitch est la rotation autour de l'axe Right
    
    # Pour convertir en repère Camera [Right, Down, Forward]:
    # - Le yaw devient une rotation autour de l'axe Forward
    # - Le pitch reste une rotation autour de l'axe Right
    
    # Calculer les coordonnées relatives au centre de l'image
    # Note: Dans le repère Camera, les coordonnées sont [Right, Down, Forward]
    # donc x correspond à Right et y correspond à Down
    x_centered = focal_length * np.tan(yaw)  # Projection sur l'axe Right
    y_centered = -focal_length * np.tan(pitch)    # Projection sur l'axe Down
    
    # Convertir en coordonnées absolues
    x = x_centered + image_width / 2
    y = y_centered + image_height / 2
    
    # Vérifier que les coordonnées sont dans les limites de l'image
    x = np.clip(x, 0, image_width - 1)
    y = np.clip(y, 0, image_height - 1)
    
    # Si les coordonnées sont NaN, retourner le centre de l'image
    if np.isnan(x) or np.isnan(y):
        return image_width // 2, image_height // 2
    
    return int(x), int(y)

def pixels_to_angles(x, y, focal_length=FOCAL_LENGTH, image_width=WIDTH, image_height=HEIGHT):
    """
    Convertit les coordonnées en pixels en angles en radians.
    
    Args:
        x: Position x en pixels (0 = gauche, image_width = droite)
        y: Position y en pixels (0 = haut, image_height = bas)
        focal_length: Focale en pixels
        image_width: Largeur de l'image en pixels
        image_height: Hauteur de l'image en pixels
    
    Returns:
        tuple: (yaw, pitch) où:
            yaw: Angle horizontal en radians
                - yaw > 0 : virage à droite
                - yaw < 0 : virage à gauche
            pitch: Angle vertical en radians
                - pitch > 0 : montée
                - pitch < 0 : descente
    """
    # Convertir en coordonnées relatives au centre de l'image
    x_centered = x - image_width / 2
    y_centered = y - image_height / 2
    
    # Calculer les angles en utilisant arctan
    # Note: On utilise -y_centered car l'axe Y de l'image est inversé
    yaw = np.arctan2(x_centered, focal_length)
    pitch = np.arctan2(-y_centered, focal_length)
    
    return yaw, pitch 