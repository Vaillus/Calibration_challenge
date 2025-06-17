import numpy as np

from src.utilities.project_constants import get_project_constants

class PixelAngleConverter:
    """
    Convertisseur entre coordonnées pixels et angles.
    Charge les constantes du projet une seule fois à l'initialisation.
    """
    
    def __init__(self):
        """Initialise le convertisseur avec les constantes du projet."""
        project_constants = get_project_constants()
        self.focal_length = project_constants["focal_length"]
        self.image_width = project_constants["frame_width"]
        self.image_height = project_constants["frame_height"]
    
    def angles_to_pixels(
            self, 
            yaw, 
            pitch, 
            focal_length=None, 
            image_width=None, 
            image_height=None
        ):
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
            focal_length: Focale en pixels. Si None, utilise la valeur du projet.
            image_width: Largeur de l'image en pixels. Si None, utilise la valeur du projet.
            image_height: Hauteur de l'image en pixels. Si None, utilise la valeur du projet.
        
        Returns:
            tuple: (x, y) où:
                x: Position x en pixels (0 = gauche, image_width = droite)
                y: Position y en pixels (0 = haut, image_height = bas)
        """
        # Use provided values or fall back to project constants
        focal_length = focal_length or self.focal_length
        image_width = image_width or self.image_width
        image_height = image_height or self.image_height
        
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
    
    def pixels_to_angles(self, x, y, focal_length=None, image_width=None, image_height=None):
        """
        Convertit les coordonnées en pixels en angles en radians.
        
        Args:
            x: Position x en pixels (0 = gauche, image_width = droite)
            y: Position y en pixels (0 = haut, image_height = bas)
            focal_length: Focale en pixels. Si None, utilise la valeur du projet.
            image_width: Largeur de l'image en pixels. Si None, utilise la valeur du projet.
            image_height: Hauteur de l'image en pixels. Si None, utilise la valeur du projet.
        
        Returns:
            tuple: (yaw, pitch) où:
                yaw: Angle horizontal en radians
                    - yaw > 0 : virage à droite
                    - yaw < 0 : virage à gauche
                pitch: Angle vertical en radians
                    - pitch > 0 : montée
                    - pitch < 0 : descente
        """
        # Use provided values or fall back to project constants
        focal_length = focal_length or self.focal_length
        image_width = image_width or self.image_width
        image_height = image_height or self.image_height
        
        # Convertir en coordonnées relatives au centre de l'image
        x_centered = x - image_width / 2
        y_centered = y - image_height / 2
        
        # Calculer les angles en utilisant arctan
        # Note: On utilise -y_centered car l'axe Y de l'image est inversé
        yaw = np.arctan2(x_centered, focal_length)
        pitch = np.arctan2(-y_centered, focal_length)
        
        return yaw, pitch

# Create a singleton instance for convenience
converter = PixelAngleConverter()

# For backward compatibility, expose the methods at module level
angles_to_pixels = converter.angles_to_pixels
pixels_to_angles = converter.pixels_to_angles 