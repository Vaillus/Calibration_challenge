from manim import *
import numpy as np

class CollinearityScoreGif(Scene):
    def construct(self):
        # Configuration de base
        origin = ORIGIN
        
        # Point noir d'origine
        origin_dot = Dot(origin, color=BLACK, radius=0.1)
        
        # Vecteur rouge fixe pointant vers le bas à gauche
        red_vector_end = origin + 2 * (LEFT + DOWN)
        red_vector = Arrow(origin, red_vector_end, color=RED, buff=0, 
                          stroke_width=4, tip_length=0.2)
        
        # Configuration initiale du point vert
        green_radius = 2.5
        green_angle = ValueTracker(0)  # Angle qui va évoluer
        
        # Point vert qui suit l'angle
        green_dot = always_redraw(lambda: Dot(
            origin + green_radius * np.array([
                np.cos(green_angle.get_value()), 
                np.sin(green_angle.get_value()), 
                0
            ]), 
            color=GREEN, radius=0.08
        ))
        
        # Vecteur bleu qui suit le point vert
        blue_vector = always_redraw(lambda: Arrow(
            origin, 
            origin + green_radius * np.array([
                np.cos(green_angle.get_value()), 
                np.sin(green_angle.get_value()), 
                0
            ]), 
            color=BLUE, buff=0, stroke_width=4, tip_length=0.2
        ))
        
        # Arc entre les deux vecteurs
        red_angle = np.arctan2(-1, -1)  # angle fixe du vecteur rouge
        
        arc = always_redraw(lambda: self.create_arc(red_angle, green_angle.get_value()))
        
        # Jauge à droite
        gauge_center = 4 * RIGHT
        gauge_background = Rectangle(width=0.5, height=4, color=GRAY, 
                                   fill_opacity=0.3).move_to(gauge_center)
        
        # Barre de la jauge qui s'ajuste
        gauge_fill = always_redraw(lambda: self.create_gauge_fill(
            gauge_background, red_angle, green_angle.get_value()
        ))
        
        # Texte du score qui s'ajuste
        score_text = always_redraw(lambda: self.create_score_text(
            gauge_background, red_angle, green_angle.get_value()
        ))
        
        # Éléments fixes
        title = Text("Score de Colinéarité", font_size=32)
        title.to_edge(UP)
        
        # Labels fixes en haut à gauche, l'un sous l'autre pour plus de lisibilité
        red_label = Text("Vecteur flux", font_size=18, color=RED)
        red_label.to_edge(UP + LEFT, buff=0.5)
        red_label.shift(DOWN * 0.8)  # Décaler vers le bas pour laisser place au titre
        
        blue_label = Text("Vers épipole", font_size=18, color=BLUE)
        blue_label.next_to(red_label, DOWN, buff=0.3)  # Positionner sous le label rouge
        
        # Graduation de la jauge
        gauge_labels = self.create_gauge_labels(gauge_background)
        
        # Ajout de tous les éléments
        self.add(
            title,
            origin_dot, 
            red_vector, 
            green_dot, 
            blue_vector, 
            arc,
            gauge_background,
            gauge_fill,
            score_text,
            red_label,
            blue_label,
            *gauge_labels
        )
        
        # Animation : rotation complète qui boucle parfaitement
        # Pour 32 frames à 32 fps = 1 seconde exactement
        self.play(
            green_angle.animate.set_value(2 * PI),
            run_time=1,  # 1 seconde pour 32 frames à 32 fps
            rate_func=linear
        )
        
        # Pas de pause pour que ça boucle directement
    
    def create_arc(self, red_angle, green_angle):
        """Crée l'arc entre les deux vecteurs"""
        # Calcul de l'angle le plus court
        angle_diff = green_angle - red_angle
        if angle_diff > PI:
            angle_diff -= 2*PI
        elif angle_diff < -PI:
            angle_diff += 2*PI
            
        arc = Arc(radius=0.8, start_angle=red_angle, angle=angle_diff, 
                 color=YELLOW, stroke_width=3)
        arc.move_arc_center_to(ORIGIN)
        return arc
    
    def create_gauge_fill(self, gauge_background, red_angle, green_angle):
        """Crée la barre de remplissage de la jauge"""
        # Calcul du score de colinéarité
        red_vec_norm = np.array([np.cos(red_angle), np.sin(red_angle)])
        green_vec_norm = np.array([np.cos(green_angle), np.sin(green_angle)])
        collinearity_score = np.dot(red_vec_norm, green_vec_norm)
        
        # Mappage du score [-1,1] sur la hauteur [0,4]
        gauge_height = 4
        score_height = max(0.1, (collinearity_score + 1) * gauge_height / 2)
        
        # Couleur inversée : vert pour score bas (bon), rouge pour score élevé (mauvais)
        # Score de [-1,1] : -1 = vert parfait, +1 = rouge parfait
        # Normaliser le score de [0,1] pour l'interpolation
        score_normalized = (collinearity_score + 1) / 2  # [-1,1] -> [0,1]
        color = interpolate_color(GREEN, RED, score_normalized)
        
        gauge_fill = Rectangle(
            width=0.4, 
            height=score_height, 
            color=color,
            fill_opacity=0.8
        )
        gauge_fill.align_to(gauge_background, DOWN)
        gauge_fill.align_to(gauge_background, LEFT)
        return gauge_fill
    
    def create_score_text(self, gauge_background, red_angle, green_angle):
        """Crée le texte du score"""
        # Calcul du score de colinéarité
        red_vec_norm = np.array([np.cos(red_angle), np.sin(red_angle)])
        green_vec_norm = np.array([np.cos(green_angle), np.sin(green_angle)])
        collinearity_score = np.dot(red_vec_norm, green_vec_norm)
        
        score_text = Text(f"Score: {collinearity_score:.3f}", font_size=24)
        score_text.next_to(gauge_background, DOWN)
        return score_text
    
    def create_gauge_labels(self, gauge_background):
        """Crée les graduations de la jauge"""
        labels = []
        
        # Labels -1, 0, 1
        label_1 = Text("1", font_size=16)
        label_1.next_to(gauge_background.get_top(), RIGHT, buff=0.1)
        
        label_0 = Text("0", font_size=16)
        label_0.next_to(gauge_background.get_center(), RIGHT, buff=0.1)
        
        label_minus1 = Text("-1", font_size=16)
        label_minus1.next_to(gauge_background.get_bottom(), RIGHT, buff=0.1)
        
        return [label_1, label_0, label_minus1] 