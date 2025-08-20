#!/usr/bin/env python3
"""
Animation du score de colinéarité global avec vraies données.
Le point candidat tourne en cercle autour du centre et on voit le score s'ajuster.
"""

import sys
import os
# Ajouter le répertoire racine du projet au path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from manim import *
import numpy as np
from typing import Tuple
import tempfile

# Imports du projet
from src.utilities.load_ground_truth import get_frame_pixel
from src.utilities.paths import get_labeled_dir
from src.utilities.load_video_frame import read_frame_rgb
from src.utilities.load_flows import load_flows
from src.utilities.load_predictions import load_predictions
from src.core.flow_filter import FlowFilterSample


def load_frame_data_for_animation(run_name: str = "5_4", video_idx: int = 4, frame_idx: int = 380):
    """
    Charge les données réelles de la frame pour l'animation.
    """
    print(f"📂 Chargement frame {frame_idx} - vidéo {video_idx}...")
    
    # Load RGB frame
    video_path = get_labeled_dir() / f'{video_idx}.hevc'
    _, frame_rgb = read_frame_rgb(video_path, frame_idx)
    
    # Load flow data
    flow_data = load_flows(video_idx, start_frame=frame_idx-1, end_frame=frame_idx-1)
    flow_data = flow_data[0]  # Remove batch dimension
    
    # Load ground truth
    gt_pixels = get_frame_pixel(video_idx, frame_idx)
    gt_point = (gt_pixels[0], gt_pixels[1])
    
    # Center point
    center_point = (frame_rgb.shape[1] // 2, frame_rgb.shape[0] // 2)
    
    # Apply simple filtering
    filter_config = {
        'norm': {'is_used': True, 'k': 150, 'x0': 13},
        'colinearity': {'is_used': True, 'k': 150, 'x0': 0.96},
        'heatmap': {'is_used': False}
    }
    
    flow_filter = FlowFilterSample(filter_config)
    filtered_flow = flow_filter.filter(flow_data)
    
    # Échantillonner les vecteurs de flux
    h, w = filtered_flow.shape[:2]
    stride = 30
    y, x = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    
    # Get flow vectors at grid points
    fx = filtered_flow[y, x, 0]
    fy = filtered_flow[y, x, 1]
    
    # Filter out zero or near-zero flow values
    magnitude = np.sqrt(fx**2 + fy**2)
    mask = magnitude > 0
    
    # Return sampled points and vectors
    x_coords, y_coords, u_flow, v_flow = x[mask], y[mask], fx[mask], fy[mask]
    
    print(f"✅ Données chargées - GT: ({gt_point[0]:.1f}, {gt_point[1]:.1f})")
    print(f"📊 {len(x_coords)} vecteurs échantillonnés")
    
    return frame_rgb, x_coords, y_coords, u_flow, v_flow, gt_point, center_point


def compute_global_collinearity_score(candidate_point: Tuple[float, float], 
                                    x_coords: np.ndarray, y_coords: np.ndarray,
                                    u_flow: np.ndarray, v_flow: np.ndarray) -> float:
    """
    Calcule le score de colinéarité GLOBAL (moyenne de tous les scores individuels).
    """
    if len(x_coords) == 0:
        return 0.0
    
    total_score = 0.0
    valid_count = 0
    
    for i in range(len(x_coords)):
        # Vector from candidate point to current pixel
        vec_to_point = np.array([x_coords[i] - candidate_point[0], y_coords[i] - candidate_point[1]])
        
        # Flow vector at current point  
        flow_vec = np.array([u_flow[i], v_flow[i]])
        
        # Normalize vectors
        vec_to_point_norm = np.linalg.norm(vec_to_point)
        flow_vec_norm = np.linalg.norm(flow_vec)
        
        # Avoid division by zero
        if vec_to_point_norm > 0 and flow_vec_norm > 0:
            vec_to_point_normalized = vec_to_point / vec_to_point_norm
            flow_vec_normalized = flow_vec / flow_vec_norm
            
            # Compute dot product (colinearity)
            collinearity = np.abs(np.dot(vec_to_point_normalized, flow_vec_normalized))
            total_score += collinearity
            valid_count += 1
    
    return total_score / valid_count if valid_count > 0 else 0.0


class GlobalCollinearityAnimation(Scene):
    def construct(self):
        # Fond blanc
        self.camera.background_color = WHITE
        
        # Charger les vraies données
        frame_rgb, x_coords, y_coords, u_flow, v_flow, gt_point, center_point = load_frame_data_for_animation()
        
        # Configuration de l'affichage
        img_height = 4  # hauteur dans l'espace manim
        img_width = img_height * frame_rgb.shape[1] / frame_rgb.shape[0]
        
        # Créer l'image de fond
        temp_img_path = tempfile.mktemp(suffix='.png')
        from PIL import Image
        Image.fromarray(frame_rgb).save(temp_img_path)
        
        background = ImageMobject(temp_img_path)
        background.height = img_height
        background.width = img_width
        background.move_to(LEFT * 2)
        
        # Nettoyer le fichier temporaire
        os.unlink(temp_img_path)
        
        # Convertir les coordonnées pixel en coordonnées manim
        def pixel_to_manim(px, py):
            manim_x = (px / frame_rgb.shape[1] - 0.5) * img_width + background.get_center()[0]
            manim_y = -(py / frame_rgb.shape[0] - 0.5) * img_height + background.get_center()[1]
            return np.array([manim_x, manim_y, 0])
        
        # Paramètres de l'animation du point candidat (ellipse)
        # Ellipse : petit axe horizontal = 30px, grand axe vertical = 300px
        ellipse_a_pixel = 90  # demi-axe horizontal en pixels
        ellipse_b_pixel = 300  # demi-axe vertical en pixels
        ellipse_a_manim = ellipse_a_pixel * img_width / frame_rgb.shape[1]
        ellipse_b_manim = ellipse_b_pixel * img_height / frame_rgb.shape[0]
        
        # Centre décalé de la moitié du petit axe vers la gauche
        center_offset_pixel = ellipse_a_pixel / 2
        center_offset_manim = center_offset_pixel * img_width / frame_rgb.shape[1]
        center_manim = pixel_to_manim(center_point[0] - center_offset_pixel, center_point[1])
        
        # ValueTracker pour l'angle de rotation
        angle_tracker = ValueTracker(0)
        
        # Point candidat qui suit une trajectoire elliptique
        candidate_dot = always_redraw(lambda: Dot(
            center_manim + np.array([
                ellipse_a_manim * np.cos(angle_tracker.get_value()),
                ellipse_b_manim * np.sin(angle_tracker.get_value()),
                0
            ]),
            color=GREEN, radius=0.08
        ))
        
        # Créer les vecteurs rouges fixes (flux optique)
        red_vectors = VGroup()
        display_stride = 2
        display_indices = np.arange(0, len(x_coords), display_stride)
        
        for i in display_indices:
            px, py = x_coords[i], y_coords[i]
            fx, fy = u_flow[i], v_flow[i]
            
            pixel_manim = pixel_to_manim(px, py)
            flow_scale = 0.015
            flow_end = pixel_manim + np.array([fx * flow_scale, -fy * flow_scale, 0])
            red_arrow = Arrow(pixel_manim, flow_end, color=RED, 
                            stroke_width=2, tip_length=0.08, buff=0)
            red_vectors.add(red_arrow)
        
        # Vecteurs bleus qui s'ajustent (vers candidat)
        blue_vectors = always_redraw(lambda: self.create_blue_vectors(
            angle_tracker.get_value(), center_manim, ellipse_a_manim, ellipse_b_manim,
            x_coords, y_coords, display_indices, pixel_to_manim
        ))
        
        # Jauge à droite
        gauge_center = 4 * RIGHT
        gauge_background = Rectangle(width=0.5, height=4, color=GRAY, 
                                   fill_opacity=0.3).move_to(gauge_center)
        
        # Barre de la jauge qui s'ajuste
        gauge_fill = always_redraw(lambda: self.create_gauge_fill(
            angle_tracker.get_value(), center_manim, ellipse_a_manim, ellipse_b_manim,
            gauge_background, x_coords, y_coords, u_flow, v_flow, 
            frame_rgb.shape[1], frame_rgb.shape[0]
        ))
        
        # Texte du score qui s'ajuste
        score_text = always_redraw(lambda: self.create_score_text(
            angle_tracker.get_value(), center_manim, ellipse_a_manim, ellipse_b_manim,
            gauge_background, x_coords, y_coords, u_flow, v_flow,
            frame_rgb.shape[1], frame_rgb.shape[0]
        ))
        
        # Éléments fixes
        title = Text("Score de Colinéarité Global", font_size=32, color=BLACK)
        title.to_edge(UP)
        
        # Labels fixes en haut à gauche, l'un sous l'autre pour plus de lisibilité
        red_label = Text("Vecteurs flux", font_size=18, color=RED)
        red_label.to_edge(UP + LEFT, buff=0.5)
        red_label.shift(DOWN * 0.8)  # Décaler vers le bas pour laisser place au titre
        
        blue_label = Text("Vers candidat", font_size=18, color=BLUE)
        blue_label.next_to(red_label, DOWN, buff=0.3)  # Positionner sous le label rouge
        
        candidate_label = always_redraw(lambda: Text("Point candidat", font_size=16, color=GREEN).next_to(
            center_manim + np.array([
                ellipse_a_manim * np.cos(angle_tracker.get_value()),
                ellipse_b_manim * np.sin(angle_tracker.get_value()),
                0
            ]), UP, buff=0.1
        ))
        
        # Graduations de la jauge (plage -0.5 à -1.0)
        label_minus05 = Text("-0.5", font_size=16, color=BLACK)
        label_minus05.next_to(gauge_background.get_top(), RIGHT, buff=0.1)
        
        label_minus075 = Text("-0.75", font_size=16, color=BLACK)
        label_minus075.next_to(gauge_background.get_center(), RIGHT, buff=0.1)
        
        label_minus1 = Text("-1.0", font_size=16, color=BLACK)
        label_minus1.next_to(gauge_background.get_bottom(), RIGHT, buff=0.1)
        
        # Points de référence
        gt_manim = pixel_to_manim(gt_point[0], gt_point[1])
        center_manim_dot = Dot(center_manim, color=PURPLE, radius=0.06)
        
        gt_dot = Dot(gt_manim, color=YELLOW, radius=0.08)
        gt_label = Text("Ground Truth", font_size=14, color=YELLOW)
        gt_label.next_to(gt_dot, UP, buff=0.1)
        
        center_label = Text("Centre", font_size=14, color=PURPLE)
        center_label.next_to(center_manim_dot, DOWN, buff=0.1)
        
        # Ajout de tous les éléments
        self.add(
            title,
            background,
            candidate_dot,
            candidate_label,
            gt_dot,
            gt_label,
            center_manim_dot,
            center_label,
            red_vectors,
            blue_vectors,
            gauge_background,
            gauge_fill,
            score_text,
            red_label,
            blue_label,
            label_minus05,
            label_minus075,
            label_minus1
        )
        
        # Animation : rotation complète en 8 secondes - pas de pause
        self.play(
            angle_tracker.animate.set_value(2 * PI),
            run_time=8,
            rate_func=linear
        )
    
    def create_blue_vectors(self, angle, center_manim, ellipse_a, ellipse_b, x_coords, y_coords, 
                          display_indices, pixel_to_manim_func):
        """Crée les vecteurs bleus pointant vers le candidat"""
        vectors = VGroup()
        
        # Position actuelle du point candidat (ellipse)
        candidate_pos = center_manim + np.array([
            ellipse_a * np.cos(angle),
            ellipse_b * np.sin(angle),
            0
        ])
        
        for i in display_indices:
            px, py = x_coords[i], y_coords[i]
            pixel_manim = pixel_to_manim_func(px, py)
            
            # Vecteur vers candidat
            vec_to_candidate = candidate_pos - pixel_manim
            vec_length = np.linalg.norm(vec_to_candidate[:2])
            
            if vec_length > 0:
                blue_scale = 0.25
                vec_normalized = vec_to_candidate / vec_length
                blue_end = pixel_manim + vec_normalized * blue_scale
                blue_arrow = Arrow(pixel_manim, blue_end, color=BLUE,
                                 stroke_width=2, tip_length=0.08, buff=0)
                vectors.add(blue_arrow)
        
        return vectors
    
    def create_gauge_fill(self, angle, center_manim, ellipse_a, ellipse_b, gauge_background,
                         x_coords, y_coords, u_flow, v_flow, img_width, img_height):
        """Crée la barre de remplissage de la jauge"""
        # Calculer la position du candidat en pixels (ellipse)
        ellipse_a_pixel = ellipse_a * img_width / (4 * img_width / img_height)
        ellipse_b_pixel = ellipse_b * img_height / 4
        
        candidate_pixel_x = img_width/2 + ellipse_a_pixel * np.cos(angle) - ellipse_a/2 * img_width / (4 * img_width / img_height)
        candidate_pixel_y = img_height/2 - ellipse_b_pixel * np.sin(angle)
        
        # Calculer le score global (valeur négative pour minimisation)
        global_score_positive = compute_global_collinearity_score(
            (candidate_pixel_x, candidate_pixel_y), x_coords, y_coords, u_flow, v_flow
        )
        global_score = -global_score_positive  # Utiliser la valeur négative
        
        # Créer la barre (plage -1.0 à -0.5)
        gauge_height = 4
        # Mapper le score de [-1.0, -0.5] vers [0, gauge_height]
        score_normalized = max(0, min(1, (global_score - (-1.0)) / 0.5))
        score_height = max(0.1, score_normalized * gauge_height)
        
        # Couleur inversée : vert pour score bas (bon), rouge pour score élevé (mauvais)
        # Score normalisé [0,1] : 0 = vert parfait (score -1.0), 1 = rouge parfait (score -0.5)
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
    
    def create_score_text(self, angle, center_manim, ellipse_a, ellipse_b, gauge_background,
                         x_coords, y_coords, u_flow, v_flow, img_width, img_height):
        """Crée le texte du score"""
        # Calculer la position du candidat en pixels (ellipse - même calcul que gauge_fill)
        ellipse_a_pixel = ellipse_a * img_width / (4 * img_width / img_height)
        ellipse_b_pixel = ellipse_b * img_height / 4
        
        candidate_pixel_x = img_width/2 + ellipse_a_pixel * np.cos(angle) - ellipse_a/2 * img_width / (4 * img_width / img_height)
        candidate_pixel_y = img_height/2 - ellipse_b_pixel * np.sin(angle)
        
        # Calculer le score global (valeur négative pour minimisation)
        global_score_positive = compute_global_collinearity_score(
            (candidate_pixel_x, candidate_pixel_y), x_coords, y_coords, u_flow, v_flow
        )
        global_score = -global_score_positive  # Utiliser la valeur négative
        
        score_text = Text(f"Score: {global_score:.3f}", font_size=24, color=BLACK)
        score_text.next_to(gauge_background, DOWN)
        
        return score_text 