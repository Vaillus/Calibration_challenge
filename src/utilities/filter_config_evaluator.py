#!/usr/bin/env python3
"""
🎯 FILTER CONFIG EVALUATOR

Évaluateur de configurations de filtrage sigmoïde.
Charge les données et évalue les performances d'une configuration donnée.
"""

import time
import numpy as np
import mlx.core as mx
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict, Any
import gc
from dataclasses import dataclass

from src.utilities.paths import get_pred_dir, get_means_dir, get_intermediate_dir
from src.utilities.load_flows import load_flows
from src.utilities.ground_truth import read_ground_truth_pixels
from src.core.flow_filter import FlowFilterBatch
from src.core.optimizers import AdamOptimizer
from src.utilities.pixel_angle_converter import angles_to_pixels


@dataclass
class FrameBatch:
    """Données chargées pour un batch de frames - structure immutable"""
    flows: mx.array                        # (batch_size, h, w, 2)
    labels: mx.array                       # (batch_size, 2) 
    baseline_predictions: mx.array         # (batch_size, 2)
    baseline_distances: mx.array           # (batch_size,)
    mean_points: Optional[mx.array]        # (batch_size, 2) ou None
    frame_metadata: List[Tuple[int, int]]  # [(video_id, frame_id), ...]


class DataLoader:
    """Chargement optimisé de données pour les benchmarks de filtrage"""
    
    def __init__(self, run_name: str = "5", verbose: bool = True):
        """
        Args:
            baseline_pred_gen: Génération de prédictions baseline
            verbose: Affichage des messages de progression
        """
        self.run_name = run_name
        self.verbose = verbose
        
        # Paths
        self.baseline_pred_dir = get_pred_dir(run_name)
        self.means_dir = get_means_dir(run_name)
    
    def load_frame_batch(self, data_source: Union[int, List[Tuple[int, int]], str]) -> FrameBatch:
        """
        Point d'entrée principal - charge toutes les données pour un batch de frames.
        
        Args:
            data_source: 
                - int: video_id pour toute la vidéo
                - [(video_id, frame_id), ...]: batch custom
                - 'all': toutes les vidéos disponibles
                
        Returns:
            FrameBatch: Toutes les données chargées et prêtes à utiliser
        """
        # 1. Déterminer les frames à charger
        frame_list = self._resolve_frame_list(data_source)
        # 2. Extraire les video_ids uniques
        video_ids = self._get_unique_video_ids(frame_list)
        if self.verbose:
            video_ids_sorted = sorted(video_ids)
            print(f"📂 Chargement de {len(frame_list)} frames depuis les vidéos {video_ids_sorted}...")
        
        # 3. Charger toutes les données
        flows = self._load_flows(frame_list, video_ids)
        labels = self._load_labels(frame_list, video_ids)
        baseline_preds = self._load_baseline(frame_list, video_ids)
        mean_points = self._load_mean_points(frame_list, video_ids)
        
        # 4. Calculer distances baseline
        baseline_distances = mx.sqrt(
            mx.sum(mx.square(baseline_preds - labels), axis=1)
        )
        mx.eval(baseline_distances)
        
        if self.verbose:
            print(f"✅ Données chargées: flows{flows.shape}, labels{labels.shape}")
        
        return FrameBatch(
            flows=flows,
            labels=labels,
            baseline_predictions=baseline_preds,
            baseline_distances=baseline_distances,
            mean_points=mean_points,
            frame_metadata=frame_list
        )
    
    def _resolve_frame_list(
            self, 
            data_source: Union[int, List[Tuple[int, int]], str]
        ) -> List[Tuple[int, int]]:
        """
        Détermine la liste des frames à charger selon data_source.
        Extrait de la logique actuelle de _determine_frame_list_and_group.
        """
        if isinstance(data_source, int):
            # Une vidéo complète
            video_id = data_source
            gt_pixels = read_ground_truth_pixels(video_id)
            num_frames = len(gt_pixels) - 1
            return [(video_id, i) for i in range(num_frames)]
                
        elif isinstance(data_source, list):
            # Batch custom
            return data_source
                
        elif data_source == 'all':
            # Toutes les vidéos
            frame_list = []
            for video_id in range(5):  # Vidéos 0-4
                try:
                    gt_pixels = read_ground_truth_pixels(video_id)
                    num_frames = len(gt_pixels)
                    frame_list.extend([(video_id, i) for i in range(num_frames)])
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Erreur vidéo {video_id}: {e}")
            return frame_list
                        
        else:
            raise ValueError(f"data_source non supporté: {data_source}")
    
    def _get_unique_video_ids(self, frame_list: List[Tuple[int, int]]) -> set[int]:
        """
        Extrait les video_ids uniques d'une liste de frames.
        
        Args:
            frame_list: Liste des (video_id, frame_id)
            
        Returns:
            Set[int]: Ensemble des video_ids uniques
        """
        return set(video_id for video_id, _ in frame_list)
    
    def _load_flows(
        self, 
        frame_list: List[Tuple[int, int]], 
        video_ids: set[int]
    ) -> mx.array:
        """Charge flows groupés par vidéo (logique extraite de _load_flows_by_video)."""
        # Charger chaque vidéo une seule fois
        flows_dict = {}
        for video_id in video_ids:
            flows = load_flows(video_id, return_mlx=False, verbose=False)
            if flows is None:
                raise ValueError(f"Impossible de charger flows pour vidéo {video_id}")
            flows_dict[video_id] = flows
        
        # Extraire les frames dans l'ordre de frame_list
        flows_list = []
        for video_id, frame_id in frame_list:
            if frame_id >= len(flows_dict[video_id]):
                raise ValueError(f"Frame {frame_id} non disponible pour vidéo {video_id}")
            flows_list.append(flows_dict[video_id][frame_id-1]) # frame_id-1 
            # car les frames sont indexées à partir de 1 dans le dossier de flows
        
        # Convertir en batch MLX
        flows_np = np.stack(flows_list, axis=0)
        flows_mlx = mx.array(flows_np, dtype=mx.float32)
        mx.eval(flows_mlx)
        
        # Libérer mémoire
        del flows_dict, flows_list, flows_np
        gc.collect()
        
        return flows_mlx
    
    def _load_labels(
            self, 
            frame_list: List[Tuple[int, int]], 
            video_ids: set[int]
        ) -> mx.array:
        """Charge labels groupés par vidéo (logique extraite de _load_labels_by_video)."""
        # Charger chaque vidéo une seule fois
        labels_dict = {}
        for video_id in video_ids:
            gt_pixels = read_ground_truth_pixels(video_id)  # Déjà en pixels !
            labels_dict[video_id] = gt_pixels
        
        # Extraire dans l'ordre
        labels_list = []
        for video_id, frame_id in frame_list:
            if frame_id >= len(labels_dict[video_id]):
                raise ValueError(f"Label {frame_id} non disponible pour vidéo {video_id}")
            
            x, y = labels_dict[video_id][frame_id]
            labels_list.append([x, y])
        
        labels_np = np.array(labels_list, dtype=np.float32)
        labels_mlx = mx.array(labels_np, dtype=mx.float32)
        mx.eval(labels_mlx)
        
        # Libérer mémoire
        del labels_dict, labels_list, labels_np
        gc.collect()
        
        return labels_mlx
    
    def _load_baseline(self, frame_list: List[Tuple[int, int]], video_ids: set[int]) -> mx.array:
        """Charge baseline groupé par vidéo (logique extraite de _load_baseline_by_video)."""
        if not self.baseline_pred_dir.exists():
            raise ValueError(f"Dossier baseline non trouvé: {self.baseline_pred_dir}")
        
        # Charger chaque fichier de prédictions une seule fois
        baseline_dict = {}
        for video_id in video_ids:
            pred_file = self.baseline_pred_dir / f"{video_id}.txt"
            
            if not pred_file.exists():
                raise ValueError(f"Prédictions baseline non trouvées: {pred_file}")
            
            baseline_preds_angles = np.loadtxt(pred_file)
            # Convertir les prédictions baseline (angles) en pixels
            baseline_preds = np.zeros((len(baseline_preds_angles), 2), dtype=np.float32)
            for i, pred in enumerate(baseline_preds_angles):
                pitch, yaw = pred[0], pred[1]  # pred est [pitch, yaw] en radians
                x, y = angles_to_pixels(pitch, yaw)
                baseline_preds[i] = [x, y]
            
            baseline_dict[video_id] = baseline_preds
        
        # Extraire dans l'ordre
        predictions_list = []
        for video_id, frame_id in frame_list:
            if frame_id >= len(baseline_dict[video_id]):
                raise ValueError(f"Prédiction baseline {frame_id} non disponible pour vidéo {video_id}")
            
            predictions_list.append(baseline_dict[video_id][frame_id])
        
        predictions_np = np.array(predictions_list, dtype=np.float32)
        baseline_mlx = mx.array(predictions_np, dtype=mx.float32)
        mx.eval(baseline_mlx)
        
        # Libérer mémoire
        del baseline_dict, predictions_list, predictions_np
        gc.collect()
        
        return baseline_mlx
    
    def _load_mean_points(self, frame_list: List[Tuple[int, int]], video_ids: set[int]) -> Optional[mx.array]:
        """Charge les points moyens groupés par vidéo (logique extraite de _load_mean_points_by_video)."""
        if not self.means_dir.exists():
            return None
        
        # Charger chaque fichier de points moyens une seule fois
        mean_points_dict = {}
        for video_id in video_ids:
            mean_file = self.means_dir / f"{video_id}.txt"
            
            if not mean_file.exists():
                mean_points_dict[video_id] = None
                continue
            
            try:
                # Charger le point moyen (x, y)
                with open(mean_file, 'r') as f:
                    line = f.readline().strip()
                    x, y = map(float, line.split())
                    mean_points_dict[video_id] = (x, y)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Erreur points moyens vidéo {video_id}: {e}")
                mean_points_dict[video_id] = None
        
        # Construire la liste des points moyens dans l'ordre des frames
        mean_points_list = []
        has_valid_points = False
        
        for video_id, frame_id in frame_list:
            if video_id in mean_points_dict and mean_points_dict[video_id] is not None:
                mean_points_list.append(list(mean_points_dict[video_id]))
                has_valid_points = True
            else:
                # Utiliser le centre d'image par défaut (sera calculé dynamiquement)
                mean_points_list.append([0.0, 0.0])  # Placeholder
        
        if has_valid_points:
            mean_points_np = np.array(mean_points_list, dtype=np.float32)
            mean_points_mlx = mx.array(mean_points_np, dtype=mx.float32)
            mx.eval(mean_points_mlx)
            
            # Libérer mémoire
            del mean_points_dict, mean_points_list, mean_points_np
            gc.collect()
            
            return mean_points_mlx
        else:
            return None


class FilterConfigEvaluator:
    """
    Évaluateur de configurations de filtrage sigmoïde.
    Charge les données et évalue les performances d'une configuration donnée.
    
    Modes de données :
    - video_id (int) : toutes les frames d'une vidéo
    - [(video_id, frame_id), ...] : batch custom  
    - 'all' : toutes les vidéos disponibles (0-4)
    """
    
    def __init__(self, data_source: Union[int, List[Tuple[int, int]], str], 
                 baseline_pred_gen: str = "5", verbose: bool = True):
        """
        Args:
            data_source: 
                - int: video_id pour toute la vidéo
                - [(video_id, frame_id), ...]: batch custom
                - 'all': toutes les vidéos disponibles
            baseline_pred_gen: Génération de prédictions baseline (default: 5)
            verbose: Affichage des messages
        """
        self.data_source = data_source
        self.baseline_pred_gen = baseline_pred_gen
        self.verbose = verbose
        
        # Données chargées (sera un FrameBatch après load_data)
        self.data_batch: Optional[FrameBatch] = None
        
    def load_data(self):
        """
        Charge toutes les données via le DataLoader.
        """
        loader = DataLoader(self.baseline_pred_gen, self.verbose)
        self.data_batch = loader.load_frame_batch(self.data_source)
    
    @property
    def flows_data(self) -> mx.array:
        """Accès aux flows (compatibilité avec l'ancienne interface)"""
        if self.data_batch is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        return self.data_batch.flows
    
    @property 
    def labels_data(self) -> mx.array:
        """Accès aux labels (compatibilité avec l'ancienne interface)"""
        if self.data_batch is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        return self.data_batch.labels
    
    @property
    def baseline_predictions(self) -> mx.array:
        """Accès aux prédictions baseline (compatibilité avec l'ancienne interface)"""
        if self.data_batch is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        return self.data_batch.baseline_predictions
    
    @property
    def baseline_distances(self) -> mx.array:
        """Accès aux distances baseline (compatibilité avec l'ancienne interface)"""
        if self.data_batch is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        return self.data_batch.baseline_distances
    
    @property
    def mean_points_data(self) -> Optional[mx.array]:
        """Accès aux points moyens (compatibilité avec l'ancienne interface)"""
        if self.data_batch is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        return self.data_batch.mean_points
    
    @property
    def frame_list(self) -> List[Tuple[int, int]]:
        """Accès à la liste des frames (compatibilité avec l'ancienne interface)"""
        if self.data_batch is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        return self.data_batch.frame_metadata

    def predict_from_config(self, 
                           filter_config: dict,
                           use_mean_points: bool = False,
                           adam_config: dict = None) -> mx.array:
        """
        Prédit les points de fuite pour une configuration de filtre donnée.
        
        Args:
            filter_config: Configuration du filtre avec structure FlowFilter
            use_mean_points: Utiliser les points moyens pour la colinéarité (False = centre image)
            adam_config: Configuration Adam (utilise les paramètres par défaut si None)
            
        Returns:
            mx.array: Prédictions optimisées de forme (batch_size, 2)
        """
        if self.flows_data is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        
        # 1. Préparer les points de référence pour la colinéarité
        reference_points = None
        if use_mean_points and self.mean_points_data is not None:
            reference_points = self.mean_points_data
        
        # 2. Créer le filtre et calculer les poids
        flow_filter = FlowFilterBatch(filter_config)
        weights = flow_filter.compute_weights(self.flows_data, reference_points=reference_points)
        mx.eval(weights)
        
        # 3. Configuration et lancement de l'optimiseur Adam
        adam_optimizer = AdamOptimizer()
        if adam_config:
            adam_optimizer = AdamOptimizer(**adam_config)
        
        # Points de départ au centre de l'image
        batch_size = self.flows_data.shape[0]
        h, w = self.flows_data.shape[1], self.flows_data.shape[2]
        center_points = mx.tile(mx.array([w // 2, h // 2], dtype=mx.float32), (batch_size, 1))
        mx.eval(center_points)
        
        # 4. Optimisation batch
        predictions = adam_optimizer.optimize_batch(
            self.flows_data,
            starting_points=center_points,
            weights_batch=weights
        )
        mx.eval(predictions)
        
        return predictions

    def evaluate_filter_config(self, 
                              filter_config: dict,
                              use_mean_points: bool = False,
                              adam_config: dict = None,
                              verbose: bool = None) -> float:
        """
        Évalue une configuration de filtre et retourne la distance moyenne.
        
        Args:
            filter_config: Configuration du filtre avec structure FlowFilter
            use_mean_points: Utiliser les points moyens pour la colinéarité (False = centre image)
            adam_config: Configuration Adam (utilise les paramètres par défaut si None)
            verbose: Afficher les détails (None = utilise self.verbose)
            
        Returns:
            float: Distance moyenne entre prédictions et labels
        """
        # Utiliser predict_from_config pour obtenir les prédictions
        predictions = self.predict_from_config(filter_config, use_mean_points, adam_config)
        
        # Calculer les distances
        distances = self.compute_distances(predictions)
        mean_distance = float(mx.mean(distances))
        
        # Affichage optionnel des détails
        show_verbose = verbose if verbose is not None else self.verbose
        if show_verbose:
            print(f"🎯 Config: norm={'✓' if filter_config.get('norm', {}).get('is_used', False) else '✗'}, "
                  f"colin={'✓' if filter_config.get('colinearity', {}).get('is_used', False) else '✗'}, "
                  f"heatmap={'✓' if filter_config.get('heatmap', {}).get('is_used', False) else '✗'}")
            print(f"   Distance moyenne: {mean_distance:.2f}")
        
        return mean_distance

    def compute_distances(self, predictions: mx.array) -> mx.array:
        """
        Calcule les distances euclidiennes entre prédictions et labels.
        
        Args:
            predictions: Prédictions de forme (batch_size, 2)
            
        Returns:
            mx.array: Distances de forme (batch_size,)
        """
        if self.labels_data is None:
            raise ValueError("Labels non chargés. Appelez load_data() d'abord.")
        
        distances = mx.sqrt(
            mx.sum(mx.square(predictions - self.labels_data), axis=1)
        )
        mx.eval(distances)
        return distances

    def _print_config(self, config: dict, use_mean_points: bool = None, indent: str = ""):
        """Affiche une configuration de manière lisible."""
        norm = config['norm']
        colin = config['colinearity']
        heatmap = config.get('heatmap', {})
        
        config_str = f"{indent}norm(k={norm['k']:.2f}, x0={norm['x0']:.2f}), "
        config_str += f"colin(k={colin['k']:.2f}, x0={colin['x0']:.4f})"
        
        if heatmap.get('is_used', False):
            config_str += f", heatmap(w={heatmap.get('weight', 0.0):.2f})"
        
        if use_mean_points is not None:
            config_str += f", mean_pts={'✓' if use_mean_points else '✗'}"
        
        print(config_str)
    
    def _print_config_compact(self, config: dict, use_mean_points: bool = None):
        """Affiche une configuration de manière compacte."""
        norm = config['norm']
        colin = config['colinearity']
        heatmap = config.get('heatmap', {})
        
        config_str = f"   Config: N(k={norm['k']:.2f},x0={norm['x0']:.2f}) "
        config_str += f"C(k={colin['k']:.2f},x0={colin['x0']:.4f})"
        
        if heatmap.get('is_used', False):
            config_str += f" H(w={heatmap.get('weight', 0.0):.2f})"
        
        if use_mean_points is not None:
            config_str += f" MP={'✓' if use_mean_points else '✗'}"
        
        print(config_str)
    
    def _print_final_results(self, results: dict):
        """Affiche les résultats finaux de la recherche."""
        print(f"\n🏆 RÉSULTATS FINAUX")
        print("=" * 40)
        
        # Gérer les deux types de résultats (random_search et coordinate_search)
        n_tests = results.get('n_samples_tested', results.get('n_evaluations', 0))
        print(f"Évaluations: {n_tests}")
        print(f"Baseline: {results['baseline_score']:.2f}")
        print(f"Meilleur score: {results['best_score']:.2f}")
        print(f"Amélioration: {results['best_improvement']:+.2f}")
        
        if results['best_config']:
            print(f"\n🎯 Meilleure configuration:")
            best_use_mean_points = results.get('best_use_mean_points')
            self._print_config(results['best_config'], best_use_mean_points, "  ")
            
        # Top 5
        sorted_results = sorted(results['all_results'], key=lambda x: x['mean_distance'])
        print(f"\n📈 Top 5:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['mean_distance']:.2f} "
                  f"({result['improvement']:+.2f}) - ", end="")
            use_mean_points = result.get('use_mean_points')
            self._print_config(result['config'], use_mean_points, "") 

if __name__ == "__main__":
    from src.utilities.worst_errors import select_frames_from_all_deciles

    frames_by_decile = select_frames_from_all_deciles(
        run_name="5_4", 
        n_frames_per_decile=10, 
        # video_id=4
    )

    fb = FilterConfigEvaluator(frames_by_decile, verbose=False)
    fb.load_data()
    # filter_config = {
    #     'norm': {'is_used': True, 'k': 150, 'x0': 13},
    #     'colinearity': {'is_used': True, 'k': 150, 'x0': 0.96},
    #     'heatmap': {
    #         'is_used': False, 
    #         'path': get_intermediate_dir() / 'heatmaps/unfiltered/global/global_heatmap.npy',
    #         'weight': 1.0
    #     }
    # }
    filter_config = {
        'norm': {'is_used': True, 'k': -10.0, 'x0': 1.0},
        'colinearity': {'is_used': True, 'k': 118.25, 'x0': 1.1053},
        'heatmap': {
            'is_used': True,    
            'path': get_intermediate_dir() / 'heatmaps/unfiltered/global/global_heatmap.npy',
            'weight': 0.79
        }
    }
    # Test de la nouvelle interface
    mean_distance = fb.evaluate_filter_config(filter_config=filter_config, use_mean_points=False)
    print(f"Distance moyenne: {mean_distance}")
    
    # Test avec les prédictions
    # predictions = fb.predict_from_config(filter_config=filter_config, use_mean_points=True)
    # print(f"Prédictions moyennes: {mx.mean(predictions)}")

    