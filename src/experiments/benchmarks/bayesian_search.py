#!/usr/bin/env python3
"""
🧠 BAYESIAN SEARCH

Recherche bayésienne intelligente pour optimiser les paramètres de filtrage sigmoïde.
Avec historique persistent pour accumulation intelligente des résultats.
"""

import mlx.core as mx
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.utilities.filter_config_evaluator import FilterConfigEvaluator
from src.utilities.worst_errors import select_frames_from_all_deciles
from src.utilities.paths import get_intermediate_dir


@dataclass
class SearchState:
    """État de la recherche bayésienne - évite les variables nonlocal"""
    results: List[Dict[str, Any]]
    best_score: float
    best_config: Optional[Dict[str, Any]]
    best_use_mean_points: bool
    last_improvement_call: int
    baseline_mean: float
    n_calls: int
    search_heatmap: bool
    search_mean_points: bool


class BayesianSearch(FilterConfigEvaluator):
    """Recherche bayésienne pour optimiser les paramètres de filtrage."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results_dir = get_intermediate_dir()
        self.results_dir.mkdir(exist_ok=True)
        self.history_file = self.results_dir / "bayesian_search_history.json"
        self.search_state: Optional[SearchState] = None

    def search(self, 
               param_ranges: dict,
               n_calls: int = 100,
               n_initial_points: int = 10,
               acq_func: str = 'EI',
               initial_configs: list = None,
               search_heatmap: bool = False,
               search_mean_points: bool = True,
               use_previous_results: bool = True,
               max_previous_points: int = 100) -> dict:
        """
        Recherche bayésienne intelligente pour optimiser les paramètres.
        Plus efficace que coordinate/random search pour espaces multi-dimensionnels.
        
        Args:
            param_ranges: Dictionnaire des ranges pour chaque paramètre
            n_calls: Nombre total d'évaluations à effectuer (default: 100)
            n_initial_points: Nombre de points d'exploration aléatoire initiaux (default: 10)
            acq_func: Fonction d'acquisition ('EI', 'LCB', 'PI') (default: 'EI')
            initial_configs: Liste de configurations de départ intelligentes (default: None)
            search_heatmap: Inclure les paramètres heatmap dans la recherche (default: False)
            search_mean_points: Inclure le paramètre points moyens dans la recherche (default: True)
            use_previous_results: Utiliser l'historique comme prior (default: True)
            max_previous_points: Nombre max de points précédents à charger (default: 100)
            
        Returns:
            dict: Résultats avec meilleure configuration et historique
        """
        if self.flows_data is None:
            raise ValueError("Données non chargées. Appelez load_data() d'abord.")
        
        # 1. Setup et initialisation
        previous_x, previous_y = self._load_previous_results_and_init_display(
            n_calls, 
            acq_func, 
            use_previous_results, 
            max_previous_points
        )
        
        # 2. Initialiser l'état de recherche
        baseline_mean = float(mx.mean(self.baseline_distances))
        self.search_state = SearchState(
            results=[],
            best_score=float('inf'),
            best_config=None,
            best_use_mean_points=False,
            last_improvement_call=0,
            baseline_mean=baseline_mean,
            n_calls=n_calls,
            search_heatmap=search_heatmap,
            search_mean_points=search_mean_points
        )
        
        # 3. Construire l'espace de recherche
        dimensions = self._build_search_space(
            param_ranges, search_heatmap, search_mean_points)
        
        # 4. Préparer les points initiaux
        x0, y0 = self._prepare_initial_points(
            previous_x, previous_y, dimensions, 
            initial_configs, search_heatmap, search_mean_points
        )
        
        # 5. Lancer l'optimisation
        skopt_result = self._run_optimization(
            dimensions, x0, y0, 
            n_calls, n_initial_points, acq_func
        )
        
        # 6. Formater et retourner les résultats
        return self._format_results(skopt_result, x0)

    def _load_previous_results_and_init_display(
            self, 
            n_calls: int, 
            acq_func: str, 
            use_previous_results: bool, 
            max_previous_points: int
        ):
        """
        Charge les résultats précédents pour initialiser le prior bayésien et affiche l'interface de démarrage.
        
        Returns:
            Tuple[List, List]: (previous_x, previous_y) - Points précédents pour le prior bayésien
                - previous_x: Liste des configurations de paramètres précédentes
                - previous_y: Liste des scores correspondants
                - Retourne (None, None) si aucun historique disponible
        """
        print(f"🧠 BAYESIAN SEARCH - {n_calls} évaluations intelligentes")
        print(f"📊 Baseline moyen: {float(mx.mean(self.baseline_distances)):.2f}")
        print(f"🎯 Fonction d'acquisition: {acq_func}")
        
        # Charger les résultats précédents pour le prior bayésien
        previous_x, previous_y = None, None
        if use_previous_results:
            print(f"🔄 Chargement de l'historique...")
            previous_x, previous_y = self.load_previous_results(max_points=max_previous_points)
        
        print("⚠️  Appuyez Ctrl+C pour arrêter et voir les résultats")
        print("=" * 60)
        
        # Lignes d'affichage live
        print("📊 Évaluation: 0/0 (0.0%)")
        print("🎯 Meilleur score: -- (amélioration: --)")
        print("🔄 Dernière tentative: --")
        print("⭐ Dernière amélioration: éval #--")
        print("   Config actuelle: --")
        
        return previous_x, previous_y
    
    def _build_search_space(self, param_ranges: dict, search_heatmap: bool, search_mean_points: bool):
        """Construction de l'espace de recherche pour scikit-optimize"""
        try:
            from skopt.space import Real, Categorical
        except ImportError:
            raise ImportError("scikit-optimize requis: pip install scikit-optimize")
        
        # Définir l'espace de recherche
        param_names = ['norm_k', 'norm_x0', 'colinearity_k', 'colinearity_x0']
        if search_heatmap:
            param_names.append('heatmap_weight')
        if search_mean_points:
            param_names.append('use_mean_points')
        
        dimensions = []
        
        for param_name in param_names:
            if param_name == 'use_mean_points':
                # Paramètre booléen spécial - toujours discret
                if param_name in param_ranges:
                    # Si présent dans param_ranges, on veut l'explorer
                    dimensions.append(Categorical([False, True], name=param_name))
                elif search_mean_points:
                    # Par défaut si search_mean_points=True
                    dimensions.append(Categorical([False, True], name=param_name))
            elif param_name in param_ranges:
                # Paramètres continus normaux
                min_val, max_val = param_ranges[param_name]
                dimensions.append(Real(min_val, max_val, name=param_name))
        
        return dimensions

    def _prepare_initial_points(self, previous_x, previous_y, dimensions, initial_configs, search_heatmap, search_mean_points):
        """Préparation des points de départ intelligents"""
        x0 = []
        y0 = []
        
        # Ajouter les résultats précédents en premier
        if previous_x and previous_y:
            # Filtrer les points qui correspondent à notre espace de recherche actuel
            for px, py in zip(previous_x, previous_y):
                if len(px) == len(dimensions):  # Même nombre de paramètres
                    # Vérifier que les valeurs sont dans les ranges
                    valid = True
                    for i, (val, dim) in enumerate(zip(px, dimensions)):
                        if hasattr(dim, 'bounds'):  # Real parameter
                            if not (dim.bounds[0] <= val <= dim.bounds[1]):
                                valid = False
                                break
                        elif hasattr(dim, 'categories'):  # Categorical parameter  
                            if val not in dim.categories:
                                valid = False
                                break
                    
                    if valid:
                        x0.append(px)
                        y0.append(py)
            
            print(f"🎯 {len(x0)} points précédents réutilisés comme prior")
        
        # Ajouter les configurations de départ manuelles
        if initial_configs:
            print(f"🎯 Ajout de {len(initial_configs)} configurations de départ intelligentes")
            
            for config_item in initial_configs:
                # config_item peut être juste une config ou un tuple (config, use_mean_points)
                if isinstance(config_item, tuple):
                    config, use_mean_points_config = config_item
                else:
                    config = config_item
                    use_mean_points_config = False
                
                # Convertir config → params avec la nouvelle fonction
                params = self.config_to_params(config, use_mean_points_config, search_heatmap, search_mean_points)
                
                x0.append(params)
                
                # Évaluer pour avoir y0
                score = self._evaluate_params(params)
                y0.append(score)
        
        return x0, y0

    def _evaluate_params(self, params) -> float:
        """
        Fonction objective extraite - évalue un set de paramètres.
        
        Args:
            params: Paramètres à évaluer (format skopt)
            
        Returns:
            float: Score à minimiser (distance moyenne)
        """
        if self.search_state is None:
            raise ValueError("SearchState non initialisé")
        
        state = self.search_state
        
        # Convertir params → config avec les fonctions existantes
        config, use_mean_points = self.params_to_config(
            params, state.search_heatmap, state.search_mean_points
        )
        
        # Évaluer (silencieux)
        original_verbose = self.verbose
        self.verbose = False
        mean_distance = self.evaluate_filter_config(config, use_mean_points=use_mean_points)
        self.verbose = original_verbose
        
        # Sauvegarder dans l'historique
        self.save_to_history(params, mean_distance)
        
        # Stocker résultat
        result = {
            'config': config.copy(),
            'use_mean_points': use_mean_points,
            'mean_distance': mean_distance,
            'improvement': state.baseline_mean - mean_distance,
            'call_id': len(state.results) + 1,
            'params': list(params)  # Conversion pour JSON
        }
        state.results.append(result)
        
        # Mettre à jour meilleur
        if mean_distance < state.best_score:
            state.best_score = mean_distance
            state.best_config = config.copy()
            state.best_use_mean_points = use_mean_points
            state.last_improvement_call = len(state.results)
        
        # Affichage live
        self._update_live_display(
            len(state.results), state.n_calls, state.best_score, state.baseline_mean,
            state.last_improvement_call, state.best_config, state.best_use_mean_points, 
            last_score=mean_distance
        )
        
        # Retourner score à minimiser (fonction objective pour gp_minimize)
        return mean_distance

    def _run_optimization(self, dimensions, x0, y0, n_calls, n_initial_points, acq_func):
        """Lance l'optimisation bayésienne avec gestion d'interruption"""
        try:
            from skopt import gp_minimize
        except ImportError:
            raise ImportError("scikit-optimize requis: pip install scikit-optimize")
        
        # Ajuster n_initial_points en fonction du nombre de points déjà disponibles
        effective_initial_points = max(n_initial_points - len(x0), 2) if x0 else n_initial_points
        
        try:
            # Lancer l'optimisation bayésienne avec tous les points précédents
            result = gp_minimize(
                func=self._evaluate_params,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=effective_initial_points,
                acq_func=acq_func,
                random_state=42,
                x0=x0 if x0 else None,
                y0=y0 if y0 else None
            )
            return result
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️  Recherche interrompue après {len(self.search_state.results)} évaluations")
            return None
        
    def _format_results(self, skopt_result, x0):
        """Formate les résultats finaux de la recherche"""
        state = self.search_state
        
        search_results = {
            'best_config': state.best_config,
            'best_use_mean_points': state.best_use_mean_points,
            'best_score': state.best_score,
            'best_improvement': state.baseline_mean - state.best_score,
            'baseline_score': state.baseline_mean,
            'n_evaluations': len(state.results),
            'all_results': state.results,
            'skopt_result': skopt_result,
            'previous_points_used': len(x0) if x0 else 0
        }
        
        self._print_final_results(search_results)
        return search_results
    
    def params_to_config(self, params, search_heatmap=False, search_mean_points=True):
        """Convertit params plat → config structuré + use_mean_points"""
        config = {
            'norm': {'is_used': True, 'k': params[0], 'x0': params[1]},
            'colinearity': {'is_used': True, 'k': params[2], 'x0': params[3]},
            'heatmap': {'is_used': False}
        }
        
        idx = 4
        use_mean_points = False
        
        if search_heatmap and len(params) > idx:
            config['heatmap']['is_used'] = True
            config['heatmap']['weight'] = params[idx]
            config['heatmap']['path'] = get_intermediate_dir() / 'heatmaps/unfiltered/global/global_heatmap.npy'
            idx += 1
        
        if search_mean_points and len(params) > idx:
            use_mean_points = params[idx]
        
        return config, use_mean_points
    
    def config_to_params(self, config, use_mean_points=False, search_heatmap=False, search_mean_points=True):
        """Convertit config structuré → params plat"""
        params = [
            config['norm']['k'],
            config['norm']['x0'],
            config['colinearity']['k'], 
            config['colinearity']['x0']
        ]
        
        if search_heatmap:
            params.append(config.get('heatmap', {}).get('weight', 0.0))
        
        if search_mean_points:
            params.append(use_mean_points)
        
        return params
    
    def save_to_history(self, params, score):
        """Ajoute un point au fichier d'historique"""
        # Charger l'historique existant
        history = {"evaluations": []}
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        
        # Convertir les types numpy en types Python natifs pour JSON
        json_params = []
        for param in params:
            if hasattr(param, 'item'):  # Types numpy (np.bool_, np.float64, etc.)
                json_params.append(param.item())
            else:
                json_params.append(param)
        
        # Ajouter le nouveau point
        history["evaluations"].append({
            "params": json_params,
            "score": float(score),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Sauvegarder
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_previous_results(self, max_points: int = 100):
        """Charge les résultats précédents pour les utiliser comme prior."""
        if not self.history_file.exists():
            print("📂 Aucun historique trouvé")
            return None, None
        
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            evaluations = history.get('evaluations', [])
            if not evaluations:
                print("📂 Historique vide")
                return None, None
            
            # Prendre les max_points plus récents
            recent_evaluations = evaluations[-max_points:] if len(evaluations) > max_points else evaluations
            
            all_x = [eval_data['params'] for eval_data in recent_evaluations]
            all_y = [eval_data['score'] for eval_data in recent_evaluations]
            
            print(f"📥 Chargé: {len(all_x)} points depuis l'historique")
            return all_x, all_y
            
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement de l'historique: {e}")
            return None, None


    
    def _update_live_display(self, current_call, total_calls, best_score, baseline_mean,
                           last_improvement_call, best_config, best_use_mean_points=None, last_score=None):
        """Met à jour l'affichage pour Bayesian search."""
        import sys
        
        # Effacer les lignes précédentes
        for _ in range(5):
            sys.stdout.write('\033[1A\033[2K\r')
        
        # Calculer améliorations
        best_improvement = baseline_mean - best_score
        last_improvement = baseline_mean - last_score if last_score is not None else 0
        progress = current_call / total_calls * 100
        
        # Affichage compact
        print(f"📊 Évaluation: {current_call:3d}/{total_calls} ({progress:5.1f}%)")
        print(f"🎯 Meilleur score: {best_score:.2f} (amélioration: {best_improvement:+.2f})")
        if last_score is not None:
            print(f"🔄 Dernière tentative: {last_score:.2f} (amélioration: {last_improvement:+.2f})")
        else:
            print(f"🔄 Dernière tentative: --")
        print(f"⭐ Dernière amélioration: éval #{last_improvement_call}")
        
        if best_config:
            self._print_config_compact(best_config, best_use_mean_points)
        else:
            print("   Config: --")
        
        sys.stdout.flush() 

if __name__ == "__main__":
    # Configuration de la recherche bayésienne avec points de départ intelligents
    print("🧠 LANCEMENT RECHERCHE BAYÉSIENNE AVANCÉE")
    print("=" * 50)
    
    # Sélectionner frames aléatoires par décile
    frames_by_decile = select_frames_from_all_deciles(run_name="8", n_frames_per_decile=10, seed=43)


    # Initialisation
    searcher = BayesianSearch(data_source=frames_by_decile, baseline_pred_gen="8", means_gen="8", verbose=True)
    searcher.load_data()
    
    # Définition des ranges de recherche
    param_ranges = {
        'norm_k': (-10.0, 150.0),  # Élargi pour inclure 0.21
        'norm_x0': (-10.0, 151.0),  # Élargi pour inclure 39.60
        'colinearity_k': (50, 151.0),
        'colinearity_x0': (0.9, 2.0),  # Élargi pour inclure 1.10
        'heatmap_weight': (0.0, 1.0),
        'use_mean_points': True  # Explore True ET False
    }
    
    # Points de départ intelligents basés sur les résultats précédents
    smart_configs = [
        ({
            'norm': {'is_used': True, 'k': 26.77, 'x0': 11.87},
            'colinearity': {'is_used': True, 'k': 50.00, 'x0': 1.7631},
            'heatmap': {'is_used': False}
        }, True),
        ({
            'norm': {'is_used': True, 'k': -10.0, 'x0': 1.0},
            'colinearity': {'is_used': True, 'k': 118.25, 'x0': 1.1053},
            'heatmap': {
                'is_used': True,    
                'path': get_intermediate_dir() / 'heatmaps/unfiltered/global/global_heatmap.npy',
                'weight': 0.79
            }
        }, True),
        ({
             'norm': {'is_used': True, 'k': 150.0, 'x0': 13},
             'colinearity': {'is_used': True, 'k': 150.0, 'x0': 0.96},
             'heatmap': {'is_used': False}
         }, True),  # use_mean_points=False
        # Configuration "soft norm + hard colinearity" (meilleure de la recherche précédente)
        ({
            'norm': {'is_used': True, 'k': 0.21, 'x0': 39.60},  # Vraies valeurs trouvées
            'colinearity': {'is_used': True, 'k': 95.56, 'x0': 0.96},
            'heatmap': {'is_used': False}
        }, False),  
        ({
            'norm': {'is_used': True, 'k': 150.0, 'x0': 10.37},
            'colinearity': {'is_used': True, 'k': 150.0, 'x0': 1.1566},
            'heatmap': {'is_used': False}
        }, False),
    ({
        'norm': {'is_used': True, 'k': 15.13, 'x0': 18.18},
        'colinearity': {'is_used': True, 'k': 94.77, 'x0': 1.3723},
        'heatmap': {'is_used': True, 'weight': 0.87, 'path': get_intermediate_dir() / 'heatmaps/unfiltered/global/global_heatmap.npy'}
    }, False)
    ]
    
    print(f"🎯 Utilisation de {len(smart_configs)} configurations de départ intelligentes")
    print("🔍 Exploration des paramètres heatmap ET mean points")
    print("📚 Réutilisation automatique de l'historique précédent")
    
    # Lancement de la recherche avec historique
    results = searcher.search(
        param_ranges=param_ranges,
        n_calls=1000,  # Moins d'évaluations car on réutilise l'historique
        n_initial_points=80,  # Moins de points aléatoires
        acq_func='EI',  # Expected Improvement
        initial_configs=smart_configs,
        search_heatmap=True,  # Activer recherche heatmap
        search_mean_points=True,  # Activer recherche mean points
        use_previous_results=True,  # Utiliser l'historique
        max_previous_points=100  # Charger jusqu'à 100 points précédents
    )
    
    print(f"\n🏁 Recherche terminée !")
    print(f"💾 Meilleure amélioration: {results['best_improvement']:+.2f}")
    print(f"📚 Points précédents utilisés: {results.get('previous_points_used', 0)}")
    print(f"🎯 Nouvelles évaluations: {results['n_evaluations']}")
    if results.get('best_use_mean_points') is not None:
        mean_pts_str = "✓" if results['best_use_mean_points'] else "✗"
        print(f"📍 Mean points: {mean_pts_str}")
    
    # Analyse des résultats heatmap
    heatmap_results = [r for r in results['all_results'] 
                      if r['config'].get('heatmap', {}).get('is_used', False)]
    if heatmap_results:
        best_heatmap = min(heatmap_results, key=lambda x: x['mean_distance'])
        print(f"🗺️  Meilleur avec heatmap: {best_heatmap['improvement']:+.2f} "
              f"(weight={best_heatmap['config']['heatmap']['weight']:.2f})")
    else:
        print(f"🗺️  Aucune config heatmap n'a été explorée avec succès")
        
    # Analyse mean points vs image center
    mean_pts_true = [r for r in results['all_results'] if r.get('use_mean_points') == True]
    mean_pts_false = [r for r in results['all_results'] if r.get('use_mean_points') == False]
    
    if mean_pts_true and mean_pts_false:
        best_mean_true = min(mean_pts_true, key=lambda x: x['mean_distance'])
        best_mean_false = min(mean_pts_false, key=lambda x: x['mean_distance'])
        
        print(f"📊 Comparaison mean points:")
        print(f"   Avec mean points: {best_mean_true['improvement']:+.2f}")
        print(f"   Avec image center: {best_mean_false['improvement']:+.2f}")
        
        if best_mean_false['improvement'] > best_mean_true['improvement']:
            print(f"   → Image center reste meilleur de {best_mean_false['improvement'] - best_mean_true['improvement']:+.2f}")
        else:
            print(f"   → Mean points est meilleur de {best_mean_true['improvement'] - best_mean_false['improvement']:+.2f}")