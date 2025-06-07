# 🧪 Dossier Experiments

Ce dossier contient tous les scripts d'expérimentation, de benchmark et d'analyse du projet.

## 📁 Organisation

### Benchmarks d'optimisation
- **`benchmark_adam_plateau.py`** : Test des paramètres d'early stopping pour Adam
- **`benchmark_lbfgs_vs_adam.py`** : Comparaison des deux méthodes d'optimisation (à déplacer)

### Benchmarks de filtrage
- **`filter_benchmark.py`** : Tests exhaustifs des paramètres de filtrage (à déplacer)

### Analyses de données
- **`norm_distribution_by_video.py`** : Analyse des distributions de normes par vidéo (à déplacer)
- **`colinearity_by_norm_analysis.py`** : Analyse de la relation colinéarité/norme (à déplacer)
- **`flow_norm_analysis.py`** : Analyse approfondie des normes de flux optique (à déplacer)

### Visualisations
- **`visualize_error.py`** : Visualisation des erreurs de prédiction (à déplacer)
- **`visualize_distances.py`** : Visualisation des distributions de distances (à déplacer)
- **`visualize_flow.py`** : Visualisation des champs de flux optique (à déplacer)

## 🎯 Usage

### Lancer un benchmark Adam
```python
from src.experiments.benchmark_adam_plateau import benchmark_plateau_detection

# Test rapide
results = benchmark_plateau_detection(max_frames=10, video_id=0)

# Test complet
results = benchmark_plateau_detection(max_frames=100, video_id=2)
```

### Lancer en command line
```bash
cd calib_challenge/src/experiments
python benchmark_adam_plateau.py
```

## 📋 À faire

- [ ] Déplacer tous les scripts de benchmark depuis `src/` vers `experiments/`
- [ ] Standardiser les formats de sortie des benchmarks
- [ ] Créer un script master qui lance tous les benchmarks
- [ ] Ajouter des tests de régression pour les expériences importantes 