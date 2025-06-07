# 🛠️ Utilities

Outils de support pour le projet calib_challenge.

## 📁 Modules disponibles

### `paths.py` - Gestionnaire de chemins
**Fonctions centralisées pour tous les chemins du projet.**

```python
from src.utilities.paths import get_flows_dir, ensure_dir_exists

# Exemples d'utilisation
flows_dir = get_flows_dir()  # data/intermediate/flows/
pred_dir = get_pred_dir(5)   # data/outputs/pred/5/
ensure_dir_exists(flows_dir) # Crée le dossier si nécessaire
```

**Fonctions principales :**
- `get_project_root()` : Racine du projet
- `get_data_dir()`, `get_inputs_dir()`, `get_outputs_dir()` : Dossiers principaux
- `get_flows_dir()`, `get_masks_dir()` : Données intermédiaires
- `get_pred_dir(video_num)`, `get_means_dir(video_num)` : Sorties par vidéo
- `ensure_dir_exists(path)` : Création automatique dossiers

---

### `generate_flows.py` - Génération flux optiques
**Génère les flux optiques à partir des vidéos sources.**

```python
from src.utilities.generate_flows import main, VideoFlowProcessor

# Usage simple
main([1, 2, 3, 4])  # Traite les vidéos labellisées

# Usage avancé
processor = VideoFlowProcessor()
success = processor.process_video(video_index=1)
```

**Fonctionnalités :**
- ✅ **Masquage intelligent** : combine masques manuels + détection véhicules automatique
- ✅ **Flux optique optimisé** : paramètres Farneback ajustés pour la conduite
- ✅ **Traitement robuste** : comptage manuel frames, gestion erreurs
- ✅ **Sauvegarde optimisée** : fichiers `.npy` compacts par vidéo

**Inputs :**
- Vidéos : `data/inputs/labeled/*.hevc`  
- Masques : `data/inputs/masks/*_mask.png` (optionnels)

**Outputs :**
- Flux : `data/intermediate/flows/{video_index}.npy`
- Shape : `(n_frames-1, height, width, 2)` avec composantes (x,y)

---

### `eval.py` - Évaluation prédictions
**Calcule MSE et score de performance des prédictions.**

```python
from src.utilities.eval import evaluate_predictions

# Évaluation sur une vidéo
mse, score = evaluate_predictions(video_index=1)

# Évaluation batch
results = evaluate_predictions()  # Toutes les vidéos
```

---

### `extract_means.py` - Extraction statistiques
**Extrait moyennes/médianes des prédictions pour post-processing.**

```python
from src.utilities.extract_means import extract_means_for_video

# Par vidéo
extract_means_for_video(video_index=1)

# Toutes les vidéos
main()  # Traite toutes les vidéos labellisées
```

---

### `ground_truth.py` - Données de référence
**Gère le chargement et traitement des labels de référence.**

## 🚀 Usage typique

```python
# 1. Génération flux optiques (étape longue ~1h)
from src.utilities.generate_flows import main as generate_flows
generate_flows([1, 2, 3, 4])

# 2. Les flux sont maintenant disponibles pour traitement
from src.utilities.paths import get_flows_dir
import numpy as np

flows = np.load(get_flows_dir() / "1.npy")
print(f"Flux vidéo 1: {flows.shape}")
```

## 📋 Standards

- **Imports** : TOUJOURS absolus `from src.utilities.module import function`
- **Chemins** : TOUJOURS via fonctions de `paths.py`
- **Type hints** : Obligatoires pour fonctions publiques
- **Documentation** : Docstrings détaillées 