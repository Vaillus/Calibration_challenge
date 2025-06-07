# 🛠️ Utilities

Outils de support pour le projet calib_challenge.

## 📁 Modules

### `paths.py`
Gestion centralisée des chemins du projet.

### `ground_truth.py`
Lecture et conversion des données de référence (angles et pixels).

### `pixel_angle_converter.py`
Conversion entre repères Car et Camera (angles ↔ pixels).

### `generate_flows.py`
Génération des flux optiques avec masquage intelligent.

### `load_flows.py`
Chargement des flux optiques (formats .npy et .npz compressé).

### `extract_means.py`
Conversion des prédictions d'angles en pixels et calcul des moyennes.

### `eval.py`
Évaluation des prédictions et calcul des scores.

### `fix_predictions.py`
Duplication de la première prédiction pour la frame 0 manquante.

### `create_mixed_batch.py`
Création d'un dataset de test équilibré par déciles d'erreur.

### `convert_to_float16.py`
Compression des flux optiques en float16 pour optimiser l'espace disque.
