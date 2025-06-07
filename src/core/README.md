# 🏗️ Core - Modules Fondamentaux

Ce dossier contient les briques fondamentales du système d'estimation de points de fuite.

## 📁 Modules (dans l'ordre d'exécution)

### 🚀 **`interactive_viewer.py`** 
Visualiseur interactif temps réel pour débugger et analyser le système. Lance une interface OpenCV qui permet de :
- Naviguer frame par frame dans les vidéos
- Comparer prédictions vs ground truth vs estimations
- Tester différentes visualisations de flux optique
- Faire de la segmentation manuelle

### 🎥 **`flow.py`** 
Génération et manipulation des flux optiques (calcul dense Farnebäck, points de séparation)

### 🎭 **`segmentation.py`** 
Détection et masquage d'objets (véhicules, capot) avec YOLO pour filtrer les flux parasites

### 🔧 **`flow_filter.py`** 
Filtrage et pondération des flux optiques (par norme, colinéarité, distance au centre)

### 🔍 **`colinearity_optimization.py`** 
Estimateur de points de fuite (version numpy/scipy classique) avec calcul de scores de colinéarité

### 🔍 **`colinearity_optimization_parallel.py`** 
Estimateur de points de fuite (version MLX parallèle) pour traitement batch haute performance

### 🎯 **`optimizers.py`** 
Méthodes d'optimisation centralisées (Adam MLX + L-BFGS-B scipy) pour trouver les points de fuite optimaux

### 🎨 **`rendering.py`** 
Fonctions de rendu pour le visualiseur interactive_viewer.py (flèches de flux, points de séparation, masques) 