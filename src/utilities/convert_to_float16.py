#!/usr/bin/env python3
"""
Conversion des fichiers de flow vectors pour optimiser l'espace disque.

PROBLÈME :
Les fichiers de flow vectors originaux (.npy en float32) sont énormes :
- Taille : ~9 GB par fichier de vidéo
- Format : Non compressé
- Précision : float32 (souvent excessive pour les flow vectors)

SOLUTION :
Ce script convertit les fichiers pour réduire drastiquement leur taille :
- Conversion : float32 → float16 (précision suffisante pour les flows)
- Compression : .npy → .npz (compression automatique)
- Réduction : Typiquement 70-80% de réduction de taille

USAGE :
    python convert_to_float16.py

Le script traite automatiquement tous les fichiers .npy dans le dossier flows/
et génère les versions optimisées avec le suffixe '_float16.npz'.

RÉSULTAT :
- Fichier original : video_0.npy (~9 GB)
- Fichier converti : video_0_float16.npz (~2-3 GB)

Author: Hugo Vaillaud
Date: 2024-05-22
"""

import numpy as np
import os
import glob

def convert_file_to_float16_npz(input_file, chunk_size=100):
    """
    Convertit un fichier .npy en float16 et sauvegarde en .npz compressé.
    """
    print(f"\nTraitement de {input_file}...")
    
    # Créer le nom du fichier de sortie
    output_file = input_file.replace('.npy', '_float16.npz')
    
    # Charger les métadonnées du fichier
    data = np.load(input_file, mmap_mode='r')
    shape = data.shape
    dtype = data.dtype
    
    print(f"Shape: {shape}")
    print(f"Type original: {dtype}")
    print(f"Taille originale: {os.path.getsize(input_file) / 1e9:.2f} GB")
    
    # Traiter par morceaux et empiler en mémoire (float16)
    total_frames = shape[0]
    float16_chunks = []
    for start_idx in range(0, total_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, total_frames)
        print(f"\rTraitement des frames {start_idx} à {end_idx-1}...", end="", flush=True)
        chunk = data[start_idx:end_idx]
        chunk_float16 = chunk.astype(np.float16)
        float16_chunks.append(chunk_float16)
    print("\nEmpilement des morceaux...")
    float16_array = np.concatenate(float16_chunks, axis=0)
    
    # Sauvegarder en .npz compressé
    np.savez_compressed(output_file, flow=float16_array)
    print(f"Conversion terminée. Fichier compressé sauvegardé dans: {output_file}")
    print(f"Taille finale: {os.path.getsize(output_file) / 1e9:.2f} GB")

def main():
    # Trouver tous les fichiers .npy dans le dossier flows/
    flow_files = glob.glob('calib_challenge/flows/*.npy')
    
    # Exclure les fichiers qui se terminent déjà par _float16.npz
    flow_files = [f for f in flow_files if not f.endswith('_float16.npz')]
    
    if not flow_files:
        print("Aucun fichier .npy trouvé dans le dossier flows/")
        return
    
    print(f"Fichiers à convertir: {len(flow_files)}")
    for file in flow_files:
        convert_file_to_float16_npz(file)

if __name__ == "__main__":
    main() 