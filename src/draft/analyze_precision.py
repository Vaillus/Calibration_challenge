import numpy as np
import matplotlib.pyplot as plt

def simulate_float8(x, scale_factor=100):
    """
    Simule une précision float8 en réduisant la précision des float32.
    Le scale_factor détermine combien de décimales on garde.
    """
    return np.round(x * scale_factor) / scale_factor

def analyze_precision_loss(flow_file, n_frames=20):
    # Charger seulement les n_frames premières frames
    flow_original = np.load(flow_file, mmap_mode='r')[:n_frames]
    print(f"Shape analysée: {flow_original.shape}")
    print(f"Type originale: {flow_original.dtype}")
    print(f"Taille analysée: {flow_original.nbytes / 1e6:.2f} MB")
    
    # Convertir en float16
    flow_float16 = flow_original.astype(np.float16)
    print(f"Taille en float16: {flow_float16.nbytes / 1e6:.2f} MB")
    
    # Simuler float8
    flow_float8 = simulate_float8(flow_original)
    print(f"Taille simulée en float8: {flow_float8.nbytes / 1e6:.2f} MB")
    
    # Reconvertir en float32 pour la comparaison
    flow_reconverted_16 = flow_float16.astype(np.float32)
    flow_reconverted_8 = flow_float8.astype(np.float32)
    
    # Calculer les différences absolues (valeurs)
    diff_16 = np.abs(flow_original - flow_reconverted_16)
    diff_8 = np.abs(flow_original - flow_reconverted_8)
    
    print(f"\nStatistiques des différences (valeurs) - float16:")
    print(f"Différence maximale: {np.max(diff_16):.6f}")
    print(f"Différence moyenne: {np.mean(diff_16):.6f}")
    print(f"Écart-type des différences: {np.std(diff_16):.6f}")
    
    print(f"\nStatistiques des différences (valeurs) - float8:")
    print(f"Différence maximale: {np.max(diff_8):.6f}")
    print(f"Différence moyenne: {np.mean(diff_8):.6f}")
    print(f"Écart-type des différences: {np.std(diff_8):.6f}")
    
    # Calcul des différences angulaires
    v1 = flow_original.reshape(-1, 2)
    v2_16 = flow_reconverted_16.reshape(-1, 2)
    v2_8 = flow_reconverted_8.reshape(-1, 2)
    
    # Calcul des normes
    norm1 = np.linalg.norm(v1, axis=1)
    norm2_16 = np.linalg.norm(v2_16, axis=1)
    norm2_8 = np.linalg.norm(v2_8, axis=1)
    
    # Masque pour ignorer les vecteurs nuls (ou très petits)
    valid = (norm1 > 1e-6) & (norm2_16 > 1e-6) & (norm2_8 > 1e-6)
    v1_valid = v1[valid]
    v2_16_valid = v2_16[valid]
    v2_8_valid = v2_8[valid]
    norm1_valid = norm1[valid]
    
    # Calcul des déciles sur les normes
    deciles = np.percentile(norm1_valid, np.arange(0, 101, 10))
    print("\nDéciles des normes des vecteurs originaux (float32):")
    for i, d in enumerate(deciles):
        print(f"Décile {i*10}%: {d:.6f}")
    
    # Calcul des différences angulaires par décile pour float16 et float8
    angles_by_decile_16 = []
    angles_by_decile_8 = []
    
    for i in range(10):
        mask = (norm1_valid >= deciles[i]) & (norm1_valid < deciles[i+1])
        if np.any(mask):
            v1_decile = v1_valid[mask]
            v2_16_decile = v2_16_valid[mask]
            v2_8_decile = v2_8_valid[mask]
            
            # Calcul pour float16
            dot_16 = np.sum(v1_decile * v2_16_decile, axis=1)
            norm_prod_16 = np.linalg.norm(v1_decile, axis=1) * np.linalg.norm(v2_16_decile, axis=1)
            cos_theta_16 = np.clip(dot_16 / norm_prod_16, -1.0, 1.0)
            angles_rad_16 = np.arccos(cos_theta_16)
            angles_deg_16 = np.degrees(angles_rad_16)
            
            # Calcul pour float8
            dot_8 = np.sum(v1_decile * v2_8_decile, axis=1)
            norm_prod_8 = np.linalg.norm(v1_decile, axis=1) * np.linalg.norm(v2_8_decile, axis=1)
            cos_theta_8 = np.clip(dot_8 / norm_prod_8, -1.0, 1.0)
            angles_rad_8 = np.arccos(cos_theta_8)
            angles_deg_8 = np.degrees(angles_rad_8)
            
            angles_by_decile_16.append(angles_deg_16)
            angles_by_decile_8.append(angles_deg_8)
    
    # Afficher les statistiques par décile
    print("\nStatistiques des différences angulaires par décile (en degrés):")
    for i, (angles_16, angles_8) in enumerate(zip(angles_by_decile_16, angles_by_decile_8)):
        print(f"\nDécile {i*10}-{(i+1)*10}%:")
        print(f"  Nombre de vecteurs: {len(angles_16)}")
        print(f"  Float16 - Différence angulaire moyenne: {np.mean(angles_16):.6f}")
        print(f"  Float16 - Différence angulaire max: {np.max(angles_16):.6f}")
        print(f"  Float16 - Écart-type: {np.std(angles_16):.6f}")
        print(f"  Float8  - Différence angulaire moyenne: {np.mean(angles_8):.6f}")
        print(f"  Float8  - Différence angulaire max: {np.max(angles_8):.6f}")
        print(f"  Float8  - Écart-type: {np.std(angles_8):.6f}")
    
    # Sauvegarder un exemple de flow en float16 et float8
    output_file_16 = flow_file.replace('.npy', '_float16_sample.npy')
    output_file_8 = flow_file.replace('.npy', '_float8_sample.npy')
    np.save(output_file_16, flow_float16)
    np.save(output_file_8, flow_float8)
    print(f"\nFlows sauvegardés dans: {output_file_16} et {output_file_8}")

if __name__ == "__main__":
    analyze_precision_loss('calib_challenge/flows/0.npy', n_frames=20) 