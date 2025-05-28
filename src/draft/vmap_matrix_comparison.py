import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

def benchmark_gradient_scenario():
    """
    Benchmark spécifique pour les cas où on doit calculer des gradients
    (fonction scalaire -> nécessite vmap)
    """
    print("🎯 BENCHMARK GRADIENT SCENARIO")
    print("="*60)
    
    # Test différentes tailles de matrices
    sizes = [(64, 64), (128, 128), (240, 320), (480, 640), (720, 1280)]
    
    results = {
        'sizes': [],
        'vectorized_times': [],
        'vmap_times': [],
        'speedups': [],
        'total_elements': []
    }
    
    for H, W in sizes:
        print(f"\n📐 Test avec matrice {H}x{W} ({H*W:,} éléments)")
        
        # Données de test
        flow_vectors = mx.random.normal((H, W, 2))
        vanishing_point = mx.array([W/2.0, H/2.0])
        
        # Précalcul des directions
        y_coords, x_coords = mx.meshgrid(mx.arange(H), mx.arange(W), indexing='ij')
        pixel_coords = mx.stack([x_coords, y_coords], axis=-1)
        directions_to_vp = vanishing_point - pixel_coords
        
        # ============ FONCTION SCALAIRE (comme pour gradient) ============
        def vanishing_point_score_vectorized(flow_vecs, directions, vp_candidate):
            """Score global vectorisé - retourne un scalaire"""
            # Recalcule les directions avec le nouveau point de fuite
            y_coords, x_coords = mx.meshgrid(mx.arange(flow_vecs.shape[0]), 
                                           mx.arange(flow_vecs.shape[1]), indexing='ij')
            pixel_coords = mx.stack([x_coords, y_coords], axis=-1)
            new_directions = vp_candidate - pixel_coords
            
            # Filtrage par norme
            flow_norms = mx.linalg.norm(flow_vecs, axis=-1)
            valid_mask = flow_norms > 1e-2
            
            # Colinéarité
            flow_normalized = flow_vecs / mx.maximum(flow_norms[..., None], 1e-8)
            dir_norms = mx.linalg.norm(new_directions, axis=-1)
            dir_normalized = new_directions / mx.maximum(dir_norms[..., None], 1e-8)
            
            collinearities = mx.abs(mx.sum(flow_normalized * dir_normalized, axis=-1))
            valid_collinearities = collinearities * valid_mask
            
            return mx.sum(valid_collinearities) / mx.maximum(mx.sum(valid_mask), 1.0)
        
        def vanishing_point_score_single_pixel(flow_vec, pixel_coord, vp_candidate):
            """Score pour un seul pixel - compatible vmap"""
            direction = vp_candidate - pixel_coord
            
            flow_norm = mx.linalg.norm(flow_vec)
            dir_norm = mx.linalg.norm(direction)
            
            valid = flow_norm > 1e-2
            
            flow_normalized = flow_vec / mx.maximum(flow_norm, 1e-8)
            dir_normalized = direction / mx.maximum(dir_norm, 1e-8)
            
            collinearity = mx.abs(mx.sum(flow_normalized * dir_normalized))
            return mx.where(valid, collinearity, 0.0)
        
        # ============ BENCHMARK VECTORISÉ ============
        times_vec = []
        for _ in range(10):
            start = time.time()
            score_vec = vanishing_point_score_vectorized(flow_vectors, directions_to_vp, vanishing_point)
            mx.eval(score_vec)
            times_vec.append(time.time() - start)
        
        avg_time_vec = np.mean(times_vec[2:])  # Skip warmup
        
        # ============ BENCHMARK VMAP ============
        # vmap sur tous les pixels
        pixel_coords_flat = pixel_coords.reshape(-1, 2)
        flow_vectors_flat = flow_vectors.reshape(-1, 2)
        
        vmap_score_fn = mx.vmap(vanishing_point_score_single_pixel, in_axes=(0, 0, None))
        
        times_vmap = []
        for _ in range(10):
            start = time.time()
            pixel_scores = vmap_score_fn(flow_vectors_flat, pixel_coords_flat, vanishing_point)
            # Utilise mx.where au lieu d'indexation booléenne
            valid_mask = pixel_scores > 0
            num_valid = mx.sum(valid_mask)
            score_vmap = mx.sum(pixel_scores) / mx.maximum(num_valid, 1.0)
            mx.eval(score_vmap)
            times_vmap.append(time.time() - start)
        
        avg_time_vmap = np.mean(times_vmap[2:])
        
        # ============ RÉSULTATS ============
        speedup = avg_time_vmap / avg_time_vec
        total_elements = H * W
        
        results['sizes'].append(f"{H}x{W}")
        results['vectorized_times'].append(avg_time_vec * 1000)  # en ms
        results['vmap_times'].append(avg_time_vmap * 1000)
        results['speedups'].append(speedup)
        results['total_elements'].append(total_elements)
        
        print(f"  Vectorisé: {avg_time_vec*1000:.2f}ms")
        print(f"  Vmap:      {avg_time_vmap*1000:.2f}ms")
        print(f"  Speedup:   {speedup:.1f}x")
        
        # Vérification cohérence
        score_vec_final = vanishing_point_score_vectorized(flow_vectors, directions_to_vp, vanishing_point)
        pixel_scores_final = vmap_score_fn(flow_vectors_flat, pixel_coords_flat, vanishing_point)
        valid_mask_final = pixel_scores_final > 0
        num_valid_final = mx.sum(valid_mask_final)
        score_vmap_final = mx.sum(pixel_scores_final) / mx.maximum(num_valid_final, 1.0)
        mx.eval(score_vec_final, score_vmap_final)
        
        diff = abs(float(score_vec_final - score_vmap_final))
        print(f"  Différence numérique: {diff:.8f}")
    
    # ============ ANALYSE DES RÉSULTATS ============
    print(f"\n{'='*60}")
    print("📊 ANALYSE DÉTAILLÉE DES PERFORMANCES")
    print(f"{'='*60}")
    
    print(f"{'Taille':<12} {'Éléments':<10} {'Vec(ms)':<10} {'Vmap(ms)':<10} {'Speedup':<8}")
    print("-" * 60)
    
    for i, size in enumerate(results['sizes']):
        print(f"{size:<12} {results['total_elements'][i]:<10,} "
              f"{results['vectorized_times'][i]:<10.2f} "
              f"{results['vmap_times'][i]:<10.2f} "
              f"{results['speedups'][i]:<8.1f}x")
    
    # Analyse des tendances
    print(f"\n🔍 TENDANCES:")
    min_speedup = min(results['speedups'])
    max_speedup = max(results['speedups'])
    avg_speedup = np.mean(results['speedups'])
    
    print(f"  Speedup minimum: {min_speedup:.1f}x")
    print(f"  Speedup maximum: {max_speedup:.1f}x")
    print(f"  Speedup moyen:   {avg_speedup:.1f}x")
    
    # Efficacité par élément
    print(f"\n⚡ EFFICACITÉ PAR ÉLÉMENT:")
    for i, size in enumerate(results['sizes']):
        vec_per_elem = results['vectorized_times'][i] / results['total_elements'][i] * 1e6  # en µs
        vmap_per_elem = results['vmap_times'][i] / results['total_elements'][i] * 1e6
        print(f"  {size}: Vec={vec_per_elem:.3f}µs/elem, Vmap={vmap_per_elem:.3f}µs/elem")
    
    return results

def benchmark_operations():
    # Simule tes flow vectors (H, W, 2) comme dans ton cas
    H, W = 480, 640  # Résolution typique de tes vidéos
    flow_vectors = mx.random.normal((H, W, 2))  # Vecteurs de flow optique
    vanishing_point = mx.array([320.0, 240.0])  # Point de fuite estimé
    
    # Précalcul des directions depuis le point de fuite vers chaque pixel
    y_coords, x_coords = mx.meshgrid(mx.arange(H), mx.arange(W), indexing='ij')
    pixel_coords = mx.stack([x_coords, y_coords], axis=-1)
    directions_to_vp = vanishing_point - pixel_coords
    
    print(f"Tenseur flow_vectors: {flow_vectors.shape}")
    # MLX arrays don't have a device attribute - they're automatically managed
    # print(f"Device utilisé: {flow_vectors.device}")
    
    # ============ TEST 1: Calcul de colinéarité ============
    def collinearity_vectorized(flow_vecs, directions):
        """Version vectorisée - ton cas d'usage principal"""
        # Normalisation
        flow_norms = mx.linalg.norm(flow_vecs, axis=-1, keepdims=True)
        dir_norms = mx.linalg.norm(directions, axis=-1, keepdims=True)
        
        # Évite division par zéro
        flow_normalized = flow_vecs / mx.maximum(flow_norms, 1e-8)
        dir_normalized = directions / mx.maximum(dir_norms, 1e-8)
        
        # Produit scalaire pour colinéarité
        dot_products = mx.sum(flow_normalized * dir_normalized, axis=-1)
        return mx.abs(dot_products)
    
    def collinearity_single(flow_vec, direction):
        """Version pour un seul vecteur - compatible vmap"""
        flow_norm = mx.linalg.norm(flow_vec)
        dir_norm = mx.linalg.norm(direction)
        
        # Évite division par zéro sans conditions
        flow_normalized = flow_vec / mx.maximum(flow_norm, 1e-8)
        dir_normalized = direction / mx.maximum(dir_norm, 1e-8)
        
        # Masque pour les normes trop petites
        valid_mask = (flow_norm >= 1e-8) & (dir_norm >= 1e-8)
        
        dot_product = mx.sum(flow_normalized * dir_normalized)
        return mx.where(valid_mask, mx.abs(dot_product), 0.0)
    
    # Benchmark colinéarité vectorisée
    times = []
    for _ in range(10):
        start = time.time()
        result_vec = collinearity_vectorized(flow_vectors, directions_to_vp)
        mx.eval(result_vec)  # Force l'évaluation
        times.append(time.time() - start)
    
    avg_time_vec = np.mean(times[2:])  # Ignore les 2 premiers (warmup)
    print(f"\n📊 Colinéarité vectorisée: {avg_time_vec*1000:.2f}ms")
    
    # Benchmark colinéarité avec vmap
    collinearity_vmap = mx.vmap(mx.vmap(collinearity_single, in_axes=(0, 0)), in_axes=(0, 0))
    
    times = []
    for _ in range(10):
        start = time.time()
        result_vmap = collinearity_vmap(flow_vectors, directions_to_vp)
        mx.eval(result_vmap)
        times.append(time.time() - start)
    
    avg_time_vmap = np.mean(times[2:])
    print(f"📊 Colinéarité vmap: {avg_time_vmap*1000:.2f}ms")
    print(f"🚀 Speedup vectorisé: {avg_time_vmap/avg_time_vec:.1f}x")
    
    # ============ TEST 2: Filtrage par norme ============
    def filter_by_norm_vectorized(vectors, threshold=1e-2):
        """Filtre vectorisé par norme"""
        norms = mx.linalg.norm(vectors, axis=-1)
        mask = norms > threshold
        return vectors * mask[..., None], mask
    
    def filter_by_norm_single(vector, threshold=1e-2):
        """Filtre un seul vecteur - compatible vmap"""
        norm = mx.linalg.norm(vector)
        valid = norm > threshold
        return mx.where(valid, vector, mx.zeros_like(vector))
    
    def get_norm_mask_single(vector, threshold=1e-2):
        """Retourne le masque de validité pour un vecteur"""
        norm = mx.linalg.norm(vector)
        return norm > threshold
    
    # Benchmark filtrage vectorisé
    times = []
    for _ in range(10):
        start = time.time()
        filtered_vec, mask_vec = filter_by_norm_vectorized(flow_vectors)
        mx.eval(filtered_vec, mask_vec)
        times.append(time.time() - start)
    
    avg_time_filter_vec = np.mean(times[2:])
    print(f"\n📊 Filtrage norme vectorisé: {avg_time_filter_vec*1000:.2f}ms")
    
    # Benchmark filtrage avec vmap
    threshold_array = mx.array(1e-2)
    filter_vmap = mx.vmap(mx.vmap(filter_by_norm_single, in_axes=(0, None)), in_axes=(0, None))
    mask_vmap = mx.vmap(mx.vmap(get_norm_mask_single, in_axes=(0, None)), in_axes=(0, None))
    
    times = []
    for _ in range(10):
        start = time.time()
        filtered_vmap = filter_vmap(flow_vectors, threshold_array)
        mask_vmap_result = mask_vmap(flow_vectors, threshold_array)
        mx.eval(filtered_vmap, mask_vmap_result)
        times.append(time.time() - start)
    
    avg_time_filter_vmap = np.mean(times[2:])
    print(f"📊 Filtrage norme vmap: {avg_time_filter_vmap*1000:.2f}ms")
    print(f"🚀 Speedup vectorisé: {avg_time_filter_vmap/avg_time_filter_vec:.1f}x")
    
    # ============ TEST 3: Calcul de score global (comme ton cas) ============
    def global_score_vectorized(flow_vecs, directions, norm_threshold=1e-2):
        """Score global vectorisé - proche de ton pipeline"""
        # Filtrage par norme
        norms = mx.linalg.norm(flow_vecs, axis=-1)
        valid_mask = norms > norm_threshold
        
        # Colinéarité seulement sur pixels valides
        flow_normalized = flow_vecs / mx.maximum(norms[..., None], 1e-8)
        dir_norms = mx.linalg.norm(directions, axis=-1)
        dir_normalized = directions / mx.maximum(dir_norms[..., None], 1e-8)
        
        collinearities = mx.abs(mx.sum(flow_normalized * dir_normalized, axis=-1))
        
        # Score final (moyenne pondérée)
        valid_collinearities = collinearities * valid_mask
        return mx.sum(valid_collinearities) / mx.sum(valid_mask)
    
    def global_score_complex_vmap(flow_vecs, directions, norm_threshold=1e-2):
        """Version vmap plus complexe"""
        def score_pixel(flow_vec, direction):
            norm = mx.linalg.norm(flow_vec)
            valid = norm > norm_threshold
            
            # Normalisation sécurisée
            flow_norm = flow_vec / mx.maximum(norm, 1e-8)
            dir_norm = direction / mx.maximum(mx.linalg.norm(direction), 1e-8)
            
            collinearity = mx.abs(mx.sum(flow_norm * dir_norm))
            return mx.where(valid, collinearity, 0.0)
        
        pixel_scores = mx.vmap(mx.vmap(score_pixel, in_axes=(0, 0)), in_axes=(0, 0))
        scores = pixel_scores(flow_vecs, directions)
        valid_count = mx.sum(scores > 0)
        return mx.sum(scores) / mx.maximum(valid_count, 1.0)
    
    # Benchmark score global vectorisé
    times = []
    for _ in range(10):
        start = time.time()
        score_vec = global_score_vectorized(flow_vectors, directions_to_vp)
        mx.eval(score_vec)
        times.append(time.time() - start)
    
    avg_time_global_vec = np.mean(times[2:])
    print(f"\n📊 Score global vectorisé: {avg_time_global_vec*1000:.2f}ms")
    
    # Benchmark score global vmap
    times = []
    for _ in range(10):
        start = time.time()
        score_vmap = global_score_complex_vmap(flow_vectors, directions_to_vp)
        mx.eval(score_vmap)
        times.append(time.time() - start)
    
    avg_time_global_vmap = np.mean(times[2:])
    print(f"📊 Score global vmap: {avg_time_global_vmap*1000:.2f}ms")
    print(f"🚀 Speedup vectorisé: {avg_time_global_vmap/avg_time_global_vec:.1f}x")
    
    # ============ RÉSUMÉ ============
    print(f"\n{'='*50}")
    print("📈 RÉSUMÉ DES PERFORMANCES")
    print(f"{'='*50}")
    print(f"Colinéarité      : {avg_time_vmap/avg_time_vec:.1f}x plus rapide vectorisé")
    print(f"Filtrage norme   : {avg_time_filter_vmap/avg_time_filter_vec:.1f}x plus rapide vectorisé")
    print(f"Score global     : {avg_time_global_vmap/avg_time_global_vec:.1f}x plus rapide vectorisé")
    
    # Vérification que les résultats sont cohérents
    print(f"\n🔍 VÉRIFICATION COHÉRENCE")
    score_vec_final = global_score_vectorized(flow_vectors, directions_to_vp)
    score_vmap_final = global_score_complex_vmap(flow_vectors, directions_to_vp)
    mx.eval(score_vec_final, score_vmap_final)
    print(f"Score vectorisé: {float(score_vec_final):.6f}")
    print(f"Score vmap: {float(score_vmap_final):.6f}")
    print(f"Différence: {abs(float(score_vec_final - score_vmap_final)):.8f}")

def benchmark_batch_images():
    """
    Benchmark pour le traitement de batches d'images avec vmap
    (le vrai cas d'usage : paralléliser sur les images, pas les pixels)
    """
    print("🎬 BENCHMARK BATCH D'IMAGES")
    print("="*60)
    
    # Paramètres réalistes pour ton projet
    H, W = 480, 640  # Résolution vidéo typique
    batch_sizes = [1, 2, 4, 8, 16, 32]  # Différentes tailles de batch
    
    results = {
        'batch_sizes': [],
        'sequential_times': [],
        'vmap_times': [],
        'speedups': [],
        'throughput_sequential': [],
        'throughput_vmap': []
    }
    
    def process_single_image(flow_vectors):
        """Traite une seule image - fonction de base"""
        # Point de fuite fixe pour le test
        vanishing_point = mx.array([W/2.0, H/2.0])
        
        # Calcul des directions depuis le point de fuite
        y_coords, x_coords = mx.meshgrid(mx.arange(H), mx.arange(W), indexing='ij')
        pixel_coords = mx.stack([x_coords, y_coords], axis=-1)
        directions = vanishing_point - pixel_coords
        
        # Score de colinéarité (version vectorisée optimale)
        flow_norms = mx.linalg.norm(flow_vectors, axis=-1)
        valid_mask = flow_norms > 1e-2
        
        flow_normalized = flow_vectors / mx.maximum(flow_norms[..., None], 1e-8)
        dir_norms = mx.linalg.norm(directions, axis=-1)
        dir_normalized = directions / mx.maximum(dir_norms[..., None], 1e-8)
        
        collinearities = mx.abs(mx.sum(flow_normalized * dir_normalized, axis=-1))
        valid_collinearities = collinearities * valid_mask
        
        return mx.sum(valid_collinearities) / mx.maximum(mx.sum(valid_mask), 1.0)
    
    # Fonction vmap pour traiter un batch d'images
    process_batch = mx.vmap(process_single_image, in_axes=0)
    
    for batch_size in batch_sizes:
        print(f"\n📦 Test avec batch de {batch_size} images")
        
        # Génère un batch d'images (flow vectors)
        batch_flow_vectors = mx.random.normal((batch_size, H, W, 2))
        
        # ============ TRAITEMENT SÉQUENTIEL ============
        times_seq = []
        for _ in range(10):
            start = time.time()
            scores_seq = []
            for i in range(batch_size):
                score = process_single_image(batch_flow_vectors[i])
                scores_seq.append(score)
            # Force l'évaluation de tous les scores
            mx.eval(*scores_seq)
            times_seq.append(time.time() - start)
        
        avg_time_seq = np.mean(times_seq[2:])  # Skip warmup
        
        # ============ TRAITEMENT VMAP (BATCH) ============
        times_vmap = []
        for _ in range(10):
            start = time.time()
            scores_vmap = process_batch(batch_flow_vectors)
            mx.eval(scores_vmap)
            times_vmap.append(time.time() - start)
        
        avg_time_vmap = np.mean(times_vmap[2:])
        
        # ============ CALCUL DES MÉTRIQUES ============
        speedup = avg_time_seq / avg_time_vmap
        throughput_seq = batch_size / avg_time_seq  # images/sec
        throughput_vmap = batch_size / avg_time_vmap
        
        results['batch_sizes'].append(batch_size)
        results['sequential_times'].append(avg_time_seq * 1000)  # ms
        results['vmap_times'].append(avg_time_vmap * 1000)
        results['speedups'].append(speedup)
        results['throughput_sequential'].append(throughput_seq)
        results['throughput_vmap'].append(throughput_vmap)
        
        print(f"  Séquentiel: {avg_time_seq*1000:.2f}ms ({throughput_seq:.1f} img/s)")
        print(f"  Vmap batch: {avg_time_vmap*1000:.2f}ms ({throughput_vmap:.1f} img/s)")
        print(f"  Speedup:    {speedup:.1f}x")
        
        # Vérification cohérence
        scores_seq_check = [process_single_image(batch_flow_vectors[i]) for i in range(batch_size)]
        scores_vmap_check = process_batch(batch_flow_vectors)
        mx.eval(*scores_seq_check, scores_vmap_check)
        
        max_diff = max(abs(float(scores_seq_check[i] - scores_vmap_check[i])) for i in range(batch_size))
        print(f"  Différence max: {max_diff:.8f}")
    
    # ============ ANALYSE DES RÉSULTATS ============
    print(f"\n{'='*60}")
    print("📊 ANALYSE BATCH PROCESSING")
    print(f"{'='*60}")
    
    print(f"{'Batch':<8} {'Seq(ms)':<10} {'Vmap(ms)':<10} {'Speedup':<10} {'Throughput':<15}")
    print("-" * 65)
    
    for i, batch_size in enumerate(results['batch_sizes']):
        throughput_gain = results['throughput_vmap'][i] / results['throughput_sequential'][i]
        print(f"{batch_size:<8} {results['sequential_times'][i]:<10.2f} "
              f"{results['vmap_times'][i]:<10.2f} {results['speedups'][i]:<10.1f}x "
              f"{results['throughput_vmap'][i]:<7.1f} img/s")
    
    # Analyse des tendances
    print(f"\n🚀 TENDANCES BATCH PROCESSING:")
    min_speedup = min(results['speedups'])
    max_speedup = max(results['speedups'])
    avg_speedup = np.mean(results['speedups'])
    
    print(f"  Speedup minimum: {min_speedup:.1f}x (batch {results['batch_sizes'][results['speedups'].index(min_speedup)]})")
    print(f"  Speedup maximum: {max_speedup:.1f}x (batch {results['batch_sizes'][results['speedups'].index(max_speedup)]})")
    print(f"  Speedup moyen:   {avg_speedup:.1f}x")
    
    # Efficacité de parallélisation
    print(f"\n⚡ EFFICACITÉ DE PARALLÉLISATION:")
    for i, batch_size in enumerate(results['batch_sizes']):
        if batch_size > 1:
            theoretical_speedup = batch_size  # Speedup théorique parfait
            actual_speedup = results['speedups'][i]
            efficiency = (actual_speedup / theoretical_speedup) * 100
            print(f"  Batch {batch_size}: {efficiency:.1f}% d'efficacité ({actual_speedup:.1f}x / {theoretical_speedup}x théorique)")
    
    # Recommandations
    best_batch_idx = results['speedups'].index(max_speedup)
    best_batch_size = results['batch_sizes'][best_batch_idx]
    
    print(f"\n💡 RECOMMANDATIONS:")
    print(f"  Taille de batch optimale: {best_batch_size} images")
    print(f"  Gain de throughput max: {max_speedup:.1f}x")
    print(f"  Throughput optimal: {results['throughput_vmap'][best_batch_idx]:.1f} images/seconde")
    
    return results

def benchmark_batch_vs_vectorized():
    """
    Compare vmap batch processing vs vectorized operations on batches
    (ajouter une dimension batch aux opérations vectorisées)
    """
    print("⚔️  VMAP vs VECTORISATION BATCH")
    print("="*60)
    
    # Paramètres
    H, W = 480, 640
    batch_sizes = [1, 2, 4, 8, 16, 32]
    
    results = {
        'batch_sizes': [],
        'sequential_times': [],
        'vmap_times': [],
        'vectorized_batch_times': [],
        'vmap_vs_seq_speedup': [],
        'vec_vs_seq_speedup': [],
        'vmap_vs_vec_speedup': []
    }
    
    def process_single_image(flow_vectors):
        """Traite une seule image"""
        vanishing_point = mx.array([W/2.0, H/2.0])
        
        y_coords, x_coords = mx.meshgrid(mx.arange(H), mx.arange(W), indexing='ij')
        pixel_coords = mx.stack([x_coords, y_coords], axis=-1)
        directions = vanishing_point - pixel_coords
        
        flow_norms = mx.linalg.norm(flow_vectors, axis=-1)
        valid_mask = flow_norms > 1e-2
        
        flow_normalized = flow_vectors / mx.maximum(flow_norms[..., None], 1e-8)
        dir_norms = mx.linalg.norm(directions, axis=-1)
        dir_normalized = directions / mx.maximum(dir_norms[..., None], 1e-8)
        
        collinearities = mx.abs(mx.sum(flow_normalized * dir_normalized, axis=-1))
        valid_collinearities = collinearities * valid_mask
        
        return mx.sum(valid_collinearities) / mx.maximum(mx.sum(valid_mask), 1.0)
    
    def process_batch_vectorized(batch_flow_vectors):
        """Traite un batch d'images avec vectorisation pure (dimension batch ajoutée)"""
        batch_size = batch_flow_vectors.shape[0]
        vanishing_point = mx.array([W/2.0, H/2.0])
        
        # Répète le point de fuite pour chaque image du batch
        vanishing_points = mx.broadcast_to(vanishing_point[None, :], (batch_size, 2))
        
        # Coordonnées des pixels (même pour toutes les images)
        y_coords, x_coords = mx.meshgrid(mx.arange(H), mx.arange(W), indexing='ij')
        pixel_coords = mx.stack([x_coords, y_coords], axis=-1)
        
        # Broadcast pour le batch : (batch_size, H, W, 2)
        pixel_coords_batch = mx.broadcast_to(pixel_coords[None, :, :, :], 
                                           (batch_size, H, W, 2))
        
        # Directions depuis le point de fuite : (batch_size, H, W, 2)
        directions_batch = vanishing_points[:, None, None, :] - pixel_coords_batch
        
        # Calculs vectorisés sur tout le batch
        flow_norms = mx.linalg.norm(batch_flow_vectors, axis=-1)  # (batch_size, H, W)
        valid_mask = flow_norms > 1e-2
        
        # Normalisation
        flow_normalized = batch_flow_vectors / mx.maximum(flow_norms[..., None], 1e-8)
        dir_norms = mx.linalg.norm(directions_batch, axis=-1)
        dir_normalized = directions_batch / mx.maximum(dir_norms[..., None], 1e-8)
        
        # Colinéarité pour tout le batch
        collinearities = mx.abs(mx.sum(flow_normalized * dir_normalized, axis=-1))
        valid_collinearities = collinearities * valid_mask
        
        # Score par image : (batch_size,)
        scores = mx.sum(valid_collinearities, axis=(1, 2)) / mx.maximum(
            mx.sum(valid_mask, axis=(1, 2)), 1.0)
        
        return scores
    
    # Fonction vmap
    process_batch_vmap = mx.vmap(process_single_image, in_axes=0)
    
    for batch_size in batch_sizes:
        print(f"\n📦 Batch de {batch_size} images")
        
        # Données
        batch_flow_vectors = mx.random.normal((batch_size, H, W, 2))
        
        # ============ SÉQUENTIEL ============
        times_seq = []
        for _ in range(10):
            start = time.time()
            scores_seq = []
            for i in range(batch_size):
                score = process_single_image(batch_flow_vectors[i])
                scores_seq.append(score)
            mx.eval(*scores_seq)
            times_seq.append(time.time() - start)
        avg_time_seq = np.mean(times_seq[2:])
        
        # ============ VMAP ============
        times_vmap = []
        for _ in range(10):
            start = time.time()
            scores_vmap = process_batch_vmap(batch_flow_vectors)
            mx.eval(scores_vmap)
            times_vmap.append(time.time() - start)
        avg_time_vmap = np.mean(times_vmap[2:])
        
        # ============ VECTORISÉ BATCH ============
        times_vec_batch = []
        for _ in range(10):
            start = time.time()
            scores_vec_batch = process_batch_vectorized(batch_flow_vectors)
            mx.eval(scores_vec_batch)
            times_vec_batch.append(time.time() - start)
        avg_time_vec_batch = np.mean(times_vec_batch[2:])
        
        # ============ CALCULS ============
        vmap_vs_seq = avg_time_seq / avg_time_vmap
        vec_vs_seq = avg_time_seq / avg_time_vec_batch
        vmap_vs_vec = avg_time_vec_batch / avg_time_vmap
        
        results['batch_sizes'].append(batch_size)
        results['sequential_times'].append(avg_time_seq * 1000)
        results['vmap_times'].append(avg_time_vmap * 1000)
        results['vectorized_batch_times'].append(avg_time_vec_batch * 1000)
        results['vmap_vs_seq_speedup'].append(vmap_vs_seq)
        results['vec_vs_seq_speedup'].append(vec_vs_seq)
        results['vmap_vs_vec_speedup'].append(vmap_vs_vec)
        
        print(f"  Séquentiel:     {avg_time_seq*1000:.2f}ms")
        print(f"  Vmap:           {avg_time_vmap*1000:.2f}ms (speedup: {vmap_vs_seq:.1f}x)")
        print(f"  Vec batch:      {avg_time_vec_batch*1000:.2f}ms (speedup: {vec_vs_seq:.1f}x)")
        print(f"  Vmap vs Vec:    {vmap_vs_vec:.1f}x {'(vmap gagne)' if vmap_vs_vec < 1 else '(vec gagne)'}")
        
        # Vérification cohérence
        scores_seq_check = [process_single_image(batch_flow_vectors[i]) for i in range(batch_size)]
        scores_vmap_check = process_batch_vmap(batch_flow_vectors)
        scores_vec_check = process_batch_vectorized(batch_flow_vectors)
        mx.eval(*scores_seq_check, scores_vmap_check, scores_vec_check)
        
        max_diff_vmap = max(abs(float(scores_seq_check[i] - scores_vmap_check[i])) for i in range(batch_size))
        max_diff_vec = max(abs(float(scores_seq_check[i] - scores_vec_check[i])) for i in range(batch_size))
        print(f"  Diff vmap: {max_diff_vmap:.8f}, Diff vec: {max_diff_vec:.8f}")
    
    # ============ ANALYSE ============
    print(f"\n{'='*60}")
    print("📊 COMPARAISON VMAP vs VECTORISATION BATCH")
    print(f"{'='*60}")
    
    print(f"{'Batch':<8} {'Seq(ms)':<10} {'Vmap(ms)':<10} {'Vec(ms)':<10} {'Vmap/Seq':<10} {'Vec/Seq':<10} {'Vmap/Vec':<10}")
    print("-" * 80)
    
    for i, batch_size in enumerate(results['batch_sizes']):
        print(f"{batch_size:<8} {results['sequential_times'][i]:<10.2f} "
              f"{results['vmap_times'][i]:<10.2f} {results['vectorized_batch_times'][i]:<10.2f} "
              f"{results['vmap_vs_seq_speedup'][i]:<10.1f}x {results['vec_vs_seq_speedup'][i]:<10.1f}x "
              f"{results['vmap_vs_vec_speedup'][i]:<10.1f}x")
    
    # Tendances
    print(f"\n🏆 GAGNANT PAR TAILLE DE BATCH:")
    for i, batch_size in enumerate(results['batch_sizes']):
        if results['vmap_vs_vec_speedup'][i] < 1:
            winner = "VMAP"
            advantage = 1 / results['vmap_vs_vec_speedup'][i]
        else:
            winner = "VECTORISÉ"
            advantage = results['vmap_vs_vec_speedup'][i]
        print(f"  Batch {batch_size}: {winner} gagne ({advantage:.1f}x plus rapide)")
    
    # Moyennes
    avg_vmap_speedup = np.mean(results['vmap_vs_seq_speedup'])
    avg_vec_speedup = np.mean(results['vec_vs_seq_speedup'])
    avg_vmap_vs_vec = np.mean(results['vmap_vs_vec_speedup'])
    
    print(f"\n📈 MOYENNES:")
    print(f"  Vmap vs séquentiel: {avg_vmap_speedup:.1f}x")
    print(f"  Vec vs séquentiel:  {avg_vec_speedup:.1f}x")
    print(f"  Vmap vs Vec:        {avg_vmap_vs_vec:.1f}x")
    
    if avg_vmap_vs_vec < 1:
        print(f"  🏆 VMAP est {1/avg_vmap_vs_vec:.1f}x plus rapide en moyenne")
    else:
        print(f"  🏆 VECTORISATION BATCH est {avg_vmap_vs_vec:.1f}x plus rapide en moyenne")
    
    return results

if __name__ == "__main__":
    # Benchmark original
    print("🔥 BENCHMARK ORIGINAL")
    print("="*50)
    benchmark_operations()
    
    print("\n" + "="*80)
    
    # Nouveau benchmark pour gradients
    gradient_results = benchmark_gradient_scenario()
    
    print("\n" + "="*80)
    
    # Nouveau benchmark pour batch images
    batch_results = benchmark_batch_images()
    
    print("\n" + "="*80)
    
    # Nouveau benchmark pour comparer vmap batch processing vs vectorized operations on batches
    batch_vs_vectorized_results = benchmark_batch_vs_vectorized()