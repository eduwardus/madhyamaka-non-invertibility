# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 21:04:58 2026

@author: eggra
"""

#!/usr/bin/env python3
"""
Experimento 5: Latent Projection Graph Model (LPGM) - Degeneración estructural.
Modelo: x_v ~ N(0, I_2) en R^2, se proyecta a h_v = ||x_v|| (radio).
P_uv = sigmoid(alpha - |h_u - h_v|).

Objetivo: Demostrar degeneración real: múltiples embeddings 2D incompatibles
(no relacionados por rotación/traslación) producen el mismo grafo.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import procrustes
from scipy.stats import pearsonr
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import time
import sys
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. MODELO CON PROYECCIÓN RADIAL
# =============================================================================

def sigmoid_stable(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def generate_graph_radial_projection(seed, alpha, N=500, d_latent=2, 
                                     output_dir="experiment_5"):
    """
    Genera grafo según modelo de proyección radial.
    x_v ~ N(0, I_d), h_v = ||x_v||, P_uv = sigmoid(alpha - |h_u - h_v|).
    """
    np.random.seed(seed)
    # Latente real 2D
    X_true = np.random.randn(N, d_latent)
    # Proyección radial (pérdida de información angular)
    h = np.linalg.norm(X_true, axis=1)  # shape (N,)
    # Matriz de diferencias absolutas
    dh = np.abs(h[:, None] - h[None, :])
    Z = alpha - dh
    P = sigmoid_stable(Z)
    np.fill_diagonal(P, 0)
    # Muestrear grafo
    A = np.random.rand(N, N) < P
    A = np.triu(A, 1)
    A = A + A.T
    A_sparse = sp.csr_matrix(A)
    
    # Guardar
    alpha_dir = f"{output_dir}/alpha_{alpha}"
    os.makedirs(alpha_dir, exist_ok=True)
    np.savez(f"{alpha_dir}/graph_seed_{seed}.npz",
             adjacency_matrix=A_sparse.data,
             indices=A_sparse.indices,
             indptr=A_sparse.indptr,
             shape=A_sparse.shape,
             latent_true=X_true,
             h=h,
             seed=seed,
             alpha=alpha)
    np.savetxt(f"{alpha_dir}/latent_true_seed_{seed}.dat", X_true, header="x y", comments='')
    np.savetxt(f"{alpha_dir}/radial_projection_seed_{seed}.dat", h, header="h")
    return A_sparse, X_true, h, P


# =============================================================================
# 2. INFERENCIA MÚLTIPLE VÍA MDS (intentando recuperar 2D)
# =============================================================================

def adjacency_to_distance_matrix(A_sparse):
    try:
        dist_matrix = shortest_path(A_sparse, directed=False, unweighted=True)
        max_dist = np.max(dist_matrix[dist_matrix != np.inf])
        dist_matrix = np.nan_to_num(dist_matrix, nan=max_dist * 2)
        dist_matrix[dist_matrix == np.inf] = max_dist * 2
        return dist_matrix
    except Exception:
        A_dense = A_sparse.toarray()
        dist_matrix = np.where(A_dense > 0, 1, np.sqrt(2))
        np.fill_diagonal(dist_matrix, 0)
        return dist_matrix


def recover_multiple_embeddings(A_sparse, d=2, n_solutions=20, verbose=False):
    """Genera múltiples embeddings 2D para el mismo grafo usando MDS."""
    dist_matrix = adjacency_to_distance_matrix(A_sparse)
    dist_matrix = dist_matrix / (np.max(dist_matrix) + 1e-10)
    solutions = []
    stresses = []
    for i in range(n_solutions):
        mds = MDS(n_components=d, dissimilarity='precomputed',
                  random_state=i, n_init=1, max_iter=500,
                  normalized_stress='auto', verbose=0)
        try:
            X = mds.fit_transform(dist_matrix)
            solutions.append(X)
            stresses.append(mds.stress_)
        except Exception as e:
            if verbose:
                print(f"    Solución {i} falló: {e}")
            continue
    if verbose:
        print(f"    Generadas {len(solutions)} soluciones válidas")
    return solutions, stresses


# =============================================================================
# 3. ANÁLISIS DE DEGENERACIÓN
# =============================================================================

def procrustes_alignment(X_ref, X_target):
    _, X_aligned, _ = procrustes(X_ref, X_target)
    return X_aligned


def solution_distance_matrix(solutions):
    n = len(solutions)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            Xj_aligned = procrustes_alignment(solutions[i], solutions[j])
            dist = np.mean((solutions[i] - Xj_aligned)**2)
            dist_mat[i, j] = dist_mat[j, i] = dist
    return dist_mat


def effective_dimension_of_solutions(solutions):
    """Dimensión efectiva del espacio de soluciones (PCA sobre vectores aplanados)."""
    n_sol = len(solutions)
    if n_sol < 3:
        return 1.0
    vectors = np.array([sol.flatten() for sol in solutions])
    pca = PCA()
    pca.fit(vectors)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    eff_dim = np.argmax(cumsum >= 0.95) + 1
    return eff_dim, pca.explained_variance_ratio_


def correlation_with_true_radial(solutions, h_true):
    """Correlación entre distancias euclídeas de las soluciones y diferencias radiales reales."""
    # h_true es el radio real (1D)
    # Queremos ver si la estructura 2D recuperada refleja la información radial
    # Una métrica: correlación entre la primera coordenada (o alguna combinación) y h_true
    # Pero mejor: correlación entre la distancia euclídea entre puntos en la solución y |h_i - h_j|
    n = solutions[0].shape[0]
    dh_true = np.abs(h_true[:, None] - h_true[None, :])
    triu = np.triu_indices(n, k=1)
    corrs = []
    for sol in solutions:
        D_sol = np.sum((sol[:, None, :] - sol[None, :, :])**2, axis=-1)
        d_sol_triu = D_sol[triu]
        dh_triu = dh_true[triu]
        # Correlación de Pearson
        try:
            r, _ = pearsonr(d_sol_triu, dh_triu)
            corrs.append(r)
        except:
            corrs.append(0.0)
    return np.mean(corrs), np.std(corrs)


def run_experiment_alpha(alpha, n_seeds=5, n_solutions=20, N=500, verbose=False):
    print(f"\n{'='*60}")
    print(f"α = {alpha:.1f}")
    print(f"{'='*60}")
    
    all_solutions = []
    all_stresses = []
    true_hs = []
    graphs = []
    
    for seed in range(n_seeds):
        A_sparse, X_true, h_true, _ = generate_graph_radial_projection(
            seed, alpha, N=N, output_dir="experiment_5"
        )
        graphs.append(A_sparse)
        true_hs.append(h_true)
        
        sols, stresses = recover_multiple_embeddings(
            A_sparse, d=2, n_solutions=n_solutions, verbose=verbose
        )
        all_solutions.append(sols)
        all_stresses.append(stresses)
        
        if verbose:
            print(f"  Seed {seed}: {len(sols)} soluciones")
    
    # Métricas por semilla
    per_seed = []
    for s_idx in range(n_seeds):
        sols = all_solutions[s_idx]
        if len(sols) < 2:
            continue
        # Matriz de distancias entre soluciones
        dist_mat = solution_distance_matrix(sols)
        triu = np.triu_indices_from(dist_mat, k=1)
        distances = dist_mat[triu]
        mean_dist = np.mean(distances)
        # Dimensión efectiva
        eff_dim, _ = effective_dimension_of_solutions(sols)
        # Correlación con información radial real
        corr_mean, corr_std = correlation_with_true_radial(sols, true_hs[s_idx])
        per_seed.append({
            'seed': s_idx,
            'n_solutions': len(sols),
            'mean_solution_distance': mean_dist,
            'effective_dimension': eff_dim,
            'correlation_mean': corr_mean,
            'correlation_std': corr_std,
            'mean_stress': np.mean(all_stresses[s_idx])
        })
    
    if not per_seed:
        return None
    
    # Agregados
    metrics = {
        'alpha': alpha,
        'n_seeds': n_seeds,
        'mean_solution_distance': np.mean([p['mean_solution_distance'] for p in per_seed]),
        'effective_dimension': np.mean([p['effective_dimension'] for p in per_seed]),
        'correlation_mean': np.mean([p['correlation_mean'] for p in per_seed]),
        'correlation_std': np.mean([p['correlation_std'] for p in per_seed]),
        'mean_stress': np.mean([p['mean_stress'] for p in per_seed]),
        'per_seed': per_seed
    }
    
    print(f"  Degeneración: distancia media = {metrics['mean_solution_distance']:.6f}")
    print(f"  Dimensión efectiva = {metrics['effective_dimension']:.1f}")
    print(f"  Correlación con radio real = {metrics['correlation_mean']:.4f} ± {metrics['correlation_std']:.4f}")
    return metrics


def plot_results(all_metrics, output_dir="experiment_5"):
    if not all_metrics:
        return
    alphas = [m['alpha'] for m in all_metrics]
    distances = [m['mean_solution_distance'] for m in all_metrics]
    eff_dims = [m['effective_dimension'] for m in all_metrics]
    corrs = [m['correlation_mean'] for m in all_metrics]
    corr_stds = [m['correlation_std'] for m in all_metrics]
    stresses = [m['mean_stress'] for m in all_metrics]
    
    plt.figure(figsize=(15, 10))
    
    # Distancia entre soluciones
    plt.subplot(2,3,1)
    plt.plot(alphas, distances, 'o-', color='red', linewidth=2)
    plt.xlabel('α')
    plt.ylabel('Distancia media')
    plt.title('Degeneración')
    plt.grid(True)
    
    # Dimensión efectiva
    plt.subplot(2,3,2)
    plt.plot(alphas, eff_dims, 'o-', color='blue', linewidth=2)
    plt.xlabel('α')
    plt.ylabel('Dimensión efectiva')
    plt.title('Dimensión del espacio de soluciones')
    plt.axhline(y=2, color='gray', linestyle='--', label='d=2 (latente real)')
    plt.legend()
    plt.grid(True)
    
    # Correlación con radio real
    plt.subplot(2,3,3)
    plt.errorbar(alphas, corrs, yerr=corr_stds, fmt='o-', capsize=5, color='green')
    plt.xlabel('α')
    plt.ylabel('Correlación con radio real')
    plt.title('Información recuperada')
    plt.grid(True)
    
    # Stress MDS
    plt.subplot(2,3,4)
    plt.plot(alphas, stresses, 'o-', color='purple')
    plt.xlabel('α')
    plt.ylabel('Stress MDS')
    plt.title('Calidad de ajuste')
    plt.grid(True)
    
    # Relación degeneración vs correlación
    plt.subplot(2,3,5)
    plt.scatter(corrs, distances, c=alphas, cmap='viridis', s=80)
    plt.colorbar(label='α')
    plt.xlabel('Correlación con radio')
    plt.ylabel('Degeneración')
    plt.title('Compromiso')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/radial_model_results.png", dpi=150)
    plt.close()
    print(f"  Guardada figura: {output_dir}/radial_model_results.png")


def visualize_solutions(alpha, seed=0, n_solutions=5, output_dir="experiment_5"):
    """Genera una figura mostrando diferentes soluciones para un mismo grafo."""
    A_sparse, X_true, h_true, _ = generate_graph_radial_projection(
        seed, alpha, N=200, output_dir=output_dir
    )
    sols, _ = recover_multiple_embeddings(A_sparse, d=2, n_solutions=n_solutions, verbose=False)
    if len(sols) < 2:
        print("No hay suficientes soluciones para visualizar.")
        return
    plt.figure(figsize=(12, 6))
    for i, sol in enumerate(sols[:5]):
        plt.subplot(2, 3, i+1)
        plt.scatter(sol[:, 0], sol[:, 1], s=5, alpha=0.6)
        plt.title(f"Solución {i}")
        plt.axis('equal')
    plt.suptitle(f"α={alpha}: Diferentes embeddings 2D para el mismo grafo")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/solutions_alpha_{alpha}.png", dpi=150)
    plt.close()
    print(f"  Guardada figura: solutions_alpha_{alpha}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Experimento 5: Modelo de Proyección Radial (LPGM)"
    )
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--n_solutions", type=int, default=20)
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--output_dir", type=str, default="experiment_5")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.quick:
        args.N = 200
        args.n_seeds = 3
        args.n_solutions = 10
        print("⚠️  Modo rápido activado\n")
    
    alphas = [float(a) for a in args.alphas.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("Experimento 5: Latent Projection Graph Model (LPGM)")
    print("Modelo: x~N(0,I2), h=||x||, P=sigmoid(α - |h_u - h_v|)")
    print(f"N={args.N}, seeds={args.n_seeds}, soluciones/grafo={args.n_solutions}")
    print(f"α: {alphas}")
    print("="*70)
    
    start = time.time()
    all_metrics = []
    for alpha in alphas:
        metrics = run_experiment_alpha(
            alpha, n_seeds=args.n_seeds, n_solutions=args.n_solutions,
            N=args.N, verbose=args.verbose
        )
        if metrics:
            all_metrics.append(metrics)
            # Visualizar para un ejemplo (primer α, primer seed)
            if alpha == alphas[0]:
                visualize_solutions(alpha, seed=0, n_solutions=5, output_dir=args.output_dir)
    
    elapsed = time.time() - start
    
    # Guardar métricas en JSON y texto
    with open(f"{args.output_dir}/metrics.json", 'w') as f:
        # Convertir a tipos nativos
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        metrics_serializable = []
        for m in all_metrics:
            m_copy = {k: convert(v) for k, v in m.items() if k != 'per_seed'}
            m_copy['per_seed'] = [{k: convert(v) for k, v in sd.items()} for sd in m['per_seed']]
            metrics_serializable.append(m_copy)
        json.dump(metrics_serializable, f, indent=2)
    
    with open(f"{args.output_dir}/metrics.txt", 'w') as f:
        f.write("# alpha\tmean_dist\teff_dim\tcorr_mean\tcorr_std\tstress\n")
        for m in all_metrics:
            f.write(f"{m['alpha']:.3f}\t{m['mean_solution_distance']:.6f}\t"
                    f"{m['effective_dimension']:.2f}\t{m['correlation_mean']:.6f}\t"
                    f"{m['correlation_std']:.6f}\t{m['mean_stress']:.6f}\n")
    
    plot_results(all_metrics, args.output_dir)
    
    print("\n"+"="*70)
    print("RESULTADOS")
    print("="*70)
    print(f"{'α':>8} {'Degeneración':>12} {'Dimensión':>10} {'Correlación radial':>18} {'Stress':>10}")
    for m in all_metrics:
        print(f"{m['alpha']:>8.2f} {m['mean_solution_distance']:>12.6f} "
              f"{m['effective_dimension']:>10.1f} {m['correlation_mean']:>12.4f} ±{m['correlation_std']:>6.4f} "
              f"{m['mean_stress']:>10.4f}")
    print("="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"Archivos guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()