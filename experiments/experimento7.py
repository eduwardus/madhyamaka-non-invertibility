# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:04:43 2026

@author: eggra
"""

#!/usr/bin/env python3
"""
Experimento 7: Universalidad de la ley información–degeneración.
Modelo: x ~ N(0, I_d) en R^d, h = ||x||, P_uv = sigmoid(α - |h_u - h_v|).
Se recuperan embeddings 2D mediante MDS y se analiza D (dimensión efectiva) e I (correlación radial).
Se calcula U = I·log(D) para diferentes d.
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
# 1. MODELO DE PROYECCIÓN RADIAL CON DIMENSIÓN LATENTE VARIABLE
# =============================================================================

def sigmoid_stable(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def generate_graph_radial_projection(seed, alpha, N=500, d_latent=2):
    """Genera grafo según modelo de proyección radial."""
    np.random.seed(seed)
    X_true = np.random.randn(N, d_latent)
    h = np.linalg.norm(X_true, axis=1)
    dh = np.abs(h[:, None] - h[None, :])
    Z = alpha - dh
    P = sigmoid_stable(Z)
    np.fill_diagonal(P, 0)
    A = np.random.rand(N, N) < P
    A = np.triu(A, 1)
    A = A + A.T
    A_sparse = sp.csr_matrix(A)
    return A_sparse, X_true, h, P


# =============================================================================
# 2. RECUPERACIÓN MÚLTIPLE VÍA MDS (SIEMPRE A 2D)
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


def recover_multiple_embeddings(A_sparse, d_embed=2, n_solutions=20, verbose=False):
    dist_matrix = adjacency_to_distance_matrix(A_sparse)
    dist_matrix = dist_matrix / (np.max(dist_matrix) + 1e-10)
    solutions = []
    stresses = []
    for i in range(n_solutions):
        mds = MDS(n_components=d_embed, dissimilarity='precomputed',
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
# 3. MÉTRICAS DE DEGENERACIÓN
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


def correlation_with_radial(solutions, h_true):
    """Correlación entre distancias euclídeas de las soluciones y |h_i - h_j|."""
    n = solutions[0].shape[0]
    dh_true = np.abs(h_true[:, None] - h_true[None, :])
    triu = np.triu_indices(n, k=1)
    corrs = []
    for sol in solutions:
        D_sol = np.sum((sol[:, None, :] - sol[None, :, :])**2, axis=-1)
        d_sol_triu = D_sol[triu]
        dh_triu = dh_true[triu]
        try:
            r, _ = pearsonr(d_sol_triu, dh_triu)
            corrs.append(r)
        except:
            corrs.append(0.0)
    return np.mean(corrs), np.std(corrs)


# =============================================================================
# 4. EXPERIMENTO PARA UNA DIMENSIÓN LATENTE DADA
# =============================================================================

def run_experiment_dimension(d_latent, alphas, n_seeds=5, n_solutions=20, N=500, verbose=False):
    """Ejecuta el experimento completo para una dimensión latente d."""
    print(f"\n{'='*70}")
    print(f"Latent dimension d = {d_latent}")
    print(f"{'='*70}")

    all_metrics = []
    for alpha in alphas:
        print(f"\nα = {alpha:.1f}")
        all_solutions = []
        true_hs = []
        for seed in range(n_seeds):
            A_sparse, _, h_true, _ = generate_graph_radial_projection(
                seed, alpha, N=N, d_latent=d_latent
            )
            true_hs.append(h_true)
            sols, _ = recover_multiple_embeddings(
                A_sparse, d_embed=2, n_solutions=n_solutions, verbose=verbose
            )
            all_solutions.append(sols)
            if verbose:
                print(f"  Seed {seed}: {len(sols)} soluciones")

        # Métricas por semilla
        per_seed = []
        for s_idx in range(n_seeds):
            sols = all_solutions[s_idx]
            if len(sols) < 2:
                continue
            eff_dim, _ = effective_dimension_of_solutions(sols)
            corr_mean, corr_std = correlation_with_radial(sols, true_hs[s_idx])
            per_seed.append({
                'seed': s_idx,
                'effective_dimension': eff_dim,
                'radial_correlation': corr_mean,
                'radial_correlation_std': corr_std,
            })

        if not per_seed:
            continue

        # Agregados
        metrics = {
            'alpha': alpha,
            'd_latent': d_latent,
            'effective_dimension_mean': np.mean([p['effective_dimension'] for p in per_seed]),
            'effective_dimension_std': np.std([p['effective_dimension'] for p in per_seed]),
            'radial_correlation_mean': np.mean([p['radial_correlation'] for p in per_seed]),
            'radial_correlation_std': np.mean([p['radial_correlation_std'] for p in per_seed]),
        }
        U_vals = [p['radial_correlation'] * np.log(p['effective_dimension']) for p in per_seed]
        metrics['U_mean'] = np.mean(U_vals)
        metrics['U_std'] = np.std(U_vals)

        print(f"  D = {metrics['effective_dimension_mean']:.2f} ± {metrics['effective_dimension_std']:.2f}")
        print(f"  I = {metrics['radial_correlation_mean']:.4f} ± {metrics['radial_correlation_std']:.4f}")
        print(f"  U = {metrics['U_mean']:.4f} ± {metrics['U_std']:.4f}")

        all_metrics.append(metrics)

    return all_metrics


def plot_comparison(metrics_by_d, output_dir="experiment_7"):
    """Genera gráficos comparativos para diferentes d."""
    plt.figure(figsize=(15, 10))

    # Subplot 1: Dimensión efectiva
    plt.subplot(2,3,1)
    for d, metrics in metrics_by_d.items():
        alphas = [m['alpha'] for m in metrics]
        D = [m['effective_dimension_mean'] for m in metrics]
        plt.plot(alphas, D, 'o-', label=f'd={d}')
    plt.xlabel('α')
    plt.ylabel('Dimensión efectiva')
    plt.title('Degeneración (D)')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Correlación radial
    plt.subplot(2,3,2)
    for d, metrics in metrics_by_d.items():
        alphas = [m['alpha'] for m in metrics]
        I = [m['radial_correlation_mean'] for m in metrics]
        plt.plot(alphas, I, 'o-', label=f'd={d}')
    plt.xlabel('α')
    plt.ylabel('Correlación radial (I)')
    plt.title('Información')
    plt.legend()
    plt.grid(True)

    # Subplot 3: U = I·log(D)
    plt.subplot(2,3,3)
    for d, metrics in metrics_by_d.items():
        alphas = [m['alpha'] for m in metrics]
        U = [m['U_mean'] for m in metrics]
        plt.plot(alphas, U, 'o-', label=f'd={d}')
    plt.xlabel('α')
    plt.ylabel('U = I·log(D)')
    plt.title('Compromiso información–degeneración')
    plt.legend()
    plt.grid(True)

    # Subplot 4: U vs α (detalle)
    plt.subplot(2,3,4)
    for d, metrics in metrics_by_d.items():
        alphas = [m['alpha'] for m in metrics]
        U = [m['U_mean'] for m in metrics]
        plt.plot(alphas, U, 'o-', label=f'd={d}')
    plt.xlabel('α')
    plt.ylabel('U')
    plt.title('U vs α (detalle)')
    plt.legend()
    plt.grid(True)

    # Subplot 5: U vs d (en α fijo, ej. α=2.0)
    plt.subplot(2,3,5)
    alpha_fixed = 2.0
    d_vals = []
    U_vals = []
    for d, metrics in metrics_by_d.items():
        for m in metrics:
            if abs(m['alpha'] - alpha_fixed) < 0.1:
                d_vals.append(d)
                U_vals.append(m['U_mean'])
                break
    if d_vals:
        plt.plot(d_vals, U_vals, 'o-', color='red')
        plt.xlabel('dimensión latente')
        plt.ylabel(f'U en α={alpha_fixed}')
        plt.title('Dependencia de U con d')
    plt.grid(True)

    # Subplot 6: D vs d (en α=2.0)
    plt.subplot(2,3,6)
    d_vals_D = []
    D_vals = []
    for d, metrics in metrics_by_d.items():
        for m in metrics:
            if abs(m['alpha'] - alpha_fixed) < 0.1:
                d_vals_D.append(d)
                D_vals.append(m['effective_dimension_mean'])
                break
    if d_vals_D:
        plt.plot(d_vals_D, D_vals, 'o-', color='purple')
        plt.xlabel('dimensión latente')
        plt.ylabel(f'Dimensión efectiva en α={alpha_fixed}')
        plt.title('Degeneración vs d')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_d.png", dpi=150)
    plt.close()
    print(f"  Guardada figura: {output_dir}/comparison_d.png")


def main():
    parser = argparse.ArgumentParser(
        description="Experimento 7: Universalidad de la ley información–degeneración"
    )
    parser.add_argument("--N", type=int, default=500, help="Número de nodos")
    parser.add_argument("--n_seeds", type=int, default=5, help="Semillas por α")
    parser.add_argument("--n_solutions", type=int, default=20, help="Soluciones por grafo")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--dims", type=str, default="3,4,5", help="Dimensiones latentes a probar")
    parser.add_argument("--output_dir", type=str, default="experiment_7")
    parser.add_argument("--quick", action="store_true", help="Modo rápido (N=200, seeds=3, solutions=10)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.N = 200
        args.n_seeds = 3
        args.n_solutions = 10
        print("⚠️  Modo rápido activado\n")

    alphas = [float(a) for a in args.alphas.split(',')]
    dims = [int(d) for d in args.dims.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Experimento 7: Universalidad de la ley información–degeneración")
    print(f"Modelo: proyección radial (LPGM) con d_latente variable")
    print(f"N={args.N}, seeds={args.n_seeds}, soluciones/grafo={args.n_solutions}")
    print(f"α: {alphas}")
    print(f"d_latent: {dims}")
    print("="*70)

    total_start = time.time()
    metrics_by_d = {}
    for d in dims:
        metrics = run_experiment_dimension(
            d_latent=d,
            alphas=alphas,
            n_seeds=args.n_seeds,
            n_solutions=args.n_solutions,
            N=args.N,
            verbose=args.verbose
        )
        metrics_by_d[d] = metrics

        # Guardar resultados individuales
        with open(f"{args.output_dir}/metrics_d{d}.json", 'w') as f:
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
            for m in metrics:
                m_copy = {k: convert(v) for k, v in m.items()}
                metrics_serializable.append(m_copy)
            json.dump(metrics_serializable, f, indent=2)

        with open(f"{args.output_dir}/metrics_d{d}.txt", 'w') as f:
            f.write("# alpha\tD_mean\tD_std\tI_mean\tI_std\tU_mean\tU_std\n")
            for m in metrics:
                f.write(f"{m['alpha']:.3f}\t{m['effective_dimension_mean']:.2f}\t{m['effective_dimension_std']:.2f}\t"
                        f"{m['radial_correlation_mean']:.4f}\t{m['radial_correlation_std']:.4f}\t"
                        f"{m['U_mean']:.4f}\t{m['U_std']:.4f}\n")

    total_elapsed = time.time() - total_start

    plot_comparison(metrics_by_d, args.output_dir)

    print("\n"+"="*70)
    print("RESUMEN")
    print("="*70)
    for d, metrics in metrics_by_d.items():
        print(f"\nd = {d}")
        print(f"{'α':>8} {'D (efectiva)':>12} {'I (correlación)':>16} {'U = I·log(D)':>16}")
        for m in metrics:
            print(f"{m['alpha']:>8.2f} {m['effective_dimension_mean']:>12.2f} "
                  f"{m['radial_correlation_mean']:>16.4f} {m['U_mean']:>16.4f}")
    print("="*70)
    print(f"Tiempo total: {total_elapsed:.1f}s")
    print(f"Archivos guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()