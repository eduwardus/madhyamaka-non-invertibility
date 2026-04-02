#!/usr/bin/env python3
"""
Experimento 9 (corregido, versión final): Dimensión intrínseca del espacio de soluciones.
- Alineación Procrustes de todas las soluciones a una referencia.
- Extracción de características invariantes: distancias por pares (triángulo superior).
- PCA sobre las matrices de distancias (como vectores) para estimar dimensión.
- Umbral de autovalores: > 1% del primer autovalor.
Objetivo: Obtener una estimación robusta de la dimensión geométrica del espacio de soluciones.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import argparse
import time
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. MODELO DE PROYECCIÓN RADIAL (LPGM)
# =============================================================================

def sigmoid_stable(z):
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def generate_graph_radial_projection(seed, alpha, N=500, d_latent=2):
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
    return sp.csr_matrix(A)


# =============================================================================
# 2. RECUPERACIÓN MÚLTIPLE VÍA MDS
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


def recover_multiple_embeddings(A_sparse, d_embed=2, n_solutions=50, verbose=False):
    dist_matrix = adjacency_to_distance_matrix(A_sparse)
    dist_matrix = dist_matrix / (np.max(dist_matrix) + 1e-10)
    solutions = []
    for i in range(n_solutions):
        mds = MDS(n_components=d_embed, dissimilarity='precomputed',
                  random_state=i, n_init=1, max_iter=500,
                  normalized_stress='auto', verbose=0)
        try:
            X = mds.fit_transform(dist_matrix)
            solutions.append(X)
        except Exception as e:
            if verbose:
                print(f"    Solución {i} falló: {e}")
            continue
    if verbose:
        print(f"    Generadas {len(solutions)} soluciones válidas")
    return solutions


# =============================================================================
# 3. ALINEACIÓN PROCRUSTES Y EXTRACCIÓN DE CARACTERÍSTICAS INVARIANTES
# =============================================================================

def align_solutions(solutions):
    """
    Alinea todas las soluciones a la primera usando Procrustes.
    Retorna lista de matrices alineadas.
    """
    if not solutions:
        return []
    ref = solutions[0]
    aligned = [ref]  # la referencia se queda igual
    for sol in solutions[1:]:
        _, sol_aligned, _ = procrustes(ref, sol)
        aligned.append(sol_aligned)
    return aligned


def solution_to_feature_vector(X):
    """
    Convierte una matriz de coordenadas X (N x d_embed) en un vector de características invariantes:
    distancias por pares (triángulo superior).
    """
    # pdist devuelve un array de longitud N*(N-1)/2
    return pdist(X)


# =============================================================================
# 4. ESTIMACIÓN DE DIMENSIÓN INTRÍNSECA
# =============================================================================

def estimate_intrinsic_dimension(feature_vectors, eigenvalue_threshold=0.01):
    """
    feature_vectors: lista de arrays (cada uno es un vector de características)
    Calcula PCA y retorna dimensión estimada como número de componentes
    cuyo autovalor es > threshold * primer autovalor.
    """
    if len(feature_vectors) < 2:
        return 1, None
    X = np.vstack(feature_vectors)
    pca = PCA()
    pca.fit(X)
    ev = pca.explained_variance_
    threshold = eigenvalue_threshold * ev[0]
    dim = np.sum(ev > threshold)
    return dim, ev


# =============================================================================
# 5. EXPERIMENTO PRINCIPAL (con correcciones)
# =============================================================================

def run_experiment(alpha, d_embed, n_seeds=5, n_solutions=50, N=300, output_dir=None, verbose=False):
    """
    Para cada semilla: genera grafo, recupera múltiples embeddings,
    los alinea, extrae características invariantes, acumula todos los vectores.
    Finalmente estima dimensión intrínseca.
    """
    all_features = []
    for seed in range(n_seeds):
        if verbose:
            print(f"      Seed {seed+1}/{n_seeds}")
        A = generate_graph_radial_projection(seed, alpha, N=N, d_latent=2)
        sols = recover_multiple_embeddings(A, d_embed=d_embed, n_solutions=n_solutions, verbose=verbose)
        if len(sols) < 2:
            continue
        # Alinear soluciones a la primera (dentro de esta semilla)
        aligned_sols = align_solutions(sols)
        for sol in aligned_sols:
            feat = solution_to_feature_vector(sol)
            all_features.append(feat)
    if len(all_features) < 2:
        return None, None
    dim, ev = estimate_intrinsic_dimension(all_features, eigenvalue_threshold=0.01)
    # Guardar gráfico de autovalores si se solicita
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(8,5))
        plt.plot(range(1, len(ev)+1), ev, 'o-')
        plt.axhline(y=0.01*ev[0], color='r', linestyle='--', label='threshold (1%)')
        plt.xlabel('Componente principal')
        plt.ylabel('Autovalor')
        plt.title(f'Espectro de autovalores (α={alpha}, d_embed={d_embed})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/eigenvalues_alpha{alpha}_d{d_embed}.png", dpi=150)
        plt.close()
    return dim, ev


def main():
    parser = argparse.ArgumentParser(
        description="Experimento 9 (corregido): Dimensión intrínseca con invariantes estructurales"
    )
    parser.add_argument("--N", type=int, default=300, help="Número de nodos (menor para eficiencia)")
    parser.add_argument("--n_seeds", type=int, default=5, help="Semillas por α")
    parser.add_argument("--n_solutions", type=int, default=50, help="Soluciones por grafo")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
    parser.add_argument("--embed_dims", type=str, default="2,3,5,10")
    parser.add_argument("--output_dir", type=str, default="experiment_9_final")
    parser.add_argument("--quick", action="store_true", help="Modo rápido: N=200, seeds=3, solutions=30")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.N = 200
        args.n_seeds = 3
        args.n_solutions = 30
        print("⚠️  Modo rápido activado\n")

    alphas = [float(a) for a in args.alphas.split(',')]
    embed_dims = [int(d) for d in args.embed_dims.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("Experimento 9 (corregido): Dimensión intrínseca con invariantes estructurales")
    print(f"N={args.N}, seeds={args.n_seeds}, soluciones/grafo={args.n_solutions}")
    print(f"α: {alphas}")
    print(f"Dimensiones de embedding: {embed_dims}")
    print("="*70)

    results = []
    start = time.time()

    for d_embed in embed_dims:
        print(f"\n--- d_embed = {d_embed} ---")
        for alpha in alphas:
            print(f"  α={alpha:.1f}")
            dim, _ = run_experiment(alpha, d_embed,
                                    n_seeds=args.n_seeds,
                                    n_solutions=args.n_solutions,
                                    N=args.N,
                                    output_dir=args.output_dir,
                                    verbose=args.verbose)
            if dim is not None:
                print(f"    Dimensión intrínseca (threshold 1%) = {dim}")
                results.append({
                    'alpha': alpha,
                    'd_embed': d_embed,
                    'intrinsic_dim': dim
                })
            else:
                print("    No se obtuvieron suficientes muestras")

    elapsed = time.time() - start

    # Guardar resultados en CSV y JSON
    with open(f"{args.output_dir}/intrinsic_dim.csv", 'w') as f:
        f.write("alpha,d_embed,intrinsic_dim\n")
        for r in results:
            f.write(f"{r['alpha']:.3f},{r['d_embed']},{r['intrinsic_dim']}\n")
    with open(f"{args.output_dir}/intrinsic_dim.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Gráficos
    # 1. Dimensión vs α para diferentes d_embed
    plt.figure(figsize=(10,6))
    for d_embed in embed_dims:
        x = [r['alpha'] for r in results if r['d_embed'] == d_embed]
        y = [r['intrinsic_dim'] for r in results if r['d_embed'] == d_embed]
        plt.plot(x, y, 'o-', label=f'd_embed={d_embed}')
    plt.xlabel('α')
    plt.ylabel('Dimensión intrínseca')
    plt.title('Dimensión intrínseca del espacio de soluciones')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/intrinsic_dim_vs_alpha.png", dpi=150)
    plt.close()

    # 2. Dimensión vs d_embed para α=2.0
    alpha_fixed = 2.0
    x_vals = []
    y_vals = []
    for d_embed in embed_dims:
        for r in results:
            if abs(r['alpha'] - alpha_fixed) < 0.1 and r['d_embed'] == d_embed:
                x_vals.append(d_embed)
                y_vals.append(r['intrinsic_dim'])
                break
    if x_vals:
        plt.figure(figsize=(8,5))
        plt.plot(x_vals, y_vals, 'o-', color='red')
        plt.xlabel('Dimensión de embedding')
        plt.ylabel('Dimensión intrínseca')
        plt.title(f'Dimensión intrínseca vs d_embed (α={alpha_fixed})')
        plt.grid(True)
        plt.savefig(f"{args.output_dir}/intrinsic_dim_vs_dembed.png", dpi=150)
        plt.close()

    # 3. Heatmap
    if len(embed_dims) > 1 and len(alphas) > 1:
        mat = np.zeros((len(alphas), len(embed_dims)))
        for i, alpha in enumerate(alphas):
            for j, d_embed in enumerate(embed_dims):
                for r in results:
                    if r['alpha'] == alpha and r['d_embed'] == d_embed:
                        mat[i, j] = r['intrinsic_dim']
                        break
        plt.figure(figsize=(8,6))
        plt.imshow(mat, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(label='Dimensión intrínseca')
        plt.xticks(np.arange(len(embed_dims)), embed_dims)
        plt.yticks(np.arange(len(alphas)), alphas)
        plt.xlabel('d_embed')
        plt.ylabel('α')
        plt.title('Heatmap de dimensión intrínseca')
        plt.savefig(f"{args.output_dir}/intrinsic_dim_heatmap.png", dpi=150)
        plt.close()

    print("\n"+"="*70)
    print("RESUMEN")
    print("="*70)
    print(f"{'α':>8} {'d_embed':>8} {'Dimensión intrínseca':>20}")
    for r in results:
        print(f"{r['alpha']:>8.2f} {r['d_embed']:>8d} {r['intrinsic_dim']:>20d}")
    print("="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"Archivos guardados en: {args.output_dir}")


if __name__ == "__main__":
    main()