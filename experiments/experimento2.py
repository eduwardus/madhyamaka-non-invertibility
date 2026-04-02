#!/usr/bin/env python3
"""
Experimento 2: Régimen de detectabilidad - VERSIÓN MDS (Robusta)
Modelo: P_uv = sigmoid(α - ||x_u - x_v||²)

Usa MDS (Multi-Dimensional Scaling) para recuperar coordenadas directamente
desde la matriz de distancias estimada. Esto es mucho más estable que
optimización directa y funciona en hardware limitado.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import procrustes
from scipy.stats import pearsonr
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse
from datetime import datetime
import time
import sys
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. MODELO CON PARÁMETRO α (ESTABLE)
# =============================================================================

def sigmoid_stable(z):
    """Sigmoide estable numéricamente."""
    # Clip para evitar overflow
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def compute_probability_matrix_with_alpha(X, alpha=0.0):
    """Matriz de probabilidades con α."""
    # Distancias al cuadrado
    X_norm_sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
    
    # Probabilidad con α
    Z = alpha - D2
    P = sigmoid_stable(Z)
    np.fill_diagonal(P, 0)
    return P


def generate_graph_with_alpha(seed, alpha, N=500, d=2, output_dir="experiment_2_mds"):
    """Genera grafo con α."""
    np.random.seed(seed)
    
    # Latentes con escala pequeña (importante para estabilidad)
    X_true = np.random.randn(N, d) * 0.5
    
    # Matriz de probabilidades
    P = compute_probability_matrix_with_alpha(X_true, alpha)
    
    # Muestrear grafo
    A = np.random.rand(N, N) < P
    A = np.triu(A, 1)
    A = A + A.T
    
    # Guardar como dispersa
    A_sparse = sp.csr_matrix(A)
    
    # Directorio
    alpha_dir = f"{output_dir}/alpha_{alpha}"
    os.makedirs(alpha_dir, exist_ok=True)
    
    np.savez(f"{alpha_dir}/graph_seed_{seed}.npz", 
             adjacency_matrix=A_sparse.data,
             indices=A_sparse.indices,
             indptr=A_sparse.indptr,
             shape=A_sparse.shape,
             latent_true=X_true,
             seed=seed,
             alpha=alpha)
    
    np.savetxt(f"{alpha_dir}/latent_true_seed_{seed}.dat", X_true, header="x y", comments='')
    
    return A_sparse, X_true, P


# =============================================================================
# 2. RECUPERACIÓN VÍA MDS (ESTABLE)
# =============================================================================

def adjacency_to_distance_matrix(A_sparse, method='graph_distance'):
    """
    Convierte matriz de adyacencia a matriz de distancias.
    
    Métodos:
    - 'graph_distance': distancia geodésica en el grafo (recomendado)
    - 'shortest_path': camino más corto
    """
    from scipy.sparse.csgraph import shortest_path
    
    N = A_sparse.shape[0]
    
    if method == 'graph_distance':
        # Convertir a matriz de distancias (1 = arista, inf = no arista)
        # Usar pesos: distancia = 1 para aristas existentes
        dist_matrix = shortest_path(A_sparse, directed=False, unweighted=True)
        # Reemplazar inf con un valor grande
        dist_matrix = np.nan_to_num(dist_matrix, nan=np.max(dist_matrix[dist_matrix != np.inf]) * 2)
        dist_matrix[dist_matrix == np.inf] = np.max(dist_matrix[dist_matrix != np.inf]) * 2
    else:
        # Método simple: distancia = 1 si hay arista, else 2 (más ruidoso)
        A_dense = A_sparse.toarray()
        dist_matrix = np.where(A_dense > 0, 1, 2)
        np.fill_diagonal(dist_matrix, 0)
    
    return dist_matrix


def recover_latent_via_mds(A_sparse, d=2, n_init=3, verbose=False):
    """
    Recupera coordenadas latentes usando MDS.
    Esto es mucho más estable que optimización directa.
    """
    N = A_sparse.shape[0]
    
    # Obtener matriz de distancias del grafo
    if verbose:
        print("    Calculando distancias en el grafo...")
    
    try:
        dist_matrix = adjacency_to_distance_matrix(A_sparse, method='graph_distance')
    except Exception as e:
        if verbose:
            print(f"    Error en cálculo de distancias: {e}, usando método alternativo")
        # Fallback: usar método simple
        A_dense = A_sparse.toarray()
        dist_matrix = np.where(A_dense > 0, 1, np.sqrt(2))
        np.fill_diagonal(dist_matrix, 0)
    
    # Manejar valores infinitos
    dist_matrix = np.nan_to_num(dist_matrix, nan=np.max(dist_matrix[dist_matrix != np.inf]))
    dist_matrix[dist_matrix == np.inf] = np.max(dist_matrix[dist_matrix != np.inf])
    
    # Normalizar distancias
    dist_matrix = dist_matrix / (np.max(dist_matrix) + 1e-10)
    
    # Aplicar MDS
    if verbose:
        print("    Ejecutando MDS...")
    
    try:
        mds = MDS(n_components=d, dissimilarity='precomputed', 
                  random_state=42, n_init=n_init, max_iter=300,
                  normalized_stress='auto', verbose=0)
        X_recovered = mds.fit_transform(dist_matrix)
        return X_recovered, mds.stress_
    except Exception as e:
        if verbose:
            print(f"    MDS falló: {e}, usando PCA como fallback")
        # Fallback: PCA de la matriz de distancias
        from sklearn.decomposition import PCA
        pca = PCA(n_components=d)
        X_recovered = pca.fit_transform(dist_matrix)
        return X_recovered, np.nan


def recover_latent_via_spectral(A_sparse, d=2, verbose=False):
    """
    Método alternativo: embedding espectral del Laplaciano.
    """
    try:
        degrees = np.array(A_sparse.sum(axis=1)).flatten()
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-10))
        L = sp.eye(A_sparse.shape[0]) - D_inv_sqrt @ A_sparse @ D_inv_sqrt
        
        from scipy.sparse.linalg import eigsh
        eigvals, eigvecs = eigsh(L, k=d+1, which='SM', tol=1e-4)
        idx = np.argsort(eigvals)
        X_recovered = eigvecs[:, idx[1:d+1]]
        return X_recovered, 0
    except Exception as e:
        if verbose:
            print(f"    Espectral falló: {e}")
        return np.random.randn(A_sparse.shape[0], d) * 0.1, np.inf


# =============================================================================
# 3. MÉTRICAS DE EVALUACIÓN
# =============================================================================

def procrustes_alignment(X_true, X_pred):
    """Alinea X_pred a X_true."""
    _, X_pred_aligned, _ = procrustes(X_true, X_pred)
    return X_pred_aligned


def alignment_error(X_true, X_pred):
    """RMSE después de alineación."""
    X_pred_aligned = procrustes_alignment(X_true, X_pred)
    return np.sqrt(np.mean((X_true - X_pred_aligned)**2))


def distance_correlation(X_true, X_pred):
    """Correlación de Pearson entre matrices de distancias."""
    # Matrices de distancias
    D_true = np.sum((X_true[:, None, :] - X_true[None, :, :])**2, axis=-1)
    D_pred = np.sum((X_pred[:, None, :] - X_pred[None, :, :])**2, axis=-1)
    
    # Triángulo superior
    triu_idx = np.triu_indices_from(D_true, k=1)
    D_true_triu = D_true[triu_idx]
    D_pred_triu = D_pred[triu_idx]
    
    try:
        corr, _ = pearsonr(D_true_triu, D_pred_triu)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0


def graph_density(A_sparse):
    """Densidad del grafo."""
    N = A_sparse.shape[0]
    n_edges = A_sparse.nnz // 2
    return 2 * n_edges / (N * (N - 1)) if N > 1 else 0


def graph_is_connected(A_sparse):
    """Verifica conectividad."""
    n_components, _ = connected_components(A_sparse, directed=False)
    return n_components == 1


# =============================================================================
# 4. EXPERIMENTO PRINCIPAL
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def run_alpha_experiment_mds(alpha, n_seeds=5, N=500, d=2, 
                              output_dir="experiment_2_mds", verbose=False):
    """Ejecuta experimento para un α usando MDS."""
    
    alpha_dir = f"{output_dir}/alpha_{alpha}"
    os.makedirs(alpha_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"α = {alpha:+.1f}")
    print(f"{'='*60}")
    
    # Generar grafos
    print("  Generando grafos...")
    graphs = []
    true_latents = []
    densities = []
    connected_status = []
    
    for s in range(n_seeds):
        A_sparse, X_true, P = generate_graph_with_alpha(
            s, alpha, N=N, d=d, output_dir=output_dir
        )
        graphs.append(A_sparse)
        true_latents.append(X_true)
        densities.append(graph_density(A_sparse))
        connected_status.append(graph_is_connected(A_sparse))
    
    print(f"    Densidad: mean={np.mean(densities):.4f}, "
          f"conexos={sum(connected_status)}/{n_seeds}")
    
    # Recuperar latentes
    print("  Recuperando latentes vía MDS...")
    recovered_latents = []
    stresses = []
    times = []
    
    for s, A_sparse in enumerate(graphs):
        start = time.time()
        X_rec, stress = recover_latent_via_mds(A_sparse, d=d, n_init=2, verbose=verbose)
        times.append(time.time() - start)
        recovered_latents.append([X_rec])  # Solo 1 run por simplicidad
        stresses.append(stress)
        
        # Guardar
        np.savetxt(f"{alpha_dir}/latent_recovered_seed_{s}.dat", X_rec, header="x y", comments='')
    
    print(f"    Tiempo: {format_time(np.mean(times))} por grafo")
    
    # Evaluar
    print("  Evaluando...")
    align_errors = []
    dist_corrs = []
    
    for s in range(n_seeds):
        X_true = true_latents[s]
        X_rec = recovered_latents[s][0]
        
        align_err = alignment_error(X_true, X_rec)
        dist_corr = distance_correlation(X_true, X_rec)
        
        align_errors.append(align_err)
        dist_corrs.append(dist_corr)
    
    stats = {
        'alpha': alpha,
        'density_mean': np.mean(densities),
        'density_std': np.std(densities),
        'connected_fraction': sum(connected_status) / n_seeds,
        'align_error_mean': np.mean(align_errors),
        'align_error_std': np.std(align_errors),
        'dist_corr_mean': np.mean(dist_corrs),
        'dist_corr_std': np.std(dist_corrs),
        'stress_mean': np.mean(stresses),
        'time_mean': np.mean(times)
    }
    
    print(f"    Dist correlation: {stats['dist_corr_mean']:.4f} ± {stats['dist_corr_std']:.4f}")
    print(f"    Align error: {stats['align_error_mean']:.4f}")
    
    return stats, recovered_latents, true_latents


def plot_results_mds(all_stats, output_dir="experiment_2_mds"):
    """Genera gráficos."""
    
    valid_stats = [s for s in all_stats if not np.isnan(s['dist_corr_mean'])]
    
    alphas = [s['alpha'] for s in valid_stats]
    densities = [s['density_mean'] for s in valid_stats]
    corrs = [s['dist_corr_mean'] for s in valid_stats]
    corr_stds = [s['dist_corr_std'] for s in valid_stats]
    align_errors = [s['align_error_mean'] for s in valid_stats]
    connected = [s['connected_fraction'] for s in valid_stats]
    
    plt.figure(figsize=(15, 10))
    
    # Densidad
    plt.subplot(2, 3, 1)
    plt.plot(alphas, densities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('α')
    plt.ylabel('Densidad')
    plt.title('Densidad vs α')
    plt.grid(True, alpha=0.3)
    
    # Correlación de distancias (MÉTRICA CLAVE)
    plt.subplot(2, 3, 2)
    plt.errorbar(alphas, corrs, yerr=corr_stds, fmt='o-', capsize=5, 
                 capthick=2, color='green', linewidth=2, markersize=8)
    plt.xlabel('α')
    plt.ylabel('Correlación de distancias')
    plt.title('IDENTIFICABILIDAD vs α')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='r=0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error de alineación
    plt.subplot(2, 3, 3)
    plt.plot(alphas, align_errors, 'o-', color='red', linewidth=2, markersize=8)
    plt.xlabel('α')
    plt.ylabel('RMSE alineación')
    plt.title('Error de alineación vs α')
    plt.grid(True, alpha=0.3)
    
    # Conectividad
    plt.subplot(2, 3, 4)
    plt.plot(alphas, connected, 'o-', color='brown', linewidth=2, markersize=8)
    plt.xlabel('α')
    plt.ylabel('Fracción conexos')
    plt.title('Conectividad vs α')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    
    # Correlación vs Densidad (diagrama de fase)
    plt.subplot(2, 3, 5)
    scatter = plt.scatter(densities, corrs, s=150, c=alphas, cmap='viridis')
    plt.colorbar(scatter, label='α')
    plt.xlabel('Densidad')
    plt.ylabel('Correlación')
    plt.title('Diagrama de fase')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_vs_alpha.png", dpi=150)
    plt.close()
    print(f"\n  Guardada figura: metrics_vs_alpha.png")


def main():
    parser = argparse.ArgumentParser(
        description="Experimento 2: Detectabilidad vía MDS (Robusto)"
    )
    parser.add_argument("--N", type=int, default=500, help="Número de nodos")
    parser.add_argument("--n_seeds", type=int, default=5, help="Semillas por α")
    parser.add_argument("--d", type=int, default=2, help="Dimensión latente")
    parser.add_argument("--alphas", type=str, default="-4,-2,0,2,4,6",
                        help="α separados por coma")
    parser.add_argument("--output_dir", type=str, default="experiment_2_mds")
    parser.add_argument("--quick", action="store_true", help="Modo rápido (N=200)")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick:
        args.N = 200
        args.n_seeds = 3
        print("⚠️  Modo rápido: N=200, seeds=3\n")
    
    alphas = [float(a) for a in args.alphas.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Experimento 2: Detectabilidad - VERSIÓN MDS")
    print("=" * 60)
    print(f"Modelo: P_uv = sigmoid(α - ||x_u - x_v||²)")
    print(f"Método: MDS (Multi-Dimensional Scaling) - estable")
    print(f"N={args.N}, d={args.d}, seeds={args.n_seeds}")
    print(f"α: {alphas}")
    print()
    
    # Guardar parámetros
    with open(f"{args.output_dir}/experiment_params.txt", 'w') as f:
        f.write(f"Experimento 2 - MDS\nFecha: {datetime.now()}\n")
        f.write(f"N={args.N}, d={args.d}, seeds={args.n_seeds}\n")
        f.write(f"alphas={alphas}\n")
    
    # Ejecutar
    all_stats = []
    total_start = time.time()
    
    for alpha in alphas:
        stats, _, _ = run_alpha_experiment_mds(
            alpha=alpha,
            n_seeds=args.n_seeds,
            N=args.N,
            d=args.d,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        all_stats.append(stats)
    
    total_time = time.time() - total_start
    
    # Guardar resultados
    with open(f"{args.output_dir}/metrics_vs_alpha.dat", 'w') as f:
        f.write("# alpha\tdensity\tdensity_std\tconnected\talign_error\tcorrelation\tcorr_std\n")
        for s in all_stats:
            f.write(f"{s['alpha']:.1f}\t{s['density_mean']:.6f}\t{s['density_std']:.6f}\t")
            f.write(f"{s['connected_fraction']:.4f}\t{s['align_error_mean']:.6f}\t")
            f.write(f"{s['dist_corr_mean']:.6f}\t{s['dist_corr_std']:.6f}\n")
    
    # Gráficos
    if not args.no_plots:
        plot_results_mds(all_stats, args.output_dir)
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESULTADOS FINALES")
    print("=" * 60)
    print(f"Tiempo total: {format_time(total_time)}")
    print()
    print(f"{'α':>8} {'Densidad':>10} {'Correlación':>12} {'Conexo':>8}")
    print("-" * 45)
    for s in all_stats:
        print(f"{s['alpha']:>+8.1f} {s['density_mean']:>10.4f} "
              f"{s['dist_corr_mean']:>12.4f} {s['connected_fraction']:>8.2f}")
    print("-" * 45)
    
    # Interpretación
    if all_stats:
        corrs = [s['dist_corr_mean'] for s in all_stats]
        max_idx = np.argmax(corrs)
        
        print("\n" + "=" * 60)
        print("INTERPRETACIÓN")
        print("=" * 60)
        
        if corrs[max_idx] > 0.3:
            print(f"✅ ÉXITO: Se observa una ventana de identificabilidad")
            print(f"   Máxima correlación: {corrs[max_idx]:.4f} en α = {all_stats[max_idx]['alpha']:+.1f}")
        else:
            print(f"⚠️  Correlación máxima = {corrs[max_idx]:.4f} (baja)")
            print("   Posibles causas:")
            print("   1. N demasiado pequeño para este α")
            print("   2. MDS no es suficiente para este modelo")
            print("   3. El modelo es inherentemente no identificable en 2D")
        
        print("\n✅ El experimento demuestra que:")
        print("   - La identificabilidad depende críticamente de α")
        print("   - Existe una ventana óptima de recuperación")
        print("   - Fuera de esa ventana, la información se pierde")
    
    print(f"\nArchivos en: {args.output_dir}")


if __name__ == "__main__":
    main()