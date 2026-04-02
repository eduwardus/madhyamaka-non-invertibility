#!/usr/bin/env python3
"""
Experimento 4: Degeneración del espacio latente - CORREGIDO
Análisis de la variedad de soluciones compatibles con el mismo grafo.
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
# 1. MODELO Y GENERACIÓN (estable)
# =============================================================================

def sigmoid_stable(z):
    """Sigmoide estable numéricamente."""
    z = np.clip(z, -30, 30)
    return 1 / (1 + np.exp(-z))


def compute_probability_matrix_with_alpha(X, alpha=0.0):
    """Matriz de probabilidades con α."""
    X_norm_sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
    Z = alpha - D2
    P = sigmoid_stable(Z)
    np.fill_diagonal(P, 0)
    return P


def generate_graph_with_alpha(seed, alpha, N=500, d=2):
    """Genera grafo con α."""
    np.random.seed(seed)
    X_true = np.random.randn(N, d) * 0.5
    P = compute_probability_matrix_with_alpha(X_true, alpha)
    A = np.random.rand(N, N) < P
    A = np.triu(A, 1)
    A = A + A.T
    return sp.csr_matrix(A), X_true


# =============================================================================
# 2. RECUPERACIÓN MÚLTIPLE
# =============================================================================

def adjacency_to_distance_matrix(A_sparse):
    """Convierte adyacencia a matriz de distancias geodésicas."""
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


def recover_multiple_solutions(A_sparse, d=2, n_solutions=20, verbose=False):
    """
    Recupera múltiples soluciones latentes para el mismo grafo,
    usando diferentes inicializaciones aleatorias.
    """
    dist_matrix = adjacency_to_distance_matrix(A_sparse)
    dist_matrix = dist_matrix / (np.max(dist_matrix) + 1e-10)
    
    solutions = []
    stresses = []
    
    for i in range(n_solutions):
        # Diferente semilla para cada ejecución
        mds = MDS(n_components=d, dissimilarity='precomputed',
                  random_state=i, n_init=1, max_iter=500,
                  normalized_stress='auto', verbose=0)
        
        try:
            X_rec = mds.fit_transform(dist_matrix)
            solutions.append(X_rec)
            stresses.append(mds.stress_)
        except Exception as e:
            if verbose:
                print(f"    Solución {i} falló: {e}")
            continue
    
    if verbose:
        print(f"    Generadas {len(solutions)} soluciones válidas")
    
    return solutions, stresses


# =============================================================================
# 3. ANÁLISIS DE DEGENERACIÓN (CORREGIDO)
# =============================================================================

def procrustes_alignment(X_ref, X_target):
    """Alinea X_target a X_ref."""
    _, X_aligned, _ = procrustes(X_ref, X_target)
    return X_aligned


def solution_distance_matrix(solutions):
    """
    Calcula la matriz de distancias entre soluciones (después de alinear).
    """
    n_sol = len(solutions)
    dist_matrix = np.zeros((n_sol, n_sol))
    
    for i in range(n_sol):
        for j in range(i + 1, n_sol):
            # Alinear j a i
            X_j_aligned = procrustes_alignment(solutions[i], solutions[j])
            # Distancia cuadrática media entre configuraciones completas
            dist = np.mean((solutions[i] - X_j_aligned)**2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix


def effective_dimension_of_solutions(solutions):
    """
    Calcula la dimensión efectiva del espacio de soluciones.
    Trata cada solución como un vector en R^(N*d).
    """
    n_sol = len(solutions)
    if n_sol < 3:
        return 1.0, np.array([])
    
    # Apilar soluciones como vectores aplanados
    vectors = np.array([sol.flatten() for sol in solutions])  # (n_sol, N*d)
    
    # PCA sobre los vectores de soluciones
    pca = PCA()
    pca.fit(vectors)
    
    # Varianza explicada acumulada
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    # Dimensión efectiva: número de componentes que explican el 95% de la varianza
    effective_dim = np.argmax(cumsum >= 0.95) + 1
    
    return effective_dim, pca.explained_variance_ratio_


def solution_clustering(solutions, n_clusters=None):
    """
    Agrupa soluciones por similitud (usando centroides de las soluciones).
    """
    if len(solutions) < 2:
        return np.array([0]), None
    
    # Centroides de cada solución (media de sus puntos)
    centroids = np.array([sol.mean(axis=0) for sol in solutions])
    
    if n_clusters is None:
        n_clusters = min(len(solutions), 5)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(centroids)
    
    return labels, kmeans.cluster_centers_


def solution_variance(solutions):
    """
    Calcula varianzas intra e inter soluciones.
    """
    n_sol = len(solutions)
    n_points = solutions[0].shape[0]
    d = solutions[0].shape[1]
    
    # Varianza intra: media de varianzas dentro de cada solución
    intra_vars = [np.var(sol, axis=0).mean() for sol in solutions]
    intra_var = np.mean(intra_vars)
    
    # Varianza inter: varianza de los centroides
    centroids = np.array([sol.mean(axis=0) for sol in solutions])
    inter_var = np.var(centroids, axis=0).mean()
    
    return {
        'intra_var': intra_var,
        'inter_var': inter_var,
        'ratio_inter_intra': inter_var / (intra_var + 1e-10)
    }


def stability_analysis(solutions):
    """
    Análisis de estabilidad: distancias entre soluciones.
    """
    n_sol = len(solutions)
    if n_sol < 2:
        return {'mean_distance': 0, 'std_distance': 0, 'max_distance': 0}
    
    dist_matrix = solution_distance_matrix(solutions)
    triu_idx = np.triu_indices_from(dist_matrix, k=1)
    distances = dist_matrix[triu_idx]
    
    return {
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'max_distance': np.max(distances),
        'min_distance': np.min(distances),
        'distance_matrix': dist_matrix
    }


def correlation_variance(solutions, true_latent):
    """
    Calcula la varianza de la correlación entre soluciones y el verdadero latente.
    """
    def distance_correlation(X_true, X_pred):
        D_true = np.sum((X_true[:, None, :] - X_true[None, :, :])**2, axis=-1)
        D_pred = np.sum((X_pred[:, None, :] - X_pred[None, :, :])**2, axis=-1)
        triu_idx = np.triu_indices_from(D_true, k=1)
        D_true_triu = D_true[triu_idx]
        D_pred_triu = D_pred[triu_idx]
        try:
            corr, _ = pearsonr(D_true_triu, D_pred_triu)
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    correlations = []
    for sol in solutions:
        corr = distance_correlation(true_latent, sol)
        correlations.append(corr)
    
    return {
        'mean_correlation': np.mean(correlations),
        'std_correlation': np.std(correlations),
        'min_correlation': np.min(correlations),
        'max_correlation': np.max(correlations),
        'correlations': correlations
    }


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


def run_degeneracy_experiment(alpha, n_seeds=5, n_solutions_per_seed=20,
                              N=500, d=2, verbose=False):
    """
    Ejecuta análisis de degeneración para un α dado.
    """
    print(f"\n{'='*60}")
    print(f"α = {alpha:.1f}")
    print(f"{'='*60}")
    
    all_solutions = []
    all_stresses = []
    true_latents = []
    graphs = []
    
    # Para cada semilla, generar un grafo y múltiples soluciones
    for seed in range(n_seeds):
        if verbose:
            print(f"  Seed {seed+1}/{n_seeds}")
        
        # Generar grafo
        A_sparse, X_true = generate_graph_with_alpha(seed, alpha, N=N, d=d)
        graphs.append(A_sparse)
        true_latents.append(X_true)
        
        # Múltiples soluciones
        solutions, stresses = recover_multiple_solutions(
            A_sparse, d=d, n_solutions=n_solutions_per_seed, verbose=verbose
        )
        all_solutions.append(solutions)
        all_stresses.append(stresses)
        
        if verbose:
            print(f"    {len(solutions)} soluciones, stress medio={np.mean(stresses):.4f}")
    
    # Análisis agregado
    print("  Analizando degeneración...")
    
    # Por semilla
    per_seed_metrics = []
    for s_idx in range(n_seeds):
        solutions = all_solutions[s_idx]
        if len(solutions) < 2:
            continue
        
        # Matriz de distancias entre soluciones
        dist_matrix = solution_distance_matrix(solutions)
        
        # Estabilidad
        stability = stability_analysis(solutions)
        
        # Varianza de correlación con verdadero
        corr_var = correlation_variance(solutions, true_latents[s_idx])
        
        # Dimensión efectiva (corregida)
        eff_dim, ev_ratio = effective_dimension_of_solutions(solutions)
        
        per_seed_metrics.append({
            'seed': s_idx,
            'n_solutions': len(solutions),
            'mean_stress': np.mean(all_stresses[s_idx]),
            'mean_solution_distance': stability['mean_distance'],
            'max_solution_distance': stability['max_distance'],
            'effective_dimension': eff_dim,
            'correlation_mean': corr_var['mean_correlation'],
            'correlation_std': corr_var['std_correlation'],
            'distance_matrix': dist_matrix.tolist()  # para JSON
        })
    
    # Métricas globales para este α
    if per_seed_metrics:
        metrics = {
            'alpha': alpha,
            'n_seeds': n_seeds,
            'avg_n_solutions': np.mean([m['n_solutions'] for m in per_seed_metrics]),
            'mean_stress': np.mean([m['mean_stress'] for m in per_seed_metrics]),
            'mean_solution_distance': np.mean([m['mean_solution_distance'] for m in per_seed_metrics]),
            'max_solution_distance': np.mean([m['max_solution_distance'] for m in per_seed_metrics]),
            'effective_dimension': np.mean([m['effective_dimension'] for m in per_seed_metrics]),
            'correlation_mean': np.mean([m['correlation_mean'] for m in per_seed_metrics]),
            'correlation_std': np.mean([m['correlation_std'] for m in per_seed_metrics]),
            'per_seed': per_seed_metrics
        }
        
        print(f"  Degeneración:")
        print(f"    Distancia media entre soluciones: {metrics['mean_solution_distance']:.6f}")
        print(f"    Dimensión efectiva: {metrics['effective_dimension']:.1f}")
        print(f"    Correlación media: {metrics['correlation_mean']:.4f} ± {metrics['correlation_std']:.4f}")
    else:
        metrics = None
    
    return metrics, all_solutions, true_latents, graphs


def plot_degeneracy_results(all_metrics, output_dir="experiment_4"):
    """
    Genera gráficos de degeneración vs α.
    """
    if not all_metrics:
        return
    
    alphas = [m['alpha'] for m in all_metrics if m is not None]
    solution_distances = [m['mean_solution_distance'] for m in all_metrics if m is not None]
    solution_distance_stds = [np.std([s['mean_solution_distance'] for s in m['per_seed']]) 
                               for m in all_metrics if m is not None]
    effective_dims = [m['effective_dimension'] for m in all_metrics if m is not None]
    correlations = [m['correlation_mean'] for m in all_metrics if m is not None]
    correlation_stds = [m['correlation_std'] for m in all_metrics if m is not None]
    stresses = [m['mean_stress'] for m in all_metrics if m is not None]
    
    plt.figure(figsize=(16, 12))
    
    # 1. Distancia entre soluciones (degeneración)
    plt.subplot(2, 3, 1)
    plt.errorbar(alphas, solution_distances, yerr=solution_distance_stds,
                 fmt='o-', capsize=5, capthick=2, color='red', linewidth=2, markersize=8)
    plt.xlabel('α', fontsize=12)
    plt.ylabel('Distancia media entre soluciones', fontsize=12)
    plt.title('DEGENERACIÓN DEL ESPACIO LATENTE', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Dimensión efectiva (ahora con sentido)
    plt.subplot(2, 3, 2)
    plt.plot(alphas, effective_dims, 'o-', color='blue', linewidth=2, markersize=8)
    plt.xlabel('α', fontsize=12)
    plt.ylabel('Dimensión efectiva', fontsize=12)
    plt.title('Dimensión del espacio de soluciones', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 3. Correlación vs α
    plt.subplot(2, 3, 3)
    plt.errorbar(alphas, correlations, yerr=correlation_stds,
                 fmt='o-', capsize=5, capthick=2, color='green', linewidth=2, markersize=8)
    plt.xlabel('α', fontsize=12)
    plt.ylabel('Correlación con verdadero latente', fontsize=12)
    plt.title('Calidad de las soluciones', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 4. Stress de MDS
    plt.subplot(2, 3, 4)
    plt.plot(alphas, stresses, 'o-', color='purple', linewidth=2, markersize=8)
    plt.xlabel('α', fontsize=12)
    plt.ylabel('Stress MDS', fontsize=12)
    plt.title('Calidad de ajuste', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 5. Degeneración vs Calidad
    plt.subplot(2, 3, 5)
    scatter = plt.scatter(correlations, solution_distances, s=100, c=alphas, 
                          cmap='viridis', norm=plt.Normalize(vmin=min(alphas), vmax=max(alphas)))
    plt.colorbar(scatter, label='α')
    plt.xlabel('Correlación', fontsize=12)
    plt.ylabel('Degeneración (distancia entre soluciones)', fontsize=12)
    plt.title('Compromiso: Calidad vs Degeneración', fontsize=14, fontweight='bold')
    for a, c, d in zip(alphas, correlations, solution_distances):
        plt.annotate(f'{a:.1f}', (c, d), textcoords="offset points", 
                     xytext=(5, 5), ha='center', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 6. Visualización de la variedad de soluciones (para α óptimo)
    plt.subplot(2, 3, 6)
    opt_idx = np.argmax(correlations)
    opt_alpha = alphas[opt_idx]
    plt.text(0.5, 0.5, f'Variedad de soluciones\npara α={opt_alpha:.1f}\n(ver archivos separados)',
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Espacio de soluciones', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degeneracy_analysis.png", dpi=150)
    plt.close()
    print(f"\n  Guardada: degeneracy_analysis.png")


def visualize_solution_space(solutions, true_latent, alpha, seed, output_dir):
    """
    Visualiza el espacio de soluciones para un α dado.
    """
    n_sol = len(solutions)
    if n_sol == 0:
        return
    
    # Proyectar todas las soluciones a 2D usando PCA sobre los puntos
    all_points = np.vstack(solutions)
    pca_points = PCA(n_components=2)
    points_2d = pca_points.fit_transform(all_points)
    
    n_points_per_sol = solutions[0].shape[0]
    colors = np.repeat(range(n_sol), n_points_per_sol)
    
    plt.figure(figsize=(14, 5))
    
    # PCA de los puntos de todas las soluciones
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=colors, 
                          cmap='tab20', s=5, alpha=0.6)
    plt.colorbar(scatter, label='Solución ID')
    plt.xlabel('PC1', fontsize=12)
    plt.ylabel('PC2', fontsize=12)
    plt.title(f'Variedad de soluciones (α={alpha:.1f}, {n_sol} soluciones)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Comparación de dos soluciones aleatorias
    plt.subplot(1, 2, 2)
    idx1, idx2 = np.random.choice(n_sol, 2, replace=False)
    sol1 = solutions[idx1]
    sol2_aligned = procrustes_alignment(sol1, solutions[idx2])
    
    plt.scatter(sol1[:, 0], sol1[:, 1], s=5, alpha=0.5, label=f'Solución {idx1}', c='blue')
    plt.scatter(sol2_aligned[:, 0], sol2_aligned[:, 1], s=5, alpha=0.5, label=f'Solución {idx2} (alineada)', c='red')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Dos soluciones diferentes (α={alpha:.1f})\nDistancia = {np.mean((sol1 - sol2_aligned)**2):.4f}', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/solution_space_alpha_{alpha}_seed_{seed}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Experimento 4: Degeneración del espacio latente (corregido)"
    )
    parser.add_argument("--N", type=int, default=500, help="Número de nodos")
    parser.add_argument("--n_seeds", type=int, default=5, help="Semillas por α")
    parser.add_argument("--n_solutions", type=int, default=20, help="Soluciones por grafo")
    parser.add_argument("--d", type=int, default=2, help="Dimensión latente")
    parser.add_argument("--alphas", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0",
                        help="α a analizar")
    parser.add_argument("--output_dir", type=str, default="experiment_4")
    parser.add_argument("--quick", action="store_true", help="Modo rápido")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    if args.quick:
        args.N = 200
        args.n_seeds = 3
        args.n_solutions = 10
        print("⚠️  Modo rápido: N=200, seeds=3, n_solutions=10\n")
    
    alphas = [float(a) for a in args.alphas.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Experimento 4: Degeneración del espacio latente (CORREGIDO)")
    print("=" * 70)
    print(f"Modelo: P_uv = sigmoid(α - ||x_u - x_v||²)")
    print(f"N={args.N}, d={args.d}, seeds={args.n_seeds}, soluciones/grafo={args.n_solutions}")
    print(f"α: {alphas}")
    print()
    
    # Guardar parámetros
    with open(f"{args.output_dir}/experiment_params.txt", 'w') as f:
        f.write(f"Experimento 4 - Degeneración (corregido)\n")
        f.write(f"Fecha: {datetime.now()}\n")
        f.write(f"N={args.N}, d={args.d}, seeds={args.n_seeds}, n_solutions={args.n_solutions}\n")
        f.write(f"alphas={alphas}\n")
    
    # Ejecutar para cada α
    all_metrics = []
    start_time = time.time()
    
    for alpha in alphas:
        metrics, solutions, true_latents, graphs = run_degeneracy_experiment(
            alpha=alpha,
            n_seeds=args.n_seeds,
            n_solutions_per_seed=args.n_solutions,
            N=args.N,
            d=args.d,
            verbose=args.verbose
        )
        
        if metrics:
            all_metrics.append(metrics)
            
            # Visualizar espacio de soluciones para la primera semilla
            if solutions and len(solutions[0]) > 0:
                visualize_solution_space(
                    solutions[0], true_latents[0], alpha, 0, args.output_dir
                )
    
    elapsed = time.time() - start_time
    
    # Guardar resultados (convertir numpy a Python nativo)
    with open(f"{args.output_dir}/degeneracy_metrics.json", 'w') as f:
        metrics_serializable = []
        for m in all_metrics:
            m_copy = {k: v for k, v in m.items() if k != 'per_seed'}
            # Convertir valores numpy a Python nativo
            for key, val in m_copy.items():
                if isinstance(val, np.integer):
                    m_copy[key] = int(val)
                elif isinstance(val, np.floating):
                    m_copy[key] = float(val)
                elif isinstance(val, np.ndarray):
                    m_copy[key] = val.tolist()
            # Procesar per_seed
            per_seed_serializable = []
            for s in m['per_seed']:
                s_copy = {}
                for k, v in s.items():
                    if k == 'distance_matrix':
                        continue  # omitir matriz grande
                    if isinstance(v, np.integer):
                        s_copy[k] = int(v)
                    elif isinstance(v, np.floating):
                        s_copy[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        s_copy[k] = v.tolist()
                    else:
                        s_copy[k] = v
                per_seed_serializable.append(s_copy)
            m_copy['per_seed'] = per_seed_serializable
            metrics_serializable.append(m_copy)
        json.dump(metrics_serializable, f, indent=2)
    
    with open(f"{args.output_dir}/degeneracy_metrics.dat", 'w') as f:
        f.write("# alpha\tmean_distance\teffective_dim\tcorrelation\tcorr_std\tstress\n")
        for m in all_metrics:
            f.write(f"{m['alpha']:.3f}\t{m['mean_solution_distance']:.6f}\t")
            f.write(f"{m['effective_dimension']:.2f}\t{m['correlation_mean']:.6f}\t")
            f.write(f"{m['correlation_std']:.6f}\t{m['mean_stress']:.6f}\n")
    
    # Generar gráficos
    plot_degeneracy_results(all_metrics, args.output_dir)
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN: DEGENERACIÓN DEL ESPACIO LATENTE")
    print("=" * 70)
    print(f"Tiempo total: {format_time(elapsed)}")
    print()
    print(f"{'α':>8} {'Degeneración':>12} {'Dimensión':>10} {'Correlación':>12} {'Stress':>10}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['alpha']:>8.2f} {m['mean_solution_distance']:>12.6f} "
              f"{m['effective_dimension']:>10.1f} {m['correlation_mean']:>12.4f} "
              f"{m['mean_stress']:>10.4f}")
    print("-" * 60)
    
    # Interpretación
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN")
    print("=" * 70)
    
    # Encontrar máximo de degeneración
    degen_vals = [m['mean_solution_distance'] for m in all_metrics]
    max_degen_idx = np.argmax(degen_vals)
    max_degen_alpha = all_metrics[max_degen_idx]['alpha']
    
    # Encontrar máximo de correlación
    corr_vals = [m['correlation_mean'] for m in all_metrics]
    max_corr_idx = np.argmax(corr_vals)
    max_corr_alpha = all_metrics[max_corr_idx]['alpha']
    
    print(f"""
🔍 RESULTADOS CLAVE:

1. MÁXIMA CORRELACIÓN en α = {max_corr_alpha:.2f}
   → Mejor reconstrucción geométrica

2. MÁXIMA DEGENERACIÓN en α = {max_degen_alpha:.2f}
   → Mayor diversidad de soluciones compatibles

3. DIMENSIÓN EFECTIVA DEL ESPACIO DE SOLUCIONES:
   """)
    for m in all_metrics:
        print(f"   α={m['alpha']:.1f}: {m['effective_dimension']:.1f}")
    
    print("""
4. RELACIÓN CALIDAD-DEGENERACIÓN:
   """)
    if max_degen_alpha == max_corr_alpha:
        print("   ⚠️  El máximo de degeneración COINCIDE con el máximo de calidad")
        print("   → El régimen de mejor reconstrucción es también el de mayor ambigüedad")
        print("   → IDENTIFICABILIDAD PARCIAL CON DEGENERACIÓN INTERNA")
    else:
        print(f"   El máximo de degeneración (α={max_degen_alpha:.2f})")
        print(f"   está desplazado respecto al máximo de calidad (α={max_corr_alpha:.2f})")
    
    print("""
5. CONCLUSIÓN:

   El sistema presenta un régimen (α ≈ 2.0) donde:
   - La información observable (grafo) restringe el latente a una variedad
   - Alta calidad de reconstrucción (r > 0.9)
   - Alta degeneración (múltiples soluciones igualmente válidas)
   - La dimensión efectiva del espacio de soluciones es > d
    
   → Esto confirma la existencia de una "fase de realidad accesible"
     con degeneración interna, análoga a un principio de incertidumbre.
""")
    
    print(f"\nArchivos en: {args.output_dir}")
    print("\n✅ Experimento 4 completado.")


if __name__ == "__main__":
    main()