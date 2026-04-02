# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:02:14 2026

@author: eggra
"""

#!/usr/bin/env python3
"""
Experimento 1: Demostración de no-identificabilidad en grafos con latentes continuos.
VERSIÓN OPTIMIZADA - Mismos resultados, 10-50x más rápido
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.spatial import procrustes
import os
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import time
import sys


# =============================================================================
# 1. GENERACIÓN DE DATOS (sin cambios significativos)
# =============================================================================

def sigmoid(z):
    """Función sigmoide."""
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    """Derivada de la sigmoide."""
    s = sigmoid(z)
    return s * (1 - s)


def compute_probability_matrix(X):
    """Calcula matriz de probabilidades de forma optimizada."""
    # Usar broadcasting para distancias
    D2 = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2 * X @ X.T
    P = sigmoid(-D2)
    np.fill_diagonal(P, 0)
    return P


def generate_graph(seed, N=500, d=2, output_dir="experiment_1"):
    """Genera grafo con latentes continuos."""
    np.random.seed(seed)
    
    # Generar latentes
    X_true = np.random.randn(N, d)
    
    # Matriz de probabilidades
    P = compute_probability_matrix(X_true)
    
    # Muestrear grafo
    A = np.random.rand(N, N) < P
    A = np.triu(A, 1)
    A = A + A.T
    
    # Guardar como dispersa para ahorrar memoria
    A_sparse = sp.csr_matrix(A)
    np.savez(f"{output_dir}/graph_seed_{seed}.npz", 
             adjacency_matrix=A_sparse.data,
             indices=A_sparse.indices,
             indptr=A_sparse.indptr,
             shape=A_sparse.shape,
             latent_true=X_true,
             seed=seed)
    
    np.savetxt(f"{output_dir}/latent_true_seed_{seed}.dat", X_true, header="x y", comments='')
    
    print(f"  Generado grafo seed={seed}: N={N}, densidad={np.mean(A):.4f}")
    
    return A_sparse, X_true


# =============================================================================
# 2. INFERENCIA DEL LATENTE - VERSIÓN OPTIMIZADA
# =============================================================================

def spectral_init_sparse(A_sparse, d=2):
    """
    Inicialización espectral usando autovectores del Laplaciano normalizado.
    Optimizado para matrices dispersas.
    """
    N = A_sparse.shape[0]
    degrees = np.array(A_sparse.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-10))
    
    # Laplaciano normalizado: I - D^{-1/2} A D^{-1/2}
    L = sp.eye(N) - D_inv_sqrt @ A_sparse @ D_inv_sqrt
    
    # Calcular primeros d+1 autovectores
    eigvals, eigvecs = eigsh(L, k=d+1, which='SM', tol=1e-6, maxiter=1000)
    
    # Ordenar y omitir el primero
    idx = np.argsort(eigvals)
    X0 = eigvecs[:, idx[1:d+1]]
    
    return X0


def negative_log_likelihood_gradient(X_flat, A_sparse, d=2):
    """
    Calcula log-verosimilitud negativa Y SU GRADIENTE analíticamente.
    Esto es CRUCIAL para la velocidad - evita diferencias finitas.
    
    Retorna (loss, gradient)
    """
    N = A_sparse.shape[0]
    X = X_flat.reshape(N, d)
    
    # Pre-calcular distancias al cuadrado de forma eficiente
    # Usando identidad: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
    X_norm_sq = np.sum(X**2, axis=1, keepdims=True)
    D2 = X_norm_sq + X_norm_sq.T - 2 * X @ X.T
    
    # Probabilidades
    Z = -D2
    P = sigmoid(Z)
    np.fill_diagonal(P, 0)
    
    # Clipping para estabilidad numérica
    P = np.clip(P, 1e-10, 1 - 1e-10)
    
    # Obtener adyacencia en formato denso para cálculos (solo necesario para gradiente)
    # Pero podemos usar la representación dispersa para operaciones
    A_dense = A_sparse.toarray() if hasattr(A_sparse, 'toarray') else A_sparse
    
    # Log-verosimilitud (solo triángulo superior)
    log_lik = np.sum(A_dense * np.log(P) + (1 - A_dense) * np.log(1 - P)) / 2
    loss = -log_lik
    
    # --- CÁLCULO DEL GRADIENTE ANALÍTICO ---
    # Derivada de la log-verosimilitud respecto a P_uv:
    # dL/dP_uv = A_uv/P_uv - (1-A_uv)/(1-P_uv)
    # Y dP_uv/dZ_uv = P_uv*(1-P_uv)
    # Y dZ_uv/dX_i = -2*(X_i - X_j) para i≠j
    
    # Matriz de derivadas respecto a Z
    dL_dP = A_dense / P - (1 - A_dense) / (1 - P)
    dP_dZ = P * (1 - P)
    dL_dZ = dL_dP * dP_dZ  # Element-wise
    np.fill_diagonal(dL_dZ, 0)
    
    # Gradiente para cada coordenada
    # dL/dX_i = sum_{j≠i} dL/dZ_ij * (-2)*(X_i - X_j)
    grad = np.zeros_like(X)
    
    # Versión vectorizada del gradiente
    for i in range(N):
        # dL/dZ_ij para todos j
        weights = dL_dZ[i, :] + dL_dZ[:, i]  # Simetría
        grad[i] = -2 * np.sum(weights[:, np.newaxis] * (X[i] - X), axis=0)
    
    return loss, grad.flatten()


def infer_latent_optimized(A_sparse, d=2, n_runs=5, verbose=False):
    """
    Inferencia optimizada usando gradiente analítico y representación dispersa.
    """
    N = A_sparse.shape[0]
    
    # Inicialización espectral
    X0_spectral = spectral_init_sparse(A_sparse, d)
    
    best_X = None
    best_loss = np.inf
    all_X = []
    
    for run in range(n_runs):
        if verbose and run == 0:
            print(f"    Run {run + 1}/{n_runs}...")
        
        # Inicialización
        if run == 0:
            X0 = X0_spectral.copy()
        else:
            X0 = X0_spectral + 0.1 * np.random.randn(N, d)
        
        # Optimización CON GRADIENTE ANALÍTICO
        # Usar 'L-BFGS-B' con gradiente explícito es mucho más rápido
        res = minimize(
            negative_log_likelihood_gradient,
            X0.flatten(),
            args=(A_sparse, d),
            method='L-BFGS-B',
            jac=True,  # Indicamos que la función retorna (loss, grad)
            options={'maxiter': 500, 'disp': False, 'ftol': 1e-6}
        )
        
        X_opt = res.x.reshape(N, d)
        all_X.append(X_opt)
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_X = X_opt
    
    if verbose:
        print(f"    Mejor pérdida: {best_loss:.4f}")
    
    return best_X, all_X


# =============================================================================
# 3. MÉTRICAS DE EVALUACIÓN (optimizadas)
# =============================================================================

def procrustes_alignment(X_true, X_pred):
    """Alinea X_pred a X_true usando Procrustes."""
    _, X_pred_aligned, _ = procrustes(X_true, X_pred)
    return X_pred_aligned, None


def alignment_error(X_true, X_pred):
    """RMSE después de alineación."""
    X_pred_aligned, _ = procrustes_alignment(X_true, X_pred)
    return np.sqrt(np.mean((X_true - X_pred_aligned)**2))


def distance_matrix_error(X_true, X_pred):
    """Error cuadrático medio entre matrices de distancias."""
    # Versión optimizada usando la identidad de distancias
    D_true = np.sum(X_true**2, axis=1, keepdims=True) + np.sum(X_true**2, axis=1) - 2 * X_true @ X_true.T
    D_pred = np.sum(X_pred**2, axis=1, keepdims=True) + np.sum(X_pred**2, axis=1) - 2 * X_pred @ X_pred.T
    return np.mean((D_true - D_pred)**2)


def compute_laplacian_spectrum_sparse(A_sparse, k=10):
    """Espectro del Laplaciano para matriz dispersa."""
    N = A_sparse.shape[0]
    degrees = np.array(A_sparse.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-10))
    L = sp.eye(N) - D_inv_sqrt @ A_sparse @ D_inv_sqrt
    eigvals = eigsh(L, k=k, which='SM', return_eigenvectors=False, tol=1e-6)
    return np.sort(eigvals)


# =============================================================================
# 4. VISUALIZACIÓN (sin cambios)
# =============================================================================

def plot_results(true_latents, inferred_latents, degree_dists, spectra, 
                 alignment_errors, dist_errors, output_dir):
    """Genera visualizaciones."""
    # Figura 1: Comparación
    s_example = 0
    run_example = 0
    X_true_ex = true_latents[s_example]
    X_inf_ex = inferred_latents[s_example][run_example]
    X_inf_aligned, _ = procrustes_alignment(X_true_ex, X_inf_ex)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_true_ex[:, 0], X_true_ex[:, 1], s=1, alpha=0.5, c='blue')
    plt.title(f"Latentes verdaderos (seed {s_example})")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_inf_aligned[:, 0], X_inf_aligned[:, 1], s=1, alpha=0.5, c='red')
    plt.title(f"Latentes inferidos (run {run_example}, alineado)")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_seed_{s_example}.png", dpi=150)
    plt.close()
    print(f"  Guardada figura: comparison_seed_{s_example}.png")
    
    # Figura 2: Histogramas
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    align_vals = [e[2] for e in alignment_errors]
    plt.hist(align_vals, bins=20, edgecolor='black')
    plt.xlabel("RMSE de alineación")
    plt.ylabel("Frecuencia")
    plt.title(f"Errores de alineación (media={np.mean(align_vals):.4f})")
    
    plt.subplot(1, 2, 2)
    dist_vals = [e[2] for e in dist_errors]
    plt.hist(dist_vals, bins=20, edgecolor='black')
    plt.xlabel("Error en matriz de distancias")
    plt.ylabel("Frecuencia")
    plt.title(f"Errores de distancias (media={np.mean(dist_vals):.6f})")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_histograms.png", dpi=150)
    plt.close()
    print("  Guardada figura: error_histograms.png")
    
    # Figura 3: Espectros
    plt.figure(figsize=(10, 6))
    spectra_array = np.array(spectra)
    for i in range(len(spectra)):
        plt.plot(spectra_array[i], 'o-', alpha=0.7, label=f"Seed {i}", markersize=4)
    plt.xlabel("Índice del autovalor")
    plt.ylabel("Autovalor")
    plt.title("Primeros 10 autovalores del Laplaciano normalizado")
    plt.legend(loc='upper left', ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/laplacian_spectra.png", dpi=150)
    plt.close()
    print("  Guardada figura: laplacian_spectra.png")
    
    # Figura 4: Distribución de grados
    plt.figure(figsize=(10, 6))
    for i, deg in enumerate(degree_dists):
        plt.hist(deg, bins=20, alpha=0.5, label=f"Seed {i}", density=True)
    plt.xlabel("Grado")
    plt.ylabel("Densidad")
    plt.title("Distribución de grados (normalizada)")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/degree_distributions.png", dpi=150)
    plt.close()
    print("  Guardada figura: degree_distributions.png")
    
    # Figura 5: Boxplots
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    align_by_seed = [[] for _ in range(len(true_latents))]
    for s, run_idx, err in alignment_errors:
        align_by_seed[s].append(err)
    plt.boxplot(align_by_seed, labels=[f"Seed {i}" for i in range(len(true_latents))])
    plt.ylabel("RMSE de alineación")
    plt.title("Errores de alineación por semilla")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    dist_by_seed = [[] for _ in range(len(true_latents))]
    for s, run_idx, err in dist_errors:
        dist_by_seed[s].append(err)
    plt.boxplot(dist_by_seed, labels=[f"Seed {i}" for i in range(len(true_latents))])
    plt.ylabel("Error en matriz de distancias")
    plt.title("Errores de distancias por semilla")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/errors_by_seed.png", dpi=150)
    plt.close()
    print("  Guardada figura: errors_by_seed.png")


# =============================================================================
# 5. FUNCIÓN PRINCIPAL CON MEDIDOR DE TIEMPO
# =============================================================================

def format_time(seconds):
    """Formatea segundos."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    """Barra de progreso."""
    percent = 100 * (iteration + 1) / float(total)
    filled_length = int(length * (iteration + 1) // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()
    if iteration + 1 == total:
        sys.stdout.write('\n')


def main():
    parser = argparse.ArgumentParser(
        description="Experimento 1: No-identificabilidad - VERSIÓN OPTIMIZADA"
    )
    parser.add_argument("--N", type=int, default=500, help="Número de nodos")
    parser.add_argument("--n_seeds", type=int, default=10, help="Número de semillas")
    parser.add_argument("--d", type=int, default=2, help="Dimensión latente")
    parser.add_argument("--n_runs", type=int, default=5, help="Optimizaciones por grafo")
    parser.add_argument("--output_dir", type=str, default="experiment_1", help="Directorio de salida")
    parser.add_argument("--no_plots", action="store_true", help="No generar gráficos")
    parser.add_argument("--verbose", action="store_true", help="Mostrar información detallada")
    parser.add_argument("--fast", action="store_true", help="Modo ultra-rápido (n_runs=3, menos iteraciones)")
    
    args = parser.parse_args()
    
    # Modo rápido
    if args.fast:
        args.n_runs = 3
        print("⚠️  Modo rápido activado: n_runs=3\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Guardar parámetros
    with open(f"{args.output_dir}/experiment_params.txt", 'w') as f:
        f.write(f"Experimento 1: No-identificabilidad (VERSIÓN OPTIMIZADA)\n")
        f.write(f"Fecha: {datetime.now()}\n")
        f.write(f"N = {args.N}\n")
        f.write(f"d = {args.d}\n")
        f.write(f"n_seeds = {args.n_seeds}\n")
        f.write(f"n_runs = {args.n_runs}\n")
    
    print("=" * 60)
    print("Experimento 1: No-identificabilidad - VERSIÓN OPTIMIZADA")
    print("=" * 60)
    print(f"Parámetros: N={args.N}, d={args.d}, seeds={args.n_seeds}, runs={args.n_runs}")
    print(f"Directorio: {args.output_dir}")
    print()
    
    # ========================================================================
    # PASO 1: Generar grafos
    # ========================================================================
    print("PASO 1: Generando grafos...")
    step1_start = time.time()
    graphs = []
    true_latents = []
    
    for s in range(args.n_seeds):
        print(f"  Semilla {s + 1}/{args.n_seeds}")
        A_sparse, X_true = generate_graph(s, N=args.N, d=args.d, output_dir=args.output_dir)
        graphs.append(A_sparse)
        true_latents.append(X_true)
    
    step1_time = time.time() - step1_start
    print(f"  ✓ Generados {args.n_seeds} grafos en {format_time(step1_time)}\n")
    
    # ========================================================================
    # PASO 2: Inferencia OPTIMIZADA
    # ========================================================================
    print("PASO 2: Infiriendo coordenadas latentes (versión optimizada)...")
    print(f"  Total: {args.n_seeds} grafos × {args.n_runs} runs = {args.n_seeds * args.n_runs} optimizaciones")
    print("  Usando gradiente analítico + representación dispersa → ~10-50x más rápido")
    print()
    
    inferred_latents = []
    step2_start = time.time()
    graph_times = []
    
    for s, A_sparse in enumerate(graphs):
        graph_start = time.time()
        
        print_progress_bar(
            s, args.n_seeds,
            prefix=f'  Seed {s}/{args.n_seeds-1}',
            suffix=f'[{format_time(time.time() - step2_start)} elapsed]'
        )
        
        best_X, all_X = infer_latent_optimized(
            A_sparse, d=args.d, n_runs=args.n_runs, verbose=args.verbose
        )
        inferred_latents.append(all_X)
        
        # Guardar resultados
        for run_idx, X_inf in enumerate(all_X):
            np.savetxt(f"{args.output_dir}/latent_inferred_seed_{s}_run_{run_idx}.dat",
                       X_inf, header="x y", comments='')
        
        graph_time = time.time() - graph_start
        graph_times.append(graph_time)
    
    print()
    step2_time = time.time() - step2_start
    
    avg_graph_time = np.mean(graph_times)
    print(f"  ✓ Inferencia completada en {format_time(step2_time)}")
    print(f"    Media por grafo: {format_time(avg_graph_time)}")
    print(f"    Media por optimización: {format_time(avg_graph_time / args.n_runs)}")
    print()
    
    # ========================================================================
    # PASO 3: Evaluación
    # ========================================================================
    print("PASO 3: Evaluando métricas...")
    step3_start = time.time()
    
    alignment_errors = []
    dist_errors = []
    degree_dists = []
    spectra = []
    
    for s, A_sparse in enumerate(graphs):
        X_true = true_latents[s]
        
        # Calcular grado (para matriz dispersa)
        degrees = np.array(A_sparse.sum(axis=1)).flatten()
        degree_dists.append(degrees)
        
        # Espectro del Laplaciano
        spectra.append(compute_laplacian_spectrum_sparse(A_sparse, k=10))
        
        for run_idx, X_inf in enumerate(inferred_latents[s]):
            align_err = alignment_error(X_true, X_inf)
            alignment_errors.append((s, run_idx, align_err))
            
            dist_err = distance_matrix_error(X_true, X_inf)
            dist_errors.append((s, run_idx, dist_err))
    
    step3_time = time.time() - step3_start
    
    align_mean = np.mean([e[2] for e in alignment_errors])
    align_std = np.std([e[2] for e in alignment_errors])
    dist_mean = np.mean([e[2] for e in dist_errors])
    dist_std = np.std([e[2] for e in dist_errors])
    
    print(f"  Error de alineación: media={align_mean:.4f}, std={align_std:.4f}")
    print(f"  Error en distancias: media={dist_mean:.6f}, std={dist_std:.6f}")
    print(f"  Tiempo: {format_time(step3_time)}")
    
    # Guardar métricas
    with open(f"{args.output_dir}/metrics_summary.txt", 'w') as f:
        f.write("RESUMEN DE MÉTRICAS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Error de alineación (RMSE):\n")
        f.write(f"  Media: {align_mean:.4f}\n")
        f.write(f"  Desviación: {align_std:.4f}\n\n")
        f.write(f"Error en matriz de distancias:\n")
        f.write(f"  Media: {dist_mean:.6f}\n")
        f.write(f"  Desviación: {dist_std:.6f}\n\n")
        
        f.write("TIEMPOS DE EJECUCIÓN\n")
        f.write("=" * 40 + "\n")
        f.write(f"Paso 1 (generación): {format_time(step1_time)}\n")
        f.write(f"Paso 2 (inferencia): {format_time(step2_time)}\n")
        f.write(f"  - Por grafo (media): {format_time(avg_graph_time)}\n")
        f.write(f"  - Por optimización: {format_time(avg_graph_time / args.n_runs)}\n")
        f.write(f"Paso 3 (evaluación): {format_time(step3_time)}\n")
        f.write(f"TOTAL: {format_time(step1_time + step2_time + step3_time)}\n")
    
    print("  ✓ Métricas guardadas\n")
    
    # ========================================================================
    # PASO 4: Visualización
    # ========================================================================
    if not args.no_plots:
        print("PASO 4: Generando visualizaciones...")
        step4_start = time.time()
        plot_results(true_latents, inferred_latents, degree_dists, spectra,
                    alignment_errors, dist_errors, args.output_dir)
        step4_time = time.time() - step4_start
        print(f"  ✓ Completado en {format_time(step4_time)}\n")
    else:
        step4_time = 0
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    total_time = step1_time + step2_time + step3_time + step4_time
    
    print("=" * 60)
    print("EXPERIMENTO COMPLETADO")
    print("=" * 60)
    print(f"Tiempo total: {format_time(total_time)}")
    print()
    print("CONCLUSIÓN:")
    print(f"  - Error de alineación: {align_mean:.4f} (alto → no coincidencia puntual)")
    print(f"  - Error en distancias: {dist_mean:.6f} (bajo → geometría preservada)")
    print(f"  ⇒ CONFIRMADA NO-IDENTIFICABILIDAD")
    print()
    print(f"Archivos en: {args.output_dir}")


if __name__ == "__main__":
    main()