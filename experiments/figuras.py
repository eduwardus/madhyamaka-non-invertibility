#!/usr/bin/env python3
"""
Genera los valores numéricos de la dimensión intrínseca para el paper.
Ejecuta el experimento 9 SOLO para el caso alpha=2.0, d_embed=2,
guarda los autovalores y produce las figuras de codo y varianza acumulada.
Incluye barra de progreso y estimación de tiempo restante.
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
import json
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Intentar importar tqdm para barra de progreso bonita
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("tqdm no instalado. Usando barra de progreso simple.")

# ----------------------------------------------------------------------
# Parámetros fijos (iguales a los del experimento 9)
N = 300
n_seeds = 5
n_solutions = 50
alpha = 2.0
d_embed = 2
output_dir = "paper_figures"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------------------------------------------------
# Funciones (copiadas del experimento 9)
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

def recover_multiple_embeddings(A_sparse, d_embed=2, n_solutions=50, verbose=False, pbar=None):
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
        except Exception:
            continue
        if pbar is not None:
            pbar.update(1)
    return solutions

def align_solutions(solutions):
    if not solutions:
        return []
    ref = solutions[0]
    aligned = [ref]
    for sol in solutions[1:]:
        _, sol_aligned, _ = procrustes(ref, sol)
        aligned.append(sol_aligned)
    return aligned

def solution_to_feature_vector(X):
    return pdist(X)   # distancias por pares

# ----------------------------------------------------------------------
# Barra de progreso manual si tqdm no está disponible
class SimpleProgressBar:
    def __init__(self, total, desc="Progress", length=50):
        self.total = total
        self.desc = desc
        self.length = length
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    def update(self, n=1):
        self.current += n
        elapsed = time.time() - self.start_time
        if self.current >= self.total:
            self.display(elapsed)
            print()
        else:
            # Actualizar cada 2% o cada segundo
            if (self.current / self.total) - self.last_update > 0.02 or (time.time() - self.last_update_time > 1.0):
                self.display(elapsed)
                self.last_update = self.current / self.total
                self.last_update_time = time.time()
    def display(self, elapsed):
        percent = self.current / self.total
        filled = int(self.length * percent)
        bar = '█' * filled + '-' * (self.length - filled)
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        else:
            eta_str = "?"
        sys.stdout.write(f'\r{self.desc}: |{bar}| {percent*100:.1f}% [{elapsed:.1f}s, eta {eta_str}]')
        sys.stdout.flush()

# ----------------------------------------------------------------------
# Recolectar vectores de características con barra de progreso
print(f"Recolectando soluciones para α={alpha}, d_embed={d_embed}")
print(f"Total: {n_seeds} semillas × {n_solutions} soluciones = {n_seeds * n_solutions} embeddings MDS")

all_features = []

if HAS_TQDM:
    pbar = tqdm(total=n_seeds * n_solutions, desc="Generando embeddings", unit="emb")
else:
    pbar = SimpleProgressBar(total=n_seeds * n_solutions, desc="Generando embeddings")

for seed in range(n_seeds):
    A = generate_graph_radial_projection(seed, alpha, N=N, d_latent=2)
    sols = recover_multiple_embeddings(A, d_embed=d_embed, n_solutions=n_solutions, verbose=False, pbar=pbar)
    if len(sols) < 2:
        continue
    aligned = align_solutions(sols)
    for sol in aligned:
        feat = solution_to_feature_vector(sol)
        all_features.append(feat)

if HAS_TQDM:
    pbar.close()
else:
    pbar.update(pbar.total - pbar.current)  # finalizar

print(f"Total de vectores de características: {len(all_features)}")
X = np.vstack(all_features)
print(f"Matriz de datos: {X.shape}")

# ----------------------------------------------------------------------
# PCA
print("Ejecutando PCA...")
pca = PCA()
pca.fit(X)
ev = pca.explained_variance_
cumvar = np.cumsum(pca.explained_variance_ratio_)

# Guardar autovalores en archivo
np.save(os.path.join(output_dir, "eigenvalues.npy"), ev)
np.save(os.path.join(output_dir, "cumulative_variance.npy"), cumvar)

# Mostrar los primeros autovalores
print("\nPrimeros 10 autovalores:")
for i, val in enumerate(ev[:10]):
    print(f"  PC{i+1}: {val:.6f}")

# Determinar dimensión por codo (elbow) usando la segunda derivada de log-ev
log_ev = np.log(ev + 1e-10)
diffs = np.diff(log_ev)
elbow = np.argmin(diffs) + 1  # posición del codo
print(f"\nDimensión intrínseca estimada por codo: {elbow}")

# Usar también el criterio de umbral (1% del primer autovalor)
threshold = 0.01 * ev[0]
dim_thresh = np.sum(ev > threshold)
print(f"Dimensión por umbral (1% del primero): {dim_thresh}")

# ----------------------------------------------------------------------
# Figura 1: Espectro de autovalores (escala lineal)
plt.figure(figsize=(8,6))
plt.plot(range(1, len(ev)+1), ev, 'o-', linewidth=2, markersize=4)
plt.axvline(x=elbow, color='red', linestyle='--', alpha=0.7, label=f'elbow at k={elbow}')
plt.xlabel("Principal Component", fontsize=12)
plt.ylabel("Eigenvalue", fontsize=12)
plt.title(f"Intrinsic Dimension (α={alpha}, d_embed={d_embed})", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "intrinsic_dimension.png"), dpi=200)
plt.close()

# Figura 2: Varianza acumulada
plt.figure(figsize=(8,6))
plt.plot(range(1, len(cumvar)+1), cumvar, 'o-', linewidth=2, markersize=4)
plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95%')
plt.axvline(x=elbow, color='red', linestyle='--', alpha=0.7, label=f'k={elbow}')
plt.xlabel("Principal Component", fontsize=12)
plt.ylabel("Cumulative Variance", fontsize=12)
plt.title(f"Cumulative Variance (α={alpha}, d_embed={d_embed})", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "intrinsic_dimension_cumulative.png"), dpi=200)
plt.close()

# ----------------------------------------------------------------------
# Guardar también un pequeño resumen en JSON
summary = {
    "alpha": alpha,
    "d_embed": d_embed,
    "N": N,
    "n_seeds": n_seeds,
    "n_solutions": n_solutions,
    "intrinsic_dim_elbow": int(elbow),
    "intrinsic_dim_threshold": int(dim_thresh),
    "first_10_eigenvalues": ev[:10].tolist(),
    "cumulative_variance_10": cumvar[:10].tolist()
}
with open(os.path.join(output_dir, "intrinsic_dim_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResultados guardados en '{output_dir}'")
print("Archivos generados:")
print("  - eigenvalues.npy")
print("  - cumulative_variance.npy")
print("  - intrinsic_dimension.png")
print("  - intrinsic_dimension_cumulative.png")
print("  - intrinsic_dim_summary.json")