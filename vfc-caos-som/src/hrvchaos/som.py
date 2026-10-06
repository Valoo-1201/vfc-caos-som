"""Mapa Autoorganizado (Kohonen) en NumPy, con las herramientas de análisis
topológico que usa el proyecto: U-matriz, planos de componentes, mapa de
impactos, error de cuantización (QE) y error topográfico (TE).
"""
from __future__ import annotations

import numpy as np


class SOM:
    def __init__(self, rows: int, cols: int, dim: int, sigma0: float | None = None, sigma_end: float = 0.5,
                 lr0: float = 0.5, lr_end: float = 0.01, seed: int = 42):
        self.rows, self.cols, self.dim = rows, cols, dim
        self.sigma0 = sigma0 or max(rows, cols) / 3
        self.sigma_end, self.lr0, self.lr_end = sigma_end, lr0, lr_end
        self.rng = np.random.default_rng(seed)
        gy, gx = np.mgrid[0:rows, 0:cols]
        # Malla hexagonal: filas impares desplazadas media celda.
        self.grid = np.stack([gx + 0.5 * (gy % 2), gy * np.sqrt(3) / 2], axis=-1).reshape(-1, 2)
        self.W = self.rng.normal(scale=0.1, size=(rows * cols, dim))

    @staticmethod
    def suggest_shape(n_samples: int, data: np.ndarray | None = None) -> tuple[int, int]:
        """Heurística de Vesanto: ~5·√N nodos, proporción según los dos primeros autovalores."""
        n_nodes = int(round(5 * np.sqrt(n_samples)))
        ratio = 1.0
        if data is not None and data.shape[1] > 1:
            ev = np.sort(np.linalg.eigvalsh(np.cov(data.T)))[::-1]
            ratio = np.sqrt(ev[0] / ev[1]) if ev[1] > 0 else 1.0
        rows = max(2, int(round(np.sqrt(n_nodes / ratio))))
        cols = max(2, int(round(n_nodes / rows)))
        return rows, cols

    def pca_init(self, X: np.ndarray):
        """Inicializa los prototipos sobre el plano de las dos primeras componentes principales."""
        mu = X.mean(0)
        _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
        std = s[:2] / np.sqrt(len(X))
        g = (self.grid - self.grid.mean(0)) / (self.grid.std(0) + 1e-12)
        self.W = mu + g[:, :1] * std[0] * vt[0] + g[:, 1:2] * std[1] * vt[1]
        return self

    def bmu(self, X: np.ndarray) -> np.ndarray:
        d = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(-1)
        return d.argmin(1)

    def fit(self, X: np.ndarray, n_iter: int = 10000, init: str = "pca", verbose: bool = False):
        X = np.asarray(X, float)
        if init == "pca":
            self.pca_init(X)
        else:
            self.W = X[self.rng.integers(0, len(X), len(self.W))].copy()
        for t in range(n_iter):
            frac = t / n_iter
            sigma = self.sigma0 * (self.sigma_end / self.sigma0) ** frac
            lr = self.lr0 * (self.lr_end / self.lr0) ** frac
            x = X[self.rng.integers(len(X))]
            b = ((self.W - x) ** 2).sum(1).argmin()
            h = np.exp(-((self.grid - self.grid[b]) ** 2).sum(1) / (2 * sigma ** 2))
            self.W += lr * h[:, None] * (x - self.W)
            if verbose and t % (n_iter // 5) == 0:
                print(f"iter {t:6d}  sigma={sigma:.2f}  lr={lr:.3f}  QE={self.quantization_error(X):.3f}")
        return self

    # --- calidad ---------------------------------------------------------
    def quantization_error(self, X: np.ndarray) -> float:
        return float(np.sqrt(((X - self.W[self.bmu(X)]) ** 2).sum(1)).mean())

    def topographic_error(self, X: np.ndarray) -> float:
        """Fracción de muestras cuyas dos mejores neuronas NO son vecinas en la malla."""
        d = ((X[:, None, :] - self.W[None, :, :]) ** 2).sum(-1)
        best2 = np.argsort(d, axis=1)[:, :2]
        gd = np.linalg.norm(self.grid[best2[:, 0]] - self.grid[best2[:, 1]], axis=1)
        return float(np.mean(gd > 1.01))

    # --- análisis topológico --------------------------------------------
    def neighbors(self, i: int) -> np.ndarray:
        d = np.linalg.norm(self.grid - self.grid[i], axis=1)
        return np.where((d > 0) & (d < 1.01))[0]

    def u_matrix(self) -> np.ndarray:
        """Distancia media de cada prototipo a sus vecinos; valores altos = fronteras entre grupos."""
        u = np.array([np.linalg.norm(self.W[self.neighbors(i)] - self.W[i], axis=1).mean()
                      for i in range(len(self.W))])
        return u.reshape(self.rows, self.cols)

    def component_planes(self) -> np.ndarray:
        return self.W.reshape(self.rows, self.cols, self.dim)

    def hit_map(self, X: np.ndarray, labels: np.ndarray | None = None) -> dict:
        b = self.bmu(X)
        if labels is None:
            return {"all": np.bincount(b, minlength=len(self.W)).reshape(self.rows, self.cols)}
        return {c: np.bincount(b[labels == c], minlength=len(self.W)).reshape(self.rows, self.cols)
                for c in np.unique(labels)}

    def label_nodes(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Etiqueta de cada nodo = clase mayoritaria de las muestras que captura.
        Nodos vacíos heredan la etiqueta del prototipo etiquetado más cercano."""
        b = self.bmu(X)
        classes = np.unique(y)
        lab = np.full(len(self.W), None, dtype=object)
        for i in range(len(self.W)):
            yi = y[b == i]
            if len(yi):
                vals, cnt = np.unique(yi, return_counts=True)
                lab[i] = vals[cnt.argmax()]
        filled = np.array([l is not None for l in lab])
        for i in np.where(~filled)[0]:
            j = np.where(filled)[0][np.linalg.norm(self.W[filled] - self.W[i], axis=1).argmin()]
            lab[i] = lab[j]
        self.node_labels = lab
        self.classes = classes
        return lab

    def node_scores(self, X: np.ndarray, y: np.ndarray, positive) -> np.ndarray:
        """Proporción de la clase positiva por nodo (suavizada con la vecindad) para obtener un puntaje continuo."""
        b = self.bmu(X)
        pos = np.bincount(b, weights=(y == positive).astype(float), minlength=len(self.W))
        tot = np.bincount(b, minlength=len(self.W)).astype(float)
        K = np.exp(-((self.grid[:, None] - self.grid[None]) ** 2).sum(-1) / 2)
        self.scores = (K @ pos + 0.5) / (K @ tot + 1.0)
        return self.scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.node_labels[self.bmu(X)]

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.scores[self.bmu(X)]
