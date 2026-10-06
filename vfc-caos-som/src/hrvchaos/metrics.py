"""Métricas lineales y no lineales de VFC, implementadas en NumPy para que cada
paso sea legible y verificable (sin cajas negras).

Convenciones: `rr` en segundos; las métricas en ms se indican en el nombre.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Dominio del tiempo y Poincaré
# ---------------------------------------------------------------------------
def time_domain(rr: np.ndarray) -> dict:
    rr_ms = np.asarray(rr) * 1000
    d = np.diff(rr_ms)
    return {
        "mean_nn_ms": rr_ms.mean(),
        "sdnn_ms": rr_ms.std(ddof=1),
        "rmssd_ms": np.sqrt(np.mean(d ** 2)),
        "pnn50": np.mean(np.abs(d) > 50) * 100,
    }


def poincare(rr: np.ndarray) -> dict:
    """SD1 = dispersión perpendicular a la identidad (variabilidad latido a latido),
    SD2 = dispersión a lo largo de la identidad (variabilidad lenta)."""
    rr_ms = np.asarray(rr) * 1000
    sd1 = np.sqrt(np.var(np.diff(rr_ms), ddof=1) / 2)
    sd2 = np.sqrt(max(2 * np.var(rr_ms, ddof=1) - sd1 ** 2, 0))
    return {"sd1_ms": sd1, "sd2_ms": sd2, "sd1_sd2": sd1 / sd2 if sd2 > 0 else np.nan,
            "poincare_area": np.pi * sd1 * sd2}


# ---------------------------------------------------------------------------
# Entropía Muestral y Multiescala
# ---------------------------------------------------------------------------
def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None, r_factor: float = 0.2) -> float:
    """SampEn(m, r) de Richman & Moorman (2000).

    Cuenta pares de plantillas (sin autocomparaciones) cuya distancia de Chebyshev es <= r,
    para longitud m (B) y m+1 (A). SampEn = -ln(A/B). Se usan las N-m plantillas en ambos casos.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if r is None:
        r = r_factor * x.std()
    if n <= m + 2:
        return np.nan
    a = b = 0
    for k in range(1, n - m):
        d = np.abs(x[:-k] - x[k:])  # d[i] = |x[i] - x[i+k]|
        n_pairs = n - m - k  # plantillas i con i+k <= n-m-1
        dm = sliding_window_view(d, m)[:n_pairs].max(axis=1)
        match_m = dm <= r
        b += match_m.sum()
        a += (match_m & (d[m:m + n_pairs] <= r)).sum()
    if a == 0 or b == 0:
        return np.nan
    return -np.log(a / b)


def coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
    n = len(x) // scale
    return np.asarray(x[: n * scale]).reshape(n, scale).mean(axis=1)


def multiscale_entropy(x: np.ndarray, scales=range(1, 11), m: int = 2, r_factor: float = 0.2) -> np.ndarray:
    """MSE de Costa et al. (2002): r fijo con la SD de la serie original."""
    x = np.asarray(x, dtype=float)
    r = r_factor * x.std()
    return np.array([sample_entropy(coarse_grain(x, s), m=m, r=r) for s in scales])


# ---------------------------------------------------------------------------
# DFA
# ---------------------------------------------------------------------------
def dfa(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Función de fluctuación F(n) de Peng et al. (1995), detrend lineal por caja."""
    x = np.asarray(x, dtype=float)
    y = np.cumsum(x - x.mean())
    F = []
    for n in scales:
        nb = len(y) // n
        boxes = y[: nb * n].reshape(nb, n)
        t = np.arange(n)
        coef = np.polyfit(t, boxes.T, 1)  # (2, nb)
        trend = np.outer(coef[0], t) + coef[1][:, None]
        F.append(np.sqrt(np.mean((boxes - trend) ** 2)))
    return np.array(F)


def dfa_alpha(x: np.ndarray, n_min: int, n_max: int) -> float:
    scales = np.unique(np.round(np.logspace(np.log10(n_min), np.log10(n_max), 12)).astype(int))
    F = dfa(x, scales)
    return np.polyfit(np.log(scales), np.log(F), 1)[0]


# ---------------------------------------------------------------------------
# Exponente máximo de Lyapunov (Rosenstein et al., 1993)
# ---------------------------------------------------------------------------
def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    n = len(x) - (m - 1) * tau
    return np.stack([x[i * tau: i * tau + n] for i in range(m)], axis=1)


def lyapunov_rosenstein(x: np.ndarray, m: int = 5, tau: int = 1, theiler: int = 10, max_t: int = 20,
                        fit: tuple[int, int] = (0, 10), return_curve: bool = False):
    """λ1 en unidades de 1/latido.

    1. Reconstruye el espacio de fases con retardo `tau` y dimensión `m`.
    2. Para cada punto busca su vecino más cercano fuera de la ventana de Theiler.
    3. Promedia ln(distancia) entre pares a medida que avanzan k pasos.
    4. La pendiente de esa curva en `fit` estima λ1.
    """
    x = np.asarray(x, dtype=float)
    x = (x - x.mean()) / x.std()
    X = embed(x, m, tau)
    N = len(X) - max_t
    Xa = X[:N]
    sq = (Xa ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * Xa @ Xa.T
    idx = np.arange(N)
    D[np.abs(idx[:, None] - idx[None, :]) <= theiler] = np.inf
    nn = D.argmin(axis=1)
    curve = np.empty(max_t)
    for k in range(max_t):
        d = np.linalg.norm(X[idx + k] - X[nn + k], axis=1)
        d = d[d > 0]
        curve[k] = np.mean(np.log(d))
    ks = np.arange(fit[0], fit[1])
    lam = np.polyfit(ks, curve[ks], 1)[0]
    return (lam, curve) if return_curve else lam


# ---------------------------------------------------------------------------
# Datos sustitutos IAAFT (prueba de no linealidad)
# ---------------------------------------------------------------------------
def iaaft(x: np.ndarray, n_iter: int = 100, seed: int | None = None) -> np.ndarray:
    """Sustituto con el mismo espectro y la misma distribución que `x`, pero sin estructura
    no lineal (Schreiber & Schmitz, 1996)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    s = rng.permutation(x)
    for _ in range(n_iter):
        phases = np.angle(np.fft.rfft(s))
        s = np.fft.irfft(amp * np.exp(1j * phases), len(x))
        s = sorted_x[np.argsort(np.argsort(s))]
    return s


# ---------------------------------------------------------------------------
# Vector de características por ventana
# ---------------------------------------------------------------------------
LINEAR = ["mean_nn_ms", "sdnn_ms", "rmssd_ms", "pnn50"]
NONLINEAR = ["sd1_ms", "sd2_ms", "sd1_sd2", "sampen", "mse_short", "mse_long", "dfa_a1", "dfa_a2", "lyap"]


def window_features(rr: np.ndarray, sampen_m: int = 2, sampen_r: float = 0.2, lyap_m: int = 5,
                    lyap_tau: int = 1) -> dict:
    rr = np.asarray(rr, dtype=float)
    f = {}
    f.update(time_domain(rr))
    f.update(poincare(rr))
    f["sampen"] = sample_entropy(rr, m=sampen_m, r_factor=sampen_r)
    mse = multiscale_entropy(rr, scales=range(1, 11), m=sampen_m, r_factor=sampen_r)
    f["mse_short"] = np.nansum(mse[:5])  # área escalas 1–5
    f["mse_long"] = np.nansum(mse[5:])   # área escalas 6–10
    for s, v in enumerate(mse, start=1):
        f[f"mse_{s}"] = v
    f["dfa_a1"] = dfa_alpha(rr, 4, 16)
    f["dfa_a2"] = dfa_alpha(rr, 16, 64)
    f["lyap"] = lyapunov_rosenstein(rr, m=lyap_m, tau=lyap_tau)
    return f
