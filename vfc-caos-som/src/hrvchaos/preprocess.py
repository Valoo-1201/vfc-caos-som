"""Limpieza de series NN y segmentación en ventanas."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def clean_rr(rr: np.ndarray, lo: float = 0.3, hi: float = 2.0, max_rel_change: float = 0.2,
             median_width: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (rr_limpio, mascara_valida).

    1. Rango fisiológico [lo, hi] s.
    2. Se descartan intervalos que se alejan más de `max_rel_change` (20 %) de la mediana local.
    Los intervalos descartados se eliminan (no se interpolan) para no inventar dinámica.
    """
    rr = np.asarray(rr, dtype=float)
    ok = (rr >= lo) & (rr <= hi)
    med = median_filter(np.where(ok, rr, np.median(rr[ok]) if ok.any() else 0.8), size=median_width, mode="nearest")
    ok &= np.abs(rr - med) <= max_rel_change * med
    return rr[ok], ok


def segment(rr: np.ndarray, valid: np.ndarray | None = None, window: int = 2000, max_windows: int = 6,
            max_bad_frac: float = 0.05, step: int | None = None) -> list[np.ndarray]:
    """Corta la serie original en ventanas de `window` latidos.

    Si se pasa `valid` (máscara de `clean_rr` sobre la serie ORIGINAL), una ventana se acepta
    solo si su fracción de intervalos descartados es <= `max_bad_frac`; de la ventana se
    devuelven únicamente los intervalos válidos. Se toman como máximo `max_windows`,
    espaciadas a lo largo del registro para cubrir distintas horas.
    """
    rr = np.asarray(rr, dtype=float)
    if valid is None:
        valid = np.ones(len(rr), bool)
    step = step or window
    starts = np.arange(0, len(rr) - window + 1, step)
    good = [s for s in starts if 1 - valid[s:s + window].mean() <= max_bad_frac]
    if len(good) > max_windows:
        idx = np.linspace(0, len(good) - 1, max_windows).round().astype(int)
        good = [good[i] for i in idx]
    return [rr[s:s + window][valid[s:s + window]] for s in good]
