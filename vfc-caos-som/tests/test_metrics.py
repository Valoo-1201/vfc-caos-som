"""Pruebas de cordura: las métricas deben reproducir comportamientos conocidos."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hrvchaos import metrics as M  # noqa: E402
from hrvchaos.som import SOM  # noqa: E402

rng = np.random.default_rng(0)


def brute_sampen(x, m, r):
    n = len(x)
    def count(mm):
        T = np.array([x[i:i + mm] for i in range(n - m)])
        c = 0
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                c += np.max(np.abs(T[i] - T[j])) <= r
        return c
    return -np.log(count(m + 1) / count(m))


def test_sampen_matches_brute_force():
    x = rng.normal(size=300)
    r = 0.2 * x.std()
    assert np.isclose(M.sample_entropy(x, 2, r=r), brute_sampen(x, 2, r))


def test_sampen_regular_lower_than_noise():
    k = np.arange(1500)
    assert M.sample_entropy(np.sin(k / 5)) < 0.5 * M.sample_entropy(rng.normal(size=1500))


def test_mse_white_noise_decreases():
    mse = M.multiscale_entropy(rng.normal(size=4000), scales=[1, 8])
    assert mse[1] < mse[0]


def test_dfa_white_noise_and_brownian():
    w = rng.normal(size=4000)
    assert abs(M.dfa_alpha(w, 16, 64) - 0.5) < 0.12
    assert abs(M.dfa_alpha(np.cumsum(w), 16, 64) - 1.5) < 0.15


def test_lyapunov_logistic_map_positive():
    x = np.empty(2000); x[0] = 0.4
    for i in range(1, len(x)):
        x[i] = 4 * x[i - 1] * (1 - x[i - 1])
    lam = M.lyapunov_rosenstein(x, m=2, tau=1, fit=(0, 4))
    assert 0.4 < lam < 0.9  # valor teórico ln 2 ≈ 0.693


def test_lyapunov_sine_near_zero():
    lam = M.lyapunov_rosenstein(np.sin(np.arange(2000) / 7), m=3)
    assert abs(lam) < 0.05


def test_poincare_identity():
    rr = 0.8 + 0.05 * rng.normal(size=1000)
    p = M.poincare(rr)
    sdnn = np.std(rr * 1000, ddof=1)
    assert np.isclose(p["sd1_ms"] ** 2 + p["sd2_ms"] ** 2, 2 * sdnn ** 2, rtol=1e-6)


def test_iaaft_preserves_distribution():
    x = rng.gamma(2, size=1024)
    s = M.iaaft(x, n_iter=50, seed=1)
    assert np.allclose(np.sort(s), np.sort(x))


def test_som_separates_two_blobs():
    X = np.vstack([rng.normal(-3, 1, (100, 3)), rng.normal(3, 1, (100, 3))])
    y = np.array(["a"] * 100 + ["b"] * 100)
    som = SOM(6, 6, 3, seed=0).fit(X, n_iter=3000)
    som.label_nodes(X, y)
    assert (som.predict(X) == y).mean() > 0.95
    assert som.topographic_error(X) < 0.2
