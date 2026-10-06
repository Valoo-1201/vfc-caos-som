"""Figuras del proyecto (matplotlib)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

from .som import SOM

GROUP_COLORS = {"RSN": "#2a6fdb", "ICC": "#d1495b", "Joven": "#2a9d8f", "Mayor": "#e9a03b"}


def _hex_axes(ax, som: SOM):
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, som.cols + 0.3)
    ax.set_ylim(-0.8, (som.rows - 1) * np.sqrt(3) / 2 + 0.8)
    ax.invert_yaxis()
    ax.axis("off")


def hex_plot(ax, som: SOM, values: np.ndarray, cmap="viridis", title="", vmin=None, vmax=None, colorbar=True):
    values = np.asarray(values).ravel()
    norm = plt.Normalize(vmin if vmin is not None else np.nanmin(values),
                         vmax if vmax is not None else np.nanmax(values))
    cm = plt.get_cmap(cmap)
    for (x, y), v in zip(som.grid, values):
        ax.add_patch(RegularPolygon((x, y), 6, radius=0.577, orientation=0,
                                    facecolor=cm(norm(v)), edgecolor="white", lw=0.5))
    _hex_axes(ax, som)
    ax.set_title(title, fontsize=10)
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)


def plot_u_matrix_with_hits(som: SOM, X: np.ndarray, labels: np.ndarray, ax=None):
    """U-matriz de fondo + un punto por nodo con el color de la clase mayoritaria y tamaño ∝ impactos."""
    ax = ax or plt.subplots(figsize=(7, 6))[1]
    hex_plot(ax, som, som.u_matrix(), cmap="Greys", title="U-matriz y clase mayoritaria por nodo")
    hits = som.hit_map(X, labels)
    total = sum(hits.values()).ravel()
    for c, h in hits.items():
        h = h.ravel()
        mask = (h > 0) & (h == np.max(np.stack([hh.ravel() for hh in hits.values()]), axis=0))
        ax.scatter(som.grid[mask, 0], som.grid[mask, 1], s=30 + 200 * h[mask] / total.max(),
                   c=GROUP_COLORS.get(c, None), label=c, edgecolor="k", lw=0.4, zorder=3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1))
    return ax


def plot_component_planes(som: SOM, names: list[str], ncols: int = 4):
    planes = som.component_planes()
    n = len(names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    for i, ax in enumerate(np.ravel(axes)):
        if i < n:
            hex_plot(ax, som, planes[..., i], cmap="coolwarm", title=names[i])
        else:
            ax.axis("off")
    fig.suptitle("Planos de componentes (valores estandarizados)")
    fig.tight_layout()
    return fig


def plot_poincare(rr_by_group: dict, ax=None, max_points: int = 1500):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    for g, rr in rr_by_group.items():
        rr = np.asarray(rr)[:max_points] * 1000
        ax.scatter(rr[:-1], rr[1:], s=3, alpha=0.4, c=GROUP_COLORS.get(g), label=g)
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=0.6)
    ax.set_xlabel("RR$_n$ (ms)"); ax.set_ylabel("RR$_{n+1}$ (ms)")
    ax.set_title("Diagrama de Poincaré"); ax.legend(markerscale=4)
    return ax


def plot_mse(df, groups=("RSN", "ICC"), ax=None):
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    cols = [f"mse_{s}" for s in range(1, 11)]
    subj = df.groupby(["subject", "group"])[cols].median().reset_index()
    for g in groups:
        m = subj[subj.group == g][cols]
        mu, se = m.mean(), m.std() / np.sqrt(len(m))
        ax.errorbar(range(1, 11), mu, yerr=se, marker="o", capsize=3, label=f"{g} (n={len(m)})",
                    color=GROUP_COLORS.get(g))
    ax.set_xlabel("Escala τ"); ax.set_ylabel("SampEn"); ax.set_title("Entropía multiescala")
    ax.legend()
    return ax


def plot_feature_boxes(df, features, groups=("RSN", "ICC")):
    subj = df.groupby(["subject", "group"])[features].median().reset_index()
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    for ax, f in zip(np.ravel(axes), features):
        data = [subj.loc[subj.group == g, f].dropna() for g in groups]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        ax.set_xticks(range(1, len(groups) + 1), groups)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(GROUP_COLORS.get(g)); patch.set_alpha(0.5)
        ax.set_title(f, fontsize=10)
    for ax in np.ravel(axes)[len(features):]:
        ax.axis("off")
    fig.tight_layout()
    return fig
