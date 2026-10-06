"""Construcción de la tabla de características y evaluación por sujeto."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from .metrics import window_features
from .preprocess import clean_rr, segment
from .som import SOM


def build_feature_table(records, window: int = 2000, max_windows: int = 6, verbose: bool = True,
                        **feat_kwargs) -> pd.DataFrame:
    rows = []
    for r in records:
        rr_c, ok = clean_rr(r.rr)
        wins = segment(r.rr, ok, window=window, max_windows=max_windows)
        for w_i, w in enumerate(wins):
            t0 = time.perf_counter()
            f = window_features(w, **feat_kwargs)
            f.update({"database": r.database, "subject": f"{r.database}/{r.record}", "group": r.group,
                      "window": w_i, "n_beats": len(w), "pct_discarded": 100 * (1 - ok.mean()),
                      "seconds_to_compute": time.perf_counter() - t0})
            rows.append(f)
        if verbose:
            print(f"{r.database}/{r.record} ({r.group}): {len(wins)} ventanas")
    return pd.DataFrame(rows)


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float((np.sign(a[:, None] - b[None, :])).mean())


def group_stats(df: pd.DataFrame, features: list[str], g1: str, g2: str) -> pd.DataFrame:
    """Mann-Whitney por sujeto (mediana de sus ventanas) con corrección de Holm y Cliff's δ."""
    subj = df.groupby(["subject", "group"])[features].median().reset_index()
    out = []
    for f in features:
        a = subj.loc[subj.group == g1, f].dropna()
        b = subj.loc[subj.group == g2, f].dropna()
        p = mannwhitneyu(a, b).pvalue
        out.append({"feature": f, f"median_{g1}": a.median(), f"median_{g2}": b.median(),
                    "p": p, "cliffs_delta": cliffs_delta(a, b)})
    res = pd.DataFrame(out).sort_values("p").reset_index(drop=True)
    k = len(res)
    adj = np.minimum(1, np.maximum.accumulate(res.p.values * (k - np.arange(k))))
    res["p_holm"] = adj
    return res


def _subject_level(df_test, y_win, score_win, positive):
    t = df_test[["subject", "group"]].copy()
    t["score"] = score_win
    s = t.groupby("subject").agg(group=("group", "first"), score=("score", "mean"))
    y = (s.group == positive).astype(int).values
    return y, s.score.values, s.index.values


def evaluate_som(df: pd.DataFrame, features: list[str], positive: str = "ICC", cv: str = "groupkfold",
                 n_splits: int = 5, n_iter: int = 5000, seed: int = 42) -> dict:
    """Validación agrupada por sujeto. Devuelve métricas a nivel sujeto para SOM y regresión logística."""
    d = df.dropna(subset=features).reset_index(drop=True)
    X_all, y_all, g_all = d[features].values, d.group.values, d.subject.values
    splitter = GroupKFold(n_splits=n_splits) if cv == "groupkfold" else LeaveOneGroupOut()
    ys, s_som, s_lr, qe, te = [], [], [], [], []
    for tr, te_idx in splitter.split(X_all, y_all, g_all):
        sc = StandardScaler().fit(X_all[tr])
        Xtr, Xte = sc.transform(X_all[tr]), sc.transform(X_all[te_idx])
        rows, cols = SOM.suggest_shape(len(Xtr), Xtr)
        som = SOM(rows, cols, Xtr.shape[1], seed=seed).fit(Xtr, n_iter=n_iter)
        som.node_scores(Xtr, y_all[tr], positive)
        qe.append(som.quantization_error(Xte)); te.append(som.topographic_error(Xte))
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y_all[tr] == positive)
        y_s, sc_som, _ = _subject_level(d.iloc[te_idx], None, som.predict_score(Xte), positive)
        _, sc_lr, _ = _subject_level(d.iloc[te_idx], None, lr.predict_proba(Xte)[:, 1], positive)
        ys.append(y_s); s_som.append(sc_som); s_lr.append(sc_lr)
    y = np.concatenate(ys)
    out = {"n_subjects": len(y), "QE": np.mean(qe), "TE": np.mean(te)}
    for name, s in [("SOM", np.concatenate(s_som)), ("LogReg", np.concatenate(s_lr))]:
        pred = (s >= 0.5).astype(int)
        out[f"{name}_AUC"] = roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan
        out[f"{name}_bal_acc"] = balanced_accuracy_score(y, pred)
        out[f"{name}_sens"] = pred[y == 1].mean()
        out[f"{name}_spec"] = 1 - pred[y == 0].mean()
    return out


def ablation(df: pd.DataFrame, feature_sets: dict, **kw) -> pd.DataFrame:
    return pd.DataFrame([{"set": k, "n_features": len(v), **evaluate_som(df, v, **kw)}
                         for k, v in feature_sets.items()])


def som_clusters(som: SOM, X: np.ndarray, k: int = 2, seed: int = 42) -> np.ndarray:
    """Agrupamiento en dos niveles (Vesanto & Alhoniemi, 2000): k-means sobre prototipos."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(som.W)
    som.node_clusters = km.labels_
    return km.labels_[som.bmu(X)]


def cluster_agreement(labels_true, labels_cluster) -> float:
    return adjusted_rand_score(labels_true, labels_cluster)
