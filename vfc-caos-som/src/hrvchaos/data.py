"""Extracción de intervalos R-R (NN) desde PhysioNet con `wfdb`.

Todas las bases usadas traen anotaciones de latido, así que no hace falta
detectar picos R: se leen los tiempos de los latidos anotados y se calculan
los intervalos entre latidos normales consecutivos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Base de PhysioNet -> (extensión de la anotación, grupo clínico)
DATABASES = {
    "nsr2db": {"ext": "ecg", "group": "RSN", "fs": 128},
    "chf2db": {"ext": "ecg", "group": "ICC", "fs": 128},
    "chfdb": {"ext": "ecg", "group": "ICC", "fs": 250},
    "nsrdb": {"ext": "atr", "group": "RSN", "fs": 128},
    "fantasia": {"ext": "ecg", "group": None, "fs": 250},  # grupo según nombre (y/o)
}

NORMAL_SYMBOLS = {"N"}


@dataclass
class RRRecord:
    database: str
    record: str
    group: str
    rr: np.ndarray  # intervalos NN en segundos
    n_beats_total: int
    n_normal_pairs: int


def group_for(database: str, record: str) -> str:
    """Grupo clínico de un registro. En Fantasia: 'f1y01' -> Joven, 'f1o01' -> Mayor."""
    if database == "fantasia":
        return "Joven" if record[2] == "y" else "Mayor"
    return DATABASES[database]["group"]


def list_records(database: str) -> list[str]:
    import wfdb

    return wfdb.get_record_list(database)


def load_rr(database: str, record: str, max_seconds: float | None = None) -> RRRecord:
    """Lee las anotaciones de un registro y devuelve los intervalos NN (s).

    Un intervalo cuenta solo si los dos latidos que lo delimitan son normales ('N').
    `max_seconds` limita cuánto se lee (útil para pruebas rápidas).
    """
    import wfdb

    info = DATABASES[database]
    sampto = int(max_seconds * info["fs"]) if max_seconds else "end"
    ann = wfdb.rdann(record, info["ext"], pn_dir=database, sampto=sampto)
    fs = ann.fs or info["fs"]

    samples = np.asarray(ann.sample, dtype=float)
    symbols = np.asarray(ann.symbol)
    is_beat = np.isin(symbols, list("NLRBAaJSVrFejnE/fQ?"))
    samples, symbols = samples[is_beat], symbols[is_beat]

    rr_all = np.diff(samples) / fs
    both_normal = np.isin(symbols[:-1], list(NORMAL_SYMBOLS)) & np.isin(symbols[1:], list(NORMAL_SYMBOLS))
    rr = rr_all[both_normal]
    return RRRecord(database, record, group_for(database, record), rr, len(samples), int(both_normal.sum()))


def load_database(database: str, cache_dir: str | Path | None = None, max_seconds: float | None = None,
                  verbose: bool = True) -> list[RRRecord]:
    """Carga todos los registros de una base, con caché en disco (.npy) si se indica `cache_dir`."""
    out = []
    cache = Path(cache_dir) / database if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    for rec in list_records(database):
        f = cache / f"{rec}.npy" if cache else None
        if f is not None and f.exists():
            rr = np.load(f)
            out.append(RRRecord(database, rec, group_for(database, rec), rr, -1, len(rr)))
            continue
        try:
            r = load_rr(database, rec, max_seconds=max_seconds)
        except Exception as e:  # registro sin anotación o error de red
            if verbose:
                print(f"  ! {database}/{rec}: {type(e).__name__}: {e}")
            continue
        if f is not None:
            np.save(f, r.rr)
        out.append(r)
        if verbose:
            print(f"  {database}/{rec}: {len(r.rr)} intervalos NN ({r.group})")
    return out


def records_summary(records: list[RRRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "database": r.database, "record": r.record, "group": r.group,
        "n_nn": len(r.rr), "hours": r.rr.sum() / 3600, "mean_rr_s": r.rr.mean() if len(r.rr) else np.nan,
    } for r in records])


# ---------------------------------------------------------------------------
# Datos sintéticos SOLO para probar el código sin conexión (no son fisiológicos)
# ---------------------------------------------------------------------------
def _pink_noise(n: int, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    f = np.fft.rfftfreq(n)
    spec = rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))
    f[0] = f[1]
    spec /= f ** (beta / 2)
    x = np.fft.irfft(spec, n)
    return (x - x.mean()) / x.std()


def synthetic_rr(group: str, n: int = 6000, seed: int = 0) -> np.ndarray:
    """Serie R-R de juguete: 'RSN' = ruido 1/f + arritmia sinusal respiratoria;
    'ICC' = variabilidad reducida y más ruido blanco. Solo para pruebas del pipeline."""
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    if group == "RSN":
        rr = 0.85 + 0.045 * _pink_noise(n, rng, 1.0) + 0.025 * np.sin(2 * np.pi * k / 4.3) + 0.008 * rng.normal(size=n)
    elif group == "ICC":
        rr = 0.68 + 0.010 * _pink_noise(n, rng, 1.8) + 0.006 * rng.normal(size=n)
    else:
        raise ValueError(group)
    return np.clip(rr, 0.35, 1.8)


def synthetic_cohort(n_per_group: int = 12, n_beats: int = 6000, seed: int = 42) -> list[RRRecord]:
    recs = []
    for gi, g in enumerate(["RSN", "ICC"]):
        for i in range(n_per_group):
            rr = synthetic_rr(g, n_beats, seed=seed + 1000 * gi + i)
            recs.append(RRRecord("sintetico", f"{g.lower()}{i:02d}", g, rr, n_beats + 1, n_beats))
    return recs
