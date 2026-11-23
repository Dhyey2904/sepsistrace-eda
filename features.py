"""Features: causal windows (rule L4) and the V / M / VM feature sets.

V = values, M = measurement pattern only, S = statics (included in every set).
`build_features()` is the single feature function shared by training and the app.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view

from config import Config
from data import VARIABLE_COLUMNS, load_labelled, load_splits, make_splits, run_all_checks

log = logging.getLogger(__name__)


# ==== causal windows ============================================================================

def causal_windows(X: np.ndarray, window_hours: int) -> np.ndarray:
    """(T, k) -> (T, W, k): element [t, j] is row t - (W-1) + j, NaN before the stay starts.
    Row t of the result only ever contains rows <= t (rule L4)."""
    T, k = X.shape
    pad = np.full((window_hours - 1, k), np.nan, dtype=np.float64)
    Xp = np.vstack([pad, X.astype(np.float64, copy=False)])
    return sliding_window_view(Xp, window_hours, axis=0).transpose(0, 2, 1)


def trailing_sum(A: np.ndarray, length: int, lag: int = 0) -> np.ndarray:
    """(T, k) -> (T, k): sum of rows t-lag-length+1 .. t-lag (rows before 0 count as 0)."""
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A, axis=0)])
    T = A.shape[0]
    hi = np.arange(T) - lag + 1
    lo = hi - length
    return cs[np.clip(hi, 0, T)] - cs[np.clip(lo, 0, T)]


# ==== feature sets ==============================================================================

FeatureSet = Literal["V", "M", "VM"]
V_STATS = ["mean", "std", "min", "max", "last", "slope"]
M_PER_VAR = ["count", "dt_log"]
M_GLOBAL = ["meas_rate", "meas_rate_delta", "hours_available", "n_vars_ever"]
KEYS = ["patient_id", "hour"]


# ------------------------------------------------------------------ column names
def value_vars(cfg: Config) -> list[str]:
    return list(cfg.data.vitals) + list(cfg.data.labs_keep)


def v_columns(cfg: Config) -> list[str]:
    return [f"{v}_{s}" for v in value_vars(cfg) for s in V_STATS]


def m_columns(cfg: Config) -> list[str]:
    return [f"{v}_{s}" for v in VARIABLE_COLUMNS for s in M_PER_VAR] + M_GLOBAL


def s_columns(cfg: Config) -> list[str]:
    return list(cfg.data.statics)


def label_columns(cfg: Config) -> list[str]:
    return [f"y_{h}" for h in cfg.data.horizons]


def feature_columns(cfg: Config, feature_set: FeatureSet) -> list[str]:
    cols = s_columns(cfg)
    if "V" in feature_set:
        cols += v_columns(cfg)
    if "M" in feature_set:
        cols += m_columns(cfg)
    return cols


# ------------------------------------------------------------- vectorised blocks
def _nan_stats(win: np.ndarray) -> tuple[np.ndarray, ...]:
    """win (T, W, k) -> n, mean, std(ddof=1), min, max over axis 1 ignoring NaN, without warnings."""
    ok = ~np.isnan(win)
    n = ok.sum(axis=1)
    z = np.where(ok, win, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = z.sum(axis=1) / n
        dev = np.where(ok, win - mean[:, None, :], 0.0)
        std = np.sqrt((dev ** 2).sum(axis=1) / (n - 1))
    std[n < 2] = np.nan
    mn = np.where(ok, win, np.inf).min(axis=1)
    mx = np.where(ok, win, -np.inf).max(axis=1)
    mn[n == 0] = np.nan
    mx[n == 0] = np.nan
    mean[n == 0] = np.nan
    return n, mean, std, mn, mx


def block_V(X: np.ndarray, window_hours: int) -> np.ndarray:
    """Values block. X (T, 16) raw values for one patient -> (T, 16*6).

    Per variable: mean, std, min, max, last over the window after forward-filling
    *within the window*; slope = OLS of value on hour over observed points only.
    """
    W = window_hours
    raw = causal_windows(X, W)                                  # (T, W, k)
    obs = ~np.isnan(raw)
    pos = np.arange(W)[None, :, None]
    idx = np.maximum.accumulate(np.where(obs, pos, -1), axis=1)
    ff = np.take_along_axis(raw, np.clip(idx, 0, None), axis=1)
    ff[idx < 0] = np.nan
    _, mean, std, mn, mx = _nan_stats(ff)
    last = ff[:, -1, :]

    n_o = obs.sum(axis=1)
    x = np.broadcast_to(pos, raw.shape).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(obs, x, 0).sum(axis=1) / n_o
        ym = np.where(obs, raw, 0).sum(axis=1) / n_o
        dx = np.where(obs, x - xm[:, None, :], 0)
        dy = np.where(obs, raw - ym[:, None, :], 0)
        sxx = (dx ** 2).sum(axis=1)
        slope = (dx * dy).sum(axis=1) / sxx
    slope[(n_o < 2) | (sxx == 0)] = np.nan
    out = np.stack([mean, std, mn, mx, last, slope], axis=2)    # (T, k, 6)
    return out.reshape(len(X), -1)


def block_M(X: np.ndarray, hours: np.ndarray, window_hours: int, half: int) -> np.ndarray:
    """Measurement block. X (T, 40) full history for one patient, hours (T,) 1-based.
    -> (T, 40*2 + 4). Uses only the NaN mask of X, never its values."""
    W = window_hours
    T, k = X.shape
    obs = (~np.isnan(X)).astype(np.float64)
    count = trailing_sum(obs, W)                                # (T, k)
    row = np.arange(T)
    last = np.maximum.accumulate(np.where(obs > 0, row[:, None], -1), axis=0)
    ever = last >= 0
    hrs = hours.astype(np.float64)[:, None]
    last_hour = np.where(ever, np.take(hours, np.clip(last, 0, None)).astype(np.float64), 0.0)
    dt = hrs - last_hour                                       # never observed -> hours since admission
    dt_log = np.log1p(dt)
    per_var = np.stack([count, dt_log], axis=2).reshape(T, -1)

    meas_rate = count.sum(axis=1) / (W * k)
    recent = trailing_sum(obs, half).sum(axis=1) / (half * k)
    prev = trailing_sum(obs, half, lag=half).sum(axis=1) / (half * k)
    hours_available = np.minimum(hours, W).astype(np.float64)
    n_vars_ever = ever.sum(axis=1).astype(np.float64)
    glob = np.column_stack([meas_rate, recent - prev, hours_available, n_vars_ever])
    return np.hstack([per_var, glob])


def block_S(S: np.ndarray) -> np.ndarray:
    """Statics at t: last observed value up to t (forward fill over the stay)."""
    return pd.DataFrame(S).ffill().to_numpy(dtype=np.float64)


# ------------------------------------------------------------ per-window wrappers


# ----------------------------------------------------------------------- assembly
def _patient_blocks(p: pd.DataFrame, cfg: Config, sets: tuple[str, ...]) -> dict[str, np.ndarray]:
    W = cfg.data.window_hours
    out = {"S": block_S(p[s_columns(cfg)].to_numpy(np.float64))}
    if "V" in sets:
        out["V"] = block_V(p[value_vars(cfg)].to_numpy(np.float64), W)
    if "M" in sets:
        out["M"] = block_M(p[VARIABLE_COLUMNS].to_numpy(np.float64), p["hour"].to_numpy(), W,
                           cfg.data.rate_half_window)
    return out


def build_features(patient_df: pd.DataFrame, cfg: Config, feature_set: FeatureSet,
                   all_hours: bool = False) -> pd.DataFrame:
    """One row per hour t >= min_history (every hour if all_hours). Statics included in
    ALL three sets so the V-vs-M comparison isolates only the blocks being tested.

    patient_df: one patient's hourly rows, consecutive hours starting at 1.
    Returns columns: patient_id, hour, <features>, plus y_h labels if present.
    """
    p = patient_df.sort_values("hour")
    hours = p["hour"].to_numpy()
    assert (np.diff(hours) == 1).all(), "hours must be consecutive"
    blocks = _patient_blocks(p, cfg, tuple(feature_set))
    parts = [blocks["S"]] + [blocks[b] for b in ("V", "M") if b in feature_set]
    X = np.hstack(parts).astype(np.float32)
    out = pd.DataFrame(X, columns=feature_columns(cfg, feature_set))
    out.insert(0, "hour", hours.astype(np.int16))
    out.insert(0, "patient_id", str(p["patient_id"].iloc[0]))
    for c in label_columns(cfg):
        if c in p:
            out[c] = p[c].to_numpy()
    if not all_hours:
        out = out[out["hour"] >= cfg.data.min_history_hours]
    return out.reset_index(drop=True)


def _build_chunk(chunk: list[pd.DataFrame], cfg: Config, sets: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    res: dict[str, list[pd.DataFrame]] = {s: [] for s in sets}
    for p in chunk:
        vm = build_features(p, cfg, "VM")
        for s in sets:
            res[s].append(vm[KEYS + feature_columns(cfg, s) + [c for c in label_columns(cfg) if c in vm]])
    return {s: pd.concat(v, ignore_index=True) for s, v in res.items()}


def build_dataset(df: pd.DataFrame, cfg: Config, sets: tuple[str, ...] = ("V", "M", "VM"),
                  n_jobs: int = 4, chunk_size: int = 500) -> dict[str, pd.DataFrame]:
    """Stream patient by patient (in parallel chunks) and return one frame per feature set."""
    groups = [g for _, g in df.groupby("patient_id", sort=True, observed=True)]
    chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]
    log.info("building features for %d patients in %d chunks", len(groups), len(chunks))
    parts = Parallel(n_jobs=n_jobs)(delayed(_build_chunk)(c, cfg, sets) for c in chunks)
    return {s: pd.concat([p[s] for p in parts], ignore_index=True) for s in sets}


# ==== step: features ============================================================================

def step_features(cfg: Config) -> None:
    """Task 2.4: build data/features/{V,M,VM}.parquet from the labelled frame."""
    df, counts = load_labelled(cfg)
    if cfg.quick.enabled:
        splits = make_splits(df.assign(SepsisLabel=df.is_septic.astype(int)), cfg)
    else:
        splits = load_splits(cfg)
    run_all_checks(df, splits, cfg.data.horizons)

    t0 = time.time()
    frames = build_dataset(df, cfg, n_jobs=6)
    out = cfg.path("features")
    out.mkdir(parents=True, exist_ok=True)
    for name, f in frames.items():
        f.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name:3s} rows={len(f):,} features={len(feature_columns(cfg, name))} cols={f.shape[1]}")
    print(f"built in {time.time() - t0:.0f}s")

    V, M, VM = frames["V"], frames["M"], frames["VM"]
    assert len(V) == len(M) == len(VM), "row counts differ across feature sets"
    assert (V[KEYS].values == M[KEYS].values).all() and (V[KEYS].values == VM[KEYS].values).all()
    shared = set(V.columns) & set(M.columns)
    assert shared == set(KEYS) | set(label_columns(cfg)) | set(s_columns(cfg)), f"unexpected overlap {shared}"
    run_all_checks(VM, splits, cfg.data.horizons)
    print("feature checks passed")
