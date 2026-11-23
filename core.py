"""App logic for the Streamlit app (eda/). The Tkinter app (python/) has its own copy in python/backend/.

Every live prediction goes through features.build_features; no feature
logic lives here.
"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from features import build_features
from models import artifact_path, clinical_metrics, load_artifact, load_calibrator


class DemoDataMissing(RuntimeError):
    """Neither the full nor the quick demo artefacts have been built."""


def _demo_ready(cfg: C.Config) -> bool:
    d = _demo_dir(cfg)
    return all((d / f).exists() for f in ("demo_thresholds.json", "test_predictions.parquet")) and \
        all(cfg.path(k).exists() for k in ("demo_patients", "demo_predictions", "shap_cache"))


def _demo_dir(cfg: C.Config | None = None) -> Path:
    return (cfg or CFG).path("demo_patients").parent


def resolve_mode() -> tuple[C.Config, str]:
    """Pick the data the app shows.

    SEPSISTRACE_QUICK=1 forces the 2000-patient quick artefacts, SEPSISTRACE_QUICK=0 forces
    the full ones. Unset: full if built, else quick if built. Returns (cfg, mode) where
    mode is "full", "quick" (asked for) or "quick-fallback" (full missing).
    """
    flag = os.environ.get("SEPSISTRACE_QUICK")
    full, quick = C.load(), C.load(quick=True)
    if flag == "1":
        return quick, "quick" if _demo_ready(quick) else "missing"
    if _demo_ready(full):
        return full, "full"
    if flag == "0":
        return full, "missing"
    if _demo_ready(quick):
        return quick, "quick-fallback"
    return full, "missing"


CFG, MODE = resolve_mode()
QUICK = MODE.startswith("quick")
MODELS = list(CFG.models.names)
FEATURE_SETS = list(CFG.models.feature_sets)
HORIZONS = list(CFG.data.horizons)
WHATIF_VARS = ["HR", "MAP", "Temp", "Lactate"]
PANEL_VITALS = ["HR", "MAP", "Temp", "Resp"]


def combo(model: str, feature_set: str, horizon: int) -> str:
    return f"{model}|{feature_set}|{horizon}"


BUILD_HELP = """No demo data found. Build it with ONE of:

  quick (~6 min):   python main.py all --quick
  full  (~2.5 h):   python main.py all

then start the app again."""


def require_data() -> None:
    if MODE == "missing":
        raise DemoDataMissing(BUILD_HELP)


@functools.lru_cache(maxsize=1)
def load_demo() -> pd.DataFrame:
    """Hourly rows of the 30 demo patients (pre-onset), with category and onset."""
    require_data()
    return pd.read_parquet(CFG.path("demo_patients"))


@functools.lru_cache(maxsize=1)
def load_predictions() -> pd.DataFrame:
    return pd.read_parquet(CFG.path("demo_predictions"))


@functools.lru_cache(maxsize=1)
def load_test_predictions() -> pd.DataFrame:
    return pd.read_parquet(_demo_dir() / "test_predictions.parquet")


@functools.lru_cache(maxsize=1)
def load_thresholds() -> dict[str, float]:
    return json.loads((_demo_dir() / "demo_thresholds.json").read_text())


@functools.lru_cache(maxsize=1)
def load_shap_cache() -> pd.DataFrame:
    return pd.read_parquet(CFG.path("shap_cache"))


@functools.lru_cache(maxsize=48)
def load_model(model: str, feature_set: str, horizon: int) -> tuple[dict, object | None]:
    """(artifact dict, calibrator or None) for seed cfg.seed."""
    path = artifact_path(CFG, model, feature_set, horizon, CFG.seed)
    cal = load_calibrator(path)
    return load_artifact(path), (cal["calibrator"] if cal else None)


def patient(patient_id: str) -> pd.DataFrame:
    d = load_demo()
    return d[d.patient_id == patient_id].sort_values("hour").reset_index(drop=True)


def precomputed(patient_id: str, model: str, feature_set: str, horizon: int) -> pd.Series:
    """Stored calibrated predictions: hour -> p."""
    p = load_predictions()
    sel = p[(p.patient_id == patient_id) & (p.model == model) & (p.feature_set == feature_set) & (p.horizon == horizon)]
    return sel.set_index("hour")["p"].sort_index()


def predict_stay(patient_df: pd.DataFrame, model: str, feature_set: str, horizon: int,
                 calibrated: bool = True) -> pd.Series:
    """Live per-hour risk for one stay: hour -> p (hours >= min_history)."""
    art, cal = load_model(model, feature_set, horizon)
    f = build_features(patient_df, CFG, feature_set)
    p = art["model"].predict_proba(f[art["feature_cols"]].to_numpy(np.float32))[:, 1]
    if calibrated and cal is not None:
        p = cal.predict(p)
    return pd.Series(p, index=f["hour"].to_numpy(), name="p")


def predict_with_hidden(patient_df: pd.DataFrame, hide_vars: list[str], model: str, feature_set: str,
                        horizon: int, calibrated: bool = True) -> pd.Series:
    """As predict_stay, with the listed variables never measured (all NaN)."""
    p = patient_df.copy()
    if hide_vars:
        p[list(hide_vars)] = np.nan
    return predict_stay(p, model, feature_set, horizon, calibrated)


def predict_with_edits(patient_df: pd.DataFrame, edits: dict[str, float], hour: int, model: str,
                       feature_set: str, horizon: int, calibrated: bool = True) -> float:
    """Risk at `hour` after overwriting values at that hour (an edit to NaN = not measured)."""
    p = patient_df[patient_df.hour <= hour].copy()
    for var, val in edits.items():
        p.loc[p.hour == hour, var] = val
    s = predict_stay(p, model, feature_set, horizon, calibrated)
    return float(s.loc[hour]) if hour in s.index else float("nan")


def _shap_row(patient_id: str, hour: int, model: str, horizon: int) -> pd.Series | None:
    c = load_shap_cache()
    sel = c[(c.patient_id == patient_id) & (c.hour == hour) & (c.model == model) & (c.horizon == horizon)]
    return None if sel.empty else sel.iloc[0]


def contribution_for(patient_id: str, hour: int, model: str = "xgb", horizon: int = 6) -> tuple[float, float] | None:
    """(share_V, share_M) from the SHAP cache for the V+M tree model; None if not cached."""
    r = _shap_row(patient_id, hour, model, horizon)
    return None if r is None else (float(r.share_V), float(r.share_M))


def top_features_for(patient_id: str, hour: int, model: str = "xgb", horizon: int = 6) -> pd.DataFrame:
    r = _shap_row(patient_id, hour, model, horizon)
    rows = [] if r is None else [(r[f"top{k}_feature"], r[f"top{k}_phi"]) for k in range(1, 6)]
    return pd.DataFrame(rows, columns=["feature", "phi"])


@functools.lru_cache(maxsize=1)
def _test_onsets() -> pd.Series | None:
    """patient_id -> onset hour for septic test patients.

    Read from test_onsets.json (written by `python main.py demo`); older builds fall back to
    relabelling data/patients.parquet. None when neither exists, so the console can still show the
    hour-level numbers instead of crashing."""
    f = _demo_dir() / "test_onsets.json"
    if f.exists():
        return pd.Series(json.loads(f.read_text()), dtype=float)
    if not CFG.path("parquet").exists():
        return None
    from data import load_labelled

    lab, _ = load_labelled(CFG)
    o = lab[lab.is_septic].groupby("patient_id", observed=True).onset.first()
    o.index = o.index.astype(str)
    return o


def threshold_console(model: str, feature_set: str, horizon: int, threshold: float) -> dict:
    """Test-split operating point: hour-level confusion matrix plus clinical metrics.
    `onsets_available` is False (and the patient-level metrics NaN) if the onsets cannot be found."""
    tp = load_test_predictions()
    p, y = tp[combo(model, feature_set, horizon)].to_numpy(), tp[f"y_{horizon}"].to_numpy()
    alert = p >= threshold
    cm = {"TP": int((alert & (y == 1)).sum()), "FP": int((alert & (y == 0)).sum()),
          "FN": int((~alert & (y == 1)).sum()), "TN": int((~alert & (y == 0)).sum())}
    onsets = _test_onsets()
    if onsets is None:
        clin = dict.fromkeys(["patients_detected_fraction", "false_alarms_per_patient_day",
                              "median_lead_time_hours"], np.nan)
    else:
        pred = tp[["patient_id", "hour"]].assign(p=p)
        clin = clinical_metrics(pred, onsets.reindex(tp.patient_id.unique()).dropna(), threshold)
    sens = cm["TP"] / max(cm["TP"] + cm["FN"], 1)
    return {"confusion": cm, "hour_sensitivity": sens, "onsets_available": onsets is not None, **clin}


def default_threshold(model: str, feature_set: str, horizon: int) -> float:
    return float(load_thresholds()[combo(model, feature_set, horizon)])
