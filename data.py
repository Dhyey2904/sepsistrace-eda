"""Data: raw .psv loading, sepsis labels, patient splits and the leakage checks L1-L6."""
from __future__ import annotations

import itertools
import json
import logging
import time
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import Config

log = logging.getLogger(__name__)


# ==== loading ===================================================================================

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
STATICS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
VARIABLE_COLUMNS = VITALS + LABS + STATICS          # 40
LABEL = "SepsisLabel"
HOSPITAL_DIRS = {"A": "training_setA", "B": "training_setB"}


def _read_one(path: Path, hospital: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", dtype=np.float32)
    df[LABEL] = df[LABEL].astype(np.int8)
    df.insert(0, "hour", np.arange(1, len(df) + 1, dtype=np.int16))
    df.insert(0, "hospital", hospital)
    df.insert(0, "patient_id", path.stem)
    return df


def list_files(root: Path) -> list[tuple[Path, str]]:
    """All (psv path, hospital) pairs under root, sorted for determinism."""
    files: list[tuple[Path, str]] = []
    for hosp, sub in HOSPITAL_DIRS.items():
        d = Path(root) / sub
        if not d.exists():
            raise FileNotFoundError(f"missing raw directory {d}")
        files += [(p, hosp) for p in sorted(d.glob("*.psv"))]
    return files


def load_raw(root: Path, n_jobs: int = 4, limit: int | None = None) -> pd.DataFrame:
    """Read every .psv under root/training_setA and root/training_setB.

    Adds: patient_id (from filename stem), hospital ('A'|'B'),
          hour (1-based position within the stay).
    Downcasts float64 -> float32. Returns the concatenated frame,
    shape (n_rows, 3 + 41).
    """
    files = list_files(root)
    if limit is not None:
        files = files[:limit]
    log.info("reading %d files with n_jobs=%d", len(files), n_jobs)
    if n_jobs == 1:
        frames = [_read_one(p, h) for p, h in files]
    else:
        frames = Parallel(n_jobs=n_jobs, batch_size=256)(delayed(_read_one)(p, h) for p, h in files)
    df = pd.concat(frames, ignore_index=True)
    df["patient_id"] = df["patient_id"].astype("category")
    df["hospital"] = df["hospital"].astype("category")
    return df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_patients(cfg: Config) -> pd.DataFrame:
    """Read data/patients.parquet; in quick mode keep a fixed 2000-patient subset."""
    df = pd.read_parquet(cfg.path("parquet"))
    df["patient_id"] = df["patient_id"].astype(str)
    df["hospital"] = df["hospital"].astype(str)
    if cfg.quick.enabled:
        df = quick_subset(df, cfg.quick.n_patients, cfg.seed)
    return df


def quick_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministic n-patient subset, split evenly between hospitals."""
    rng = np.random.default_rng(seed)
    keep: list[str] = []
    for hosp in sorted(df["hospital"].unique()):
        ids = np.array(sorted(df.loc[df.hospital == hosp, "patient_id"].unique()))
        keep += list(rng.choice(ids, size=min(n // 2, len(ids)), replace=False))
    return df[df.patient_id.isin(set(keep))].reset_index(drop=True)


def load_labelled(cfg: Config) -> tuple[pd.DataFrame, dict]:
    """patients.parquet -> modelling frame (L6 exclusion, onset, y_h, L3 drop)."""
    return prepare(load_patients(cfg), cfg.data.horizons, cfg.data.label_shift_hours)


# ==== labels (SepsisLabel = 1 from onset - 6h onward) ===========================================

LABEL_SHIFT = 6


def derive_onset(df: pd.DataFrame, shift: int = LABEL_SHIFT) -> pd.Series:
    """patient_id -> onset hour. NaN for non-septic.
    onset = first hour with SepsisLabel == 1, plus 6."""
    pos = df.loc[df["SepsisLabel"] == 1, ["patient_id", "hour"]]
    first = pos.groupby("patient_id", observed=True)["hour"].min()
    all_ids = pd.Index(df["patient_id"].unique(), name="patient_id")
    onset = (first + shift).reindex(all_ids).astype("float32")
    onset.name = "onset"
    return onset


def attach_labels(df: pd.DataFrame, shift: int = LABEL_SHIFT) -> pd.DataFrame:
    """Adds: onset, is_septic, t_to_onset (hour - onset, negative = before)."""
    out = df.copy()
    onset = derive_onset(out, shift)
    out["onset"] = out["patient_id"].map(onset).astype("float32")
    out["is_septic"] = out["onset"].notna()
    out["t_to_onset"] = (out["hour"] - out["onset"]).astype("float32")
    return out


def horizon_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Adds y_1, y_3, y_6, y_12 (one per horizon).
    y_h(t) = 1 if 0 < onset - t <= h else 0. Non-septic rows are always 0."""
    out = df.copy()
    lead = out["onset"] - out["hour"]            # NaN for non-septic
    for h in horizons:
        out[f"y_{h}"] = ((lead > 0) & (lead <= h)).astype(np.int8)
    return out


def drop_post_onset(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Removes rows at/after onset for septic patients. Returns (df, n_dropped)."""
    mask = df["is_septic"] & (df["hour"] >= df["onset"])
    n = int(mask.sum())
    log.info("dropped %d post-onset rows (rule L3)", n)
    return df.loc[~mask].reset_index(drop=True), n


def septic_at_admission_ids(df: pd.DataFrame) -> set[str]:
    first = df.loc[df["hour"] == 1, ["patient_id", "SepsisLabel"]]
    return set(first.loc[first["SepsisLabel"] == 1, "patient_id"].astype(str))


def drop_septic_at_admission(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Removes patients whose SepsisLabel is 1 at hour 1 (rule L6).
    Returns (df, n_patients_dropped)."""
    ids = septic_at_admission_ids(df)
    log.info("excluded %d patients septic at admission (rule L6)", len(ids))
    return df.loc[~df["patient_id"].isin(ids)].reset_index(drop=True), len(ids)


def prepare(df: pd.DataFrame, horizons: list[int], shift: int = LABEL_SHIFT) -> tuple[pd.DataFrame, dict]:
    """Full label pipeline: L6 exclusion -> onset -> horizon labels -> L3 drop.
    Returns the modelling frame and a dict of counts for the stats card."""
    df, n_adm = drop_septic_at_admission(df)
    df = attach_labels(df, shift)
    df = horizon_labels(df, horizons)
    df, n_post = drop_post_onset(df)
    return df, {"septic_at_admission": n_adm, "post_onset_rows_dropped": n_post}


def label_shift_report(df: pd.DataFrame, n: int = 20, seed: int = 42) -> dict:
    """Empirical check of the 6-hour shift.

    For septic patients the run of 1s must be contiguous to the end of the
    record and, if records stop a few hours after onset, never longer than
    shift + a few hours. Returns summary statistics and n sample sequences.
    """
    pos = df[df["SepsisLabel"] == 1].groupby("patient_id", observed=True)["hour"].min()
    last = df.groupby("patient_id", observed=True)["hour"].max().reindex(pos.index)
    n_ones = df.groupby("patient_id", observed=True)["SepsisLabel"].sum().reindex(pos.index)
    tail = last - pos + 1
    contiguous = bool((n_ones == tail).all())
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.array(pos.index, dtype=str), size=min(n, len(pos)), replace=False)
    seqs = {}
    for pid in sample:
        x = df.loc[df["patient_id"] == pid].sort_values("hour")
        f = int(pos[pid])
        seqs[pid] = {"stay": int(len(x)), "first_label": f,
                     "seq_from_first_minus_3": "".join(map(str, x["SepsisLabel"].values[max(0, f - 4):]))}
    return {"n_septic": int(len(pos)), "labels_contiguous_to_end": contiguous,
            "label_run_length_median": float(tail.median()), "label_run_length_max": int(tail.max()),
            "label_run_length_counts": {int(k): int(v) for k, v in tail.value_counts().items()},
            "samples": seqs}


# ==== patient-level splits ======================================================================

INTERNAL_SPLITS = ["train", "val", "calib", "test"]
SPLIT_NAMES = INTERNAL_SPLITS + ["external"]


def make_splits(df: pd.DataFrame, cfg: Config, save: bool = True) -> dict[str, list[str]]:
    """Patient-level, stratified on patient-level septic flag.
    Hospital A -> train/val/calib/test by cfg ratios.
    Hospital B -> 'external' entirely.
    Deterministic given cfg.seed. Saves to splits.json."""
    pat = (df.groupby("patient_id", observed=True)
             .agg(hospital=("hospital", "first"), septic=("SepsisLabel", "max"))
             .reset_index())
    pat["patient_id"] = pat["patient_id"].astype(str)
    pat = pat.sort_values("patient_id").reset_index(drop=True)
    ratios = np.array([cfg.splits[s] for s in INTERNAL_SPLITS], dtype=float)
    ratios = ratios / ratios.sum()
    rng = np.random.default_rng(cfg.seed)

    splits: dict[str, list[str]] = {s: [] for s in SPLIT_NAMES}
    internal = pat[pat.hospital == cfg.splits.internal_hospital]
    for _, grp in internal.groupby("septic", sort=True):
        ids = grp["patient_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        cuts = np.round(np.cumsum(ratios)[:-1] * len(ids)).astype(int)
        for name, part in zip(INTERNAL_SPLITS, np.split(ids, cuts)):
            splits[name] += part.tolist()
    splits["external"] = pat.loc[pat.hospital == cfg.splits.external_hospital, "patient_id"].tolist()
    splits = {k: sorted(v) for k, v in splits.items()}

    if save:
        save_splits(splits, cfg.path("splits"))
    for k, v in splits.items():
        log.info("split %-8s %6d patients", k, len(v))
    return splits


def save_splits(splits: dict[str, list[str]], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(splits, fh, indent=1, sort_keys=True)
        fh.write("\n")


def load_splits(cfg: Config) -> dict[str, list[str]]:
    with open(cfg.path("splits")) as fh:
        return json.load(fh)


# ==== leakage checks L1-L6 ======================================================================

def check_L1_disjoint_splits(splits: dict[str, list[str]]) -> None:
    """L1: no patient id appears in two splits."""
    for a, b in itertools.combinations(splits, 2):
        both = set(splits[a]) & set(splits[b])
        assert not both, f"L1 violated: {len(both)} patients in both '{a}' and '{b}', e.g. {sorted(both)[:3]}"


def check_L1_frame_matches_splits(df: pd.DataFrame, splits: dict[str, list[str]],
                                  split_col: str = "split") -> None:
    """L1 (row level): every row's split column agrees with its patient's split."""
    if split_col not in df:
        return
    owner = {pid: s for s, ids in splits.items() for pid in ids}
    expected = df["patient_id"].astype(str).map(owner)
    bad = expected != df[split_col]
    assert not bad.any(), f"L1 violated: {int(bad.sum())} rows carry a split different from their patient's"


def check_L2_fit_on_train(pipeline, X_train: np.ndarray | pd.DataFrame) -> None:
    """L2: imputer / scaler statistics equal those of the training rows."""
    X = np.asarray(X_train, dtype=float)
    steps = dict(getattr(pipeline, "named_steps", {}))
    if "impute" in steps:
        stats = steps["impute"].statistics_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns (e.g. EtCO2 in hospital A)
            med = np.nanmedian(X, axis=0)
        ok = np.isclose(stats, np.nan_to_num(med), rtol=1e-4, atol=1e-5)
        assert ok.all(), f"L2 violated: imputer medians differ from train medians in {int((~ok).sum())} columns"
    if "scale" in steps:
        imputed = steps["impute"].transform(X) if "impute" in steps else X
        mean = imputed.mean(axis=0)
        ok = np.isclose(steps["scale"].mean_, mean, rtol=1e-3, atol=1e-4)
        assert ok.all(), f"L2 violated: scaler means differ from train means in {int((~ok).sum())} columns"


def check_L3_no_post_onset(df: pd.DataFrame) -> None:
    """L3: septic patients have no row with hour >= onset."""
    sep = df[df["is_septic"]]
    bad = sep[sep["hour"] >= sep["onset"]]
    assert bad.empty, (f"L3 violated: {len(bad)} rows at/after onset, "
                       f"patients {bad.patient_id.astype(str).unique()[:3].tolist()}")


def check_L5_calibration_disjoint(calib_ids: Iterable[str], train_ids: Iterable[str]) -> None:
    """L5: calibration patients were never used to train the model."""
    both = set(calib_ids) & set(train_ids)
    assert not both, f"L5 violated: {len(both)} calibration patients were in the training set"


def check_L6_no_septic_at_admission(df: pd.DataFrame) -> None:
    """L6: no patient with SepsisLabel == 1 at hour 1 remains."""
    first = df[df["hour"] == 1]
    bad = first[first["SepsisLabel"] == 1]
    assert bad.empty, (f"L6 violated: {len(bad)} patients septic at hour 1 remain, "
                       f"e.g. {bad.patient_id.astype(str).tolist()[:3]}")


def check_labels_nested(df: pd.DataFrame, horizons: list[int]) -> None:
    """y_1 <= y_3 <= y_6 <= y_12 everywhere; non-septic rows all zero."""
    hs = sorted(horizons)
    for a, b in zip(hs, hs[1:]):
        assert (df[f"y_{a}"] <= df[f"y_{b}"]).all(), f"labels not nested: y_{a} > y_{b} somewhere"
    if "is_septic" in df:
        non = df.loc[~df["is_septic"], [f"y_{h}" for h in hs]]
        assert (non.to_numpy() == 0).all(), "non-septic patient has a positive horizon label"


def run_all_checks(df: pd.DataFrame, splits: dict[str, list[str]] | None = None,
                   horizons: list[int] | None = None) -> None:
    """Run every frame-level rule in order. L2/L4/L5 are checked where they apply
    (pipeline fitting, windowing, calibration) because they need those objects."""
    if splits is not None:
        check_L1_disjoint_splits(splits)
        check_L1_frame_matches_splits(df, splits)
        known = set(itertools.chain.from_iterable(splits.values()))
        missing = set(df["patient_id"].astype(str).unique()) - known
        assert not missing, f"L1: {len(missing)} patients in the frame belong to no split"
    if "is_septic" in df:
        check_L3_no_post_onset(df)
    if "SepsisLabel" in df:
        check_L6_no_septic_at_admission(df)
    if horizons is not None:
        check_labels_nested(df, horizons)
    log.info("checks passed (L1, L3, L6%s)", ", labels" if horizons else "")


# ==== step: load ================================================================================

def step_load(cfg: Config) -> None:
    """Task 1.2: raw .psv files -> data/patients.parquet."""
    n_jobs = cfg.io.n_jobs

    t0 = time.time()
    df = load_raw(cfg.path("raw"), n_jobs=n_jobs)
    save_parquet(df, cfg.path("parquet"))
    t_load = time.time() - t0

    t0 = time.time()
    df2 = pd.read_parquet(cfg.path("parquet"))
    t_reload = time.time() - t0

    print(f"rows              {len(df):,}")
    print(f"patients          {df.patient_id.nunique():,}")
    print("patients/hospital", df.groupby("hospital", observed=True).patient_id.nunique().to_dict())
    print(f"columns           {df.shape[1]}")
    print(f"memory            {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"raw load          {t_load:.1f}s")
    print(f"parquet reload    {t_reload:.1f}s  ({len(df2):,} rows)")


# ==== step: labels ==============================================================================

def step_labels(cfg: Config) -> None:
    """Tasks 1.3-1.5: verify the label shift, derive labels, make splits, run checks."""
    raw = load_patients(cfg)
    rep = label_shift_report(raw, n=cfg.eda.n_label_check, seed=cfg.seed)
    print("== label shift check ==")
    for pid, s in rep["samples"].items():
        print(f"  {pid} stay={s['stay']:4d} first1={s['first_label']:4d}  {s['seq_from_first_minus_3']}")
    print(f"  labels contiguous to end of record: {rep['labels_contiguous_to_end']}")
    print(f"  run of 1s: median {rep['label_run_length_median']}, max {rep['label_run_length_max']}")
    assert rep["labels_contiguous_to_end"], "label runs are not contiguous - stop and investigate the shift"
    assert rep["label_run_length_max"] <= cfg.data.label_shift_hours + 4, "label runs longer than shift+4h - shift suspicious"

    df, counts = prepare(raw, cfg.data.horizons, cfg.data.label_shift_hours)
    print(f"== labels ==  septic at admission excluded: {counts['septic_at_admission']}, "
          f"post-onset rows dropped: {counts['post_onset_rows_dropped']}")
    rates = {h: float(df[f'y_{h}'].mean()) for h in cfg.data.horizons}
    for h, r in rates.items():
        print(f"  y_{h:<2d} positive hour rate {r:.4%}")
    assert all(a < b for a, b in zip(rates.values(), list(rates.values())[1:])), "rates not increasing in h"

    splits = make_splits(raw[raw.patient_id.isin(set(df.patient_id))], cfg)
    run_all_checks(df, splits, cfg.data.horizons)
    pat = df.groupby("patient_id").is_septic.first()
    print("== splits ==")
    for k, ids in splits.items():
        print(f"  {k:9s} {len(ids):6d} patients, septic rate {pat.reindex(ids).mean():.3%}")
    septic = [pat.reindex(splits[s]).mean() for s in INTERNAL_SPLITS]
    # Plan: within 1pp. With small splits (quick mode, ~100 patients) one patient already moves the
    # rate by ~1pp, so the tolerance is widened to 1.5 patients' worth of the smallest split.
    tol = max(0.01, 1.5 / min(len(splits[s]) for s in INTERNAL_SPLITS))
    assert max(septic) - min(septic) < tol, f"septic rate differs by >{tol:.1%} across internal splits"
    print(json.dumps({"label_run_length_counts": rep["label_run_length_counts"]}))
