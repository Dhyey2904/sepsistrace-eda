"""Models: LR / RF / XGB / MLP factory, training, metrics and calibration.

Every model is an sklearn Pipeline so preprocessing is fitted on the training split only (rule L2).
"""
from __future__ import annotations

import functools
import inspect
import itertools
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import config as C
from config import Config, load
from data import check_L1_disjoint_splits, check_L2_fit_on_train, check_L5_calibration_disjoint, load_splits
from features import KEYS, feature_columns

log = logging.getLogger(__name__)


# ==== model factory =============================================================================

MODEL_NAMES = ("lr", "rf", "xgb", "mlp")
NEEDS_SCALING = {"lr": True, "rf": False, "xgb": False, "mlp": True}
MLP_ACCEPTS_SAMPLE_WEIGHT = "sample_weight" in inspect.signature(MLPClassifier.fit).parameters


def make_model(name: str, pos_weight: float, seed: int, needs_scaling: bool | None = None,
               cfg: Config | None = None, n_jobs: int = -1) -> Pipeline:
    """name in {'lr','rf','xgb','mlp'}. Returns an sklearn Pipeline.

    lr/mlp: median imputation + standardisation inside the pipeline.
    rf/xgb: no preprocessing; both handle NaN natively.
    Imbalance: lr/rf class weights, xgb scale_pos_weight=pos_weight,
    mlp sample weights passed at fit time (see fit_params)."""
    cfg = cfg or load()
    p = cfg.models[name]
    if needs_scaling is None:
        needs_scaling = NEEDS_SCALING[name]
    if name == "lr":
        est = LogisticRegression(max_iter=p.max_iter, C=p.C, class_weight="balanced", random_state=seed)
    elif name == "rf":
        est = RandomForestClassifier(n_estimators=p.n_estimators, min_samples_leaf=p.min_samples_leaf,
                                     max_samples=p.get("max_samples"), class_weight="balanced_subsample",
                                     n_jobs=n_jobs, random_state=seed)
    elif name == "xgb":
        est = XGBClassifier(n_estimators=p.n_estimators, max_depth=p.max_depth, learning_rate=p.learning_rate,
                            subsample=p.subsample, colsample_bytree=p.colsample_bytree, eval_metric=p.eval_metric,
                            tree_method=p.tree_method, scale_pos_weight=pos_weight, random_state=seed,
                            n_jobs=n_jobs)
    elif name == "mlp":
        est = MLPClassifier(hidden_layer_sizes=tuple(p.hidden_layer_sizes), early_stopping=p.early_stopping,
                            max_iter=p.max_iter, random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    steps = []
    if needs_scaling:
        steps += [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                  ("scale", StandardScaler())]
    steps.append(("model", est))
    return Pipeline(steps)


def fit_params(name: str, y, pos_weight: float) -> dict:
    """Extra fit kwargs. For the MLP, positive rows get weight pos_weight."""
    if name == "mlp":
        if not MLP_ACCEPTS_SAMPLE_WEIGHT:
            raise RuntimeError("this scikit-learn has no MLP sample_weight; use oversampling (see train.py)")
        import numpy as np

        return {"model__sample_weight": np.where(np.asarray(y) == 1, pos_weight, 1.0)}
    return {}


# ==== metrics ===================================================================================

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising F1 on (y_true, y_prob)."""
    p, r, t = precision_recall_curve(y_true, y_prob)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0)[:-1]
    return float(t[int(np.nanargmax(f1))]) if len(t) else 0.5


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-frequency bins."""
    y_true, y_prob = np.asarray(y_true, float), np.asarray(y_prob, float)
    order = np.argsort(y_prob, kind="mergesort")
    bins = np.array_split(order, n_bins)
    n = len(y_prob)
    return float(sum(len(b) / n * abs(y_true[b].mean() - y_prob[b].mean()) for b in bins if len(b)))


def reliability(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per equal-frequency bin: mean predicted vs observed rate."""
    order = np.argsort(y_prob, kind="mergesort")
    rows = [(float(np.mean(y_prob[b])), float(np.mean(y_true[b])), len(b))
            for b in np.array_split(order, n_bins) if len(b)]
    return pd.DataFrame(rows, columns=["mean_pred", "frac_pos", "n"])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, positive_rate: float | None = None,
             threshold: float | None = None, n_bins: int = 15) -> dict:
    """auroc, auprc, auprc_lift (= auprc / positive_rate), brier, ece,
    precision/recall/f1 at the best-F1 threshold, and that threshold.

    If `threshold` is given (e.g. chosen on the validation split) it is used instead
    of the best-F1 threshold on these data, which would be optimistic."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if positive_rate is None:
        positive_rate = float(y_true.mean())
    auprc = float(average_precision_score(y_true, y_prob))
    thr = best_f1_threshold(y_true, y_prob) if threshold is None else float(threshold)
    pred = y_prob >= thr
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"auroc": float(roc_auc_score(y_true, y_prob)), "auprc": auprc,
            "auprc_lift": auprc / positive_rate if positive_rate > 0 else np.nan,
            "brier": float(brier_score_loss(y_true, y_prob)), "ece": ece(y_true, y_prob, n_bins),
            "precision": prec, "recall": rec, "f1": f1, "threshold": thr,
            "n_pos": int(y_true.sum()), "n_neg": int((1 - y_true).sum()), "positive_rate": positive_rate}


def clinical_metrics(df_pred: pd.DataFrame, onset: pd.Series, threshold: float) -> dict:
    """median_lead_time_hours, alerts_per_patient_day, false_alarms_per_true_alarm,
    patients_detected_fraction.

    df_pred: columns patient_id, hour, p (one row per scored hour, pre-onset only).
    onset:   patient_id -> onset hour (NaN / missing for non-septic).
    Lead time = onset - first hour with p >= threshold, over septic patients alerted
    before onset; detected fraction is reported alongside because lead time is
    conditional on detection. An alert is an hour with p >= threshold; it is a true
    alarm if the patient is septic, otherwise a false alarm.
    """
    d = df_pred[["patient_id", "hour", "p"]].copy()
    d["onset"] = d["patient_id"].map(onset)
    d["alert"] = d["p"] >= threshold
    septic = d["onset"].notna()
    first = d[d.alert & septic].groupby("patient_id")["hour"].min()
    lead = onset.reindex(first.index) - first
    n_septic = d.loc[septic, "patient_id"].nunique()
    patient_days = len(d) / 24.0
    true_alerts = int((d.alert & septic).sum())
    false_alerts = int((d.alert & ~septic).sum())
    return {
        "median_lead_time_hours": float(lead.median()) if len(lead) else np.nan,
        "patients_detected_fraction": len(first) / n_septic if n_septic else np.nan,
        "alerts_per_patient_day": float(d.alert.sum() / patient_days) if patient_days else np.nan,
        "false_alarms_per_patient_day": float(false_alerts / patient_days) if patient_days else np.nan,
        "false_alarms_per_true_alarm": false_alerts / true_alerts if true_alerts else np.nan,
        "nonseptic_patients_alerted_fraction": float(d[~septic & d.alert].patient_id.nunique()
                                                     / max(d[~septic].patient_id.nunique(), 1)),
        "threshold": float(threshold),
    }


def score_artifact(art: dict, cfg, split: str, calibrator=None) -> tuple[pd.DataFrame, dict]:
    """Predict one split with a saved artifact. Returns (keys + y + p frame, metrics).
    Precision/recall/F1 use the validation-chosen threshold stored in the artifact."""
    df = load_features(cfg, art["feature_set"])
    X, y, keys = get_xy(df, art["feature_cols"], art["horizon"], split)
    p = art["model"].predict_proba(X)[:, 1]
    if calibrator is not None:
        p = calibrator.predict(p)
    pred = keys.assign(y=y, p=p)
    return pred, evaluate(y, p, float(y.mean()), threshold=art["val_threshold"], n_bins=cfg.calibration.ece_bins)


# ==== training ==================================================================================

@functools.lru_cache(maxsize=3)
def _read_features(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def load_features(cfg: Config, feature_set: str) -> pd.DataFrame:
    """data/features/{set}.parquet with a `split` column attached from splits.json."""
    df = _read_features(str(cfg.path("features") / f"{feature_set}.parquet"))
    if "split" not in df:
        splits = load_splits(cfg)
        check_L1_disjoint_splits(splits)
        owner = {pid: s for s, ids in splits.items() for pid in ids}
        df["split"] = df["patient_id"].map(owner)
        assert df["split"].notna().all(), "L1: feature rows for patients outside every split"
    return df


def get_xy(df: pd.DataFrame, cols: list[str], horizon: int, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(X float32, y int8, keys frame) for one split."""
    part = df[df["split"] == split]
    return part[cols].to_numpy(np.float32), part[f"y_{horizon}"].to_numpy(np.int8), part[KEYS].reset_index(drop=True)


def subsample_negatives(X: np.ndarray, y: np.ndarray, max_rows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep every positive and a random subset of negatives so len <= max_rows."""
    if not max_rows or len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = np.sort(np.concatenate([pos, rng.choice(neg, size=max(max_rows - len(pos), 0), replace=False)]))
    log.info("subsampled training rows %d -> %d (all %d positives kept)", len(y), len(keep), len(pos))
    return X[keep], y[keep]


def fit_model(name: str, X: np.ndarray, y: np.ndarray, seed: int, cfg: Config, n_jobs: int | None = None):
    """Fit one registry model on (X, y); returns (pipeline, pos_weight, seconds)."""
    C.set_seed(seed)
    X, y = subsample_negatives(X, y, cfg.train.get("max_train_rows"), seed)
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = make_model(name, pos_weight, seed, cfg=cfg, n_jobs=cfg.train.n_jobs if n_jobs is None else n_jobs)
    t0 = time.time()
    if name == "mlp" and not MLP_ACCEPTS_SAMPLE_WEIGHT:
        X_fit, y_fit = oversample_positives(X, y, seed)
        log.warning("MLPClassifier has no sample_weight in this scikit-learn: random-oversampled positives "
                    "in the training split only (%d -> %d rows)", len(y), len(y_fit))
        model.fit(X_fit, y_fit)
    else:
        model.fit(X, y, **fit_params(name, y, pos_weight))
    secs = time.time() - t0
    if "impute" in model.named_steps:
        check_L2_fit_on_train(model, X)
    return model, pos_weight, secs


def oversample_positives(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate random positives until the classes are balanced (training split only)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    extra = rng.choice(pos, size=max(len(neg) - len(pos), 0), replace=True)
    idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
    return X[idx], y[idx]


def artifact_path(cfg: Config, model: str, feature_set: str, horizon: int, seed: int) -> Path:
    return cfg.path("artifacts") / f"{model}_{feature_set}_{horizon}h_s{seed}.joblib"


def train_one(feature_set: str, model_name: str, horizon: int, seed: int, cfg: Config,
              force: bool = False, n_jobs: int | None = None) -> Path:
    """Loads features, applies splits, trains, saves to
    artifacts/{model}_{set}_{horizon}h_s{seed}.joblib.
    Returns the path. Skips if the artifact already exists unless force."""
    path = artifact_path(cfg, model_name, feature_set, horizon, seed)
    if path.exists() and not force:
        log.info("exists, skipping: %s", path.name)
        return path
    df = load_features(cfg, feature_set)
    cols = feature_columns(cfg, feature_set)
    X_tr, y_tr, _ = get_xy(df, cols, horizon, "train")
    model, pos_weight, secs = fit_model(model_name, X_tr, y_tr, seed, cfg, n_jobs)
    X_va, y_va, _ = get_xy(df, cols, horizon, "val")
    thr = best_f1_threshold(y_va, model.predict_proba(X_va)[:, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": cols, "model_name": model_name, "feature_set": feature_set,
                 "horizon": horizon, "seed": seed, "pos_weight": pos_weight, "val_threshold": thr,
                 "train_seconds": secs, "n_train": int(len(y_tr)),
                 "train_patients": sorted(df.loc[df.split == "train", "patient_id"].unique())}, path)
    log.info("trained %s in %.1fs (pos_weight %.1f) -> %s", path.stem, secs, pos_weight, path.name)
    return path


def load_artifact(path: Path) -> dict:
    return joblib.load(path)


# ==== calibration on the held-out calib split (rule L5) =========================================

@dataclass
class Calibrator:
    """Monotone map from raw model probability to calibrated probability."""
    method: str
    model: object

    def predict(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return self.model.predict(p)
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return self.model.predict_proba(z[:, None])[:, 1]


# Calibrators saved before the flat layout were pickled as `models.calibrate.Calibrator`.
sys.modules.setdefault("models.calibrate", sys.modules[__name__])


def calibrate(model, X_calib: np.ndarray, y_calib: np.ndarray, method: str = "isotonic",
              min_positives: int = 50) -> Calibrator:
    """Fit on the calibration split only (rule L5).
    Fall back to 'sigmoid' if positives in the calib split < min_positives;
    log which was used."""
    p = model.predict_proba(X_calib)[:, 1]
    n_pos = int(np.sum(y_calib))
    if method == "isotonic" and n_pos < min_positives:
        log.warning("only %d positives in calib split: falling back to sigmoid", n_pos)
        method = "sigmoid"
    if method == "isotonic":
        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y_calib)
    else:
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        m = LogisticRegression(C=1e6).fit(z[:, None], y_calib)
    log.info("calibration method: %s (%d positives)", method, n_pos)
    return Calibrator(method, m)


def calib_path(art_path: Path) -> Path:
    return art_path.with_suffix(".calib.joblib")


def calibrate_artifact(path: Path, cfg: Config) -> tuple[dict, Calibrator]:
    """Fit + save the calibrator for one artifact; also store a calibrated val threshold."""
    art = load_artifact(path)
    df = load_features(cfg, art["feature_set"])
    calib_ids = df.loc[df.split == "calib", "patient_id"].unique()
    check_L5_calibration_disjoint(calib_ids, art["train_patients"])
    X_c, y_c, _ = get_xy(df, art["feature_cols"], art["horizon"], "calib")
    cal = calibrate(art["model"], X_c, y_c, cfg.calibration.method, cfg.calibration.min_positives_isotonic)
    X_v, y_v, _ = get_xy(df, art["feature_cols"], art["horizon"], "val")
    thr = best_f1_threshold(y_v, cal.predict(art["model"].predict_proba(X_v)[:, 1]))
    joblib.dump({"calibrator": cal, "val_threshold": thr}, calib_path(path))
    return art, cal


def load_calibrator(art_path: Path) -> dict | None:
    p = calib_path(art_path)
    return joblib.load(p) if p.exists() else None


def run_calibration(cfg: Config, seed: int) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    nb = cfg.calibration.ece_bins
    for m, s, h in itertools.product(cfg.models.names, cfg.models.feature_sets, cfg.data.horizons):
        path = artifact_path(cfg, m, s, h, seed)
        if not path.exists():
            log.warning("missing artifact %s", path.name)
            continue
        art, cal = calibrate_artifact(path, cfg)
        for split in ("test", "external"):
            pred, _ = score_artifact(art, cfg, split)
            y, p_raw = pred["y"].to_numpy(), pred["p"].to_numpy()
            p_cal = cal.predict(p_raw)
            rows.append({"model": m, "feature_set": s, "horizon": h, "seed": seed, "split": split,
                         "method": cal.method, "brier_before": float(np.mean((p_raw - y) ** 2)),
                         "brier_after": float(np.mean((p_cal - y) ** 2)), "ece_before": ece(y, p_raw, nb),
                         "ece_after": ece(y, p_cal, nb), "positive_rate": float(y.mean())})
            if m == cfg.demo.model and s == cfg.demo.feature_set:
                curves[(h, split)] = (reliability(y, p_raw, nb), reliability(y, p_cal, nb))
        log.info("calibrated %s", path.stem)
    return pd.DataFrame(rows), curves


def curves_frame(curves: dict) -> pd.DataFrame:
    rows = []
    for (h, split), (before, after) in curves.items():
        rows += [before.assign(horizon=h, split=split, stage="before"), after.assign(horizon=h, split=split, stage="after")]
    return pd.concat(rows, ignore_index=True)


def plot_reliability(rel: pd.DataFrame, cfg: Config) -> None:
    """rel: reliability.csv (horizon, split, stage, mean_pred, frac_pos, n)."""
    import plots as P

    curves = {(h, sp): (g[g.stage == "before"], g[g.stage == "after"]) for (h, sp), g in rel.groupby(["horizon", "split"])}
    horizons = sorted({h for h, _ in curves})
    fig, axes = P.plt.subplots(2, len(horizons), figsize=(3.6 * len(horizons), 7), squeeze=False)
    for j, h in enumerate(horizons):
        for i, split in enumerate(("test", "external")):
            ax = axes[i, j]
            before, after = curves[(h, split)]
            hi = max(after.mean_pred.max(), after.frac_pos.max(), before.frac_pos.max()) * 1.1
            ax.plot([0, hi], [0, hi], color=P.MUTED, ls="--", lw=1, label="perfect")
            ax.plot(before.mean_pred.clip(upper=hi), before.frac_pos, marker="o", ms=4, color=P.NONSEPTIC,
                    label="before (raw)")
            ax.plot(after.mean_pred, after.frac_pos, marker="o", ms=4, color=P.SEPTIC, label="after (isotonic)")
            ax.set(xlim=(0, hi), ylim=(0, hi), title=f"{h}h — {'internal test' if split == 'test' else 'hospital B'}")
            if j == 0:
                ax.set_ylabel("Observed positive rate")
            if i == 1:
                ax.set_xlabel("Mean predicted probability")
    axes[0, 0].legend()
    fig.suptitle(f"Reliability before and after calibration ({cfg.demo.model}, {cfg.demo.feature_set})",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    P.caption(fig, f"{cfg.calibration.ece_bins} equal-frequency bins. Axes are zoomed to the calibrated range; "
                   "raw class-weighted probabilities beyond it are clipped to the right edge.")
    P.save(fig, cfg.path("figures"), "fig11_reliability.png")


def step_calibrate(cfg: Config) -> None:
    """Post-hoc calibration on the held-out `calib` split (rule L5).

Calibrates every (model, feature_set, horizon) artifact for one seed, writes
    results/calibration.csv, saves *.calib.joblib next to each artifact, and draws fig11_reliability.png for the reference model (cfg.demo.model / feature_set)."""
    res, curves = run_calibration(cfg, cfg.seed)
    out = cfg.path("results")
    res.to_csv(out / "calibration.csv", index=False)
    rel = curves_frame(curves)
    rel.to_csv(out / "reliability.csv", index=False)
    plot_reliability(rel, cfg)
    t = res.groupby(["split", "model"])[["ece_before", "ece_after", "brier_before", "brier_after"]].mean()
    print(t.round(4).to_string())
    test = res[res.split == "test"]
    print(f"\nECE improved on internal test in {(test.ece_after < test.ece_before).sum()} of {len(test)} models")
    ext = res[res.split == "external"]
    print(f"ECE improved on hospital B in {(ext.ece_after < ext.ece_before).sum()} of {len(ext)} models")
