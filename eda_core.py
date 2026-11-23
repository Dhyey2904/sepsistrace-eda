"""Framework-agnostic EDA helpers behind the EDA lab page (eda_lab.py).

Every function takes and returns pandas/numpy objects, so it runs without Streamlit.
Sections follow the course syllabus: sampling, import & cleaning, missing
data, descriptive statistics, univariate/bivariate analysis, correlation & regression, advanced EDA.
"""
from __future__ import annotations

import csv
import io
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from data import LABS, VITALS

TIME_VARYING = VITALS + LABS
# Generous clinical plausibility limits: values outside are entry or unit errors, not rare patients.
VALID_RANGES = {"HR": (25, 250), "O2Sat": (50, 100), "Temp": (30, 43), "SBP": (40, 280), "MAP": (30, 200),
                "DBP": (15, 180), "Resp": (2, 70), "EtCO2": (5, 100), "FiO2": (0.21, 1.0), "pH": (6.5, 8.0),
                "BaseExcess": (-40, 40), "HCO3": (2, 60), "Lactate": (0.1, 30), "Glucose": (20, 1500),
                "Potassium": (1.5, 10), "WBC": (0.1, 200), "Platelets": (1, 2000), "Hct": (5, 75), "Age": (14, 110)}
MOJIBAKE = ("Ã", "â€", "Â", "�")       # UTF-8 text decoded with the wrong codec


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def is_text(s: pd.Series) -> bool:
    """Text-like column: pandas 3 `str`, or `object` (which may also mix numbers and text)."""
    return not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def categorical_cols(df: pd.DataFrame, max_levels: int = 30) -> list[str]:
    """Text, boolean and low-cardinality numeric columns (e.g. Gender, Unit1)."""
    return [c for c in df.columns if df[c].nunique() <= max_levels or not pd.api.types.is_numeric_dtype(df[c])]


def cap_rows(df: pd.DataFrame, n: int = 200_000, seed: int = 42) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


# ============================================================ 1. data import
def detect_encoding(data: bytes) -> str:
    """UTF-8 (with or without BOM), else Windows-1252 (the usual spreadsheet export), else Latin-1,
    which decodes any byte string; the mojibake check then flags the damage."""
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    for enc in ("utf-8", "cp1252"):
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            pass
    return "latin-1"


def read_upload(name: str, data: bytes) -> tuple[pd.DataFrame, dict]:
    """CSV / TSV / PSV / TXT / JSON / Parquet / Feather / Excel bytes -> (frame, import notes)."""
    ext = name.rsplit(".", 1)[-1].lower()
    info = {"file": name, "format": ext, "bytes": len(data)}
    if ext == "parquet":
        return pd.read_parquet(io.BytesIO(data)), info
    if ext == "feather":
        return pd.read_feather(io.BytesIO(data)), info
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data)), info          # needs openpyxl / xlrd
    info["encoding"] = enc = detect_encoding(data)
    text = data.decode(enc, errors="replace")
    if ext == "json":
        return pd.read_json(io.StringIO(text)), info
    sep = {"tsv": "\t", "psv": "|"}.get(ext)
    if sep is None:
        try:
            sep = csv.Sniffer().sniff(text[:20_000], delimiters=",;|\t").delimiter
        except csv.Error:
            sep = ","
    info["delimiter"] = {"\t": "tab"}.get(sep, sep)
    return pd.read_csv(io.StringIO(text), sep=sep), info


def make_messy(df: pd.DataFrame, seed: int = 0, frac: float = 0.02) -> pd.DataFrame:
    """Inject the classic data-quality problems into a copy of a stay-level frame, so every cleaning
    check has something to find: duplicates, text-format variants, numbers stored as text, unit
    mix-ups, impossible values and mojibake."""
    rng = np.random.default_rng(seed)
    d = df.copy()
    n = len(d)
    pick = lambda f=frac: rng.random(n) < f  # noqa: E731
    if "hospital" in d:
        d["hospital"] = d["hospital"].astype(object)
        d.loc[pick(), "hospital"] = d["hospital"].map(lambda h: f" {str(h).lower()} ")
        d.loc[pick(0.005), "hospital"] = "HospitÃ¤l " + d["hospital"].astype(str).str.strip()
    if "Temp" in d:
        m = pick() & d["Temp"].notna()
        d.loc[m, "Temp"] = d.loc[m, "Temp"] * 9 / 5 + 32                       # °F mixed into °C
    if "HR" in d:
        d.loc[pick(0.002), "HR"] = 999                                            # sentinel code
    if "Age" in d:
        d["Age"] = d["Age"].astype(object)
        d.loc[pick(0.05), "Age"] = d["Age"].map(lambda a: f"{a:.0f} yrs")       # number stored as text
    return pd.concat([d, d.sample(frac=frac / 2, random_state=seed)]).reset_index(drop=True)


# ========================================================== 2. cleaning
def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, completeness, cardinality, range and an example value."""
    rows, nums = [], set(numeric_cols(df))
    for c in df.columns:
        s = df[c]
        num = c in nums
        first = s.dropna()
        rows.append({"column": c, "dtype": str(s.dtype), "non-null": int(s.notna().sum()),
                     "missing %": round(100 * s.isna().mean(), 2), "unique": int(s.nunique()),
                     "min": s.min() if num else None, "max": s.max() if num else None,
                     "example": str(first.iloc[0]) if len(first) else ""})
    return pd.DataFrame(rows)


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Mixed-type object columns (e.g. Age holding 64.0 and "64 yrs") as text, so Arrow can store them."""
    mixed = [c for c in df.columns if df[c].dtype == object]
    return df.astype({c: "string" for c in mixed}) if mixed else df


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    arrow_safe(df).to_parquet(buf, index=False)
    return buf.getvalue()


def key_columns(df: pd.DataFrame) -> list[str] | None:
    return ["patient_id", "hour"] if {"patient_id", "hour"} <= set(df.columns) else None


def text_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Per text column: padded values, case variants of the same label, numbers stored as text, mojibake."""
    rows = []
    for c in [c for c in df.columns if is_text(df[c])]:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        canon = s.str.strip().str.lower()
        variants = s.groupby(canon).nunique()
        rows.append({"column": c, "padded with spaces": int((s != s.str.strip()).sum()),
                     "labels with case/space variants": int((variants > 1).sum()),
                     "parseable as number %": round(100 * pd.to_numeric(s.str.extract(r"^\s*(-?[\d.]+)")[0],
                                                                         errors="coerce").notna().mean(), 1),
                     "mojibake": int(s.map(lambda v: any(m in v for m in MOJIBAKE)).sum()),
                     "levels": int(s.nunique())})
    return pd.DataFrame(rows)


def invalid_entries(df: pd.DataFrame, ranges: dict[str, tuple[float, float]] = VALID_RANGES) -> pd.DataFrame:
    rows = []
    for c, (lo, hi) in ranges.items():
        if c not in df:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        rows.append({"column": c, "valid range": f"{lo} – {hi}", "below": int((x < lo).sum()),
                     "above": int((x > hi).sum()), "observed min": x.min(), "observed max": x.max()})
    r = pd.DataFrame(rows)
    return r.assign(invalid=r.below + r.above).sort_values("invalid", ascending=False) if len(r) else r


def type_suggestions(df: pd.DataFrame) -> pd.DataFrame:
    """Columns whose storage type does not match their content."""
    rows = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        if nn.empty:
            continue
        if is_text(s):
            num = pd.to_numeric(nn.astype(str).str.extract(r"^\s*(-?[\d.]+)")[0], errors="coerce").notna().mean()
            if num >= 0.9:
                rows.append({"column": c, "now": str(s.dtype), "suggest": "float (strip units, coerce)",
                             "why": f"{num:.0%} of values are numbers"})
            elif nn.nunique() <= 50:
                rows.append({"column": c, "now": str(s.dtype), "suggest": "category",
                             "why": f"{nn.nunique()} distinct labels"})
        elif pd.api.types.is_float_dtype(s) and set(np.unique(nn.to_numpy()[:100_000])) <= {0.0, 1.0}:
            rows.append({"column": c, "now": str(s.dtype), "suggest": "boolean / category", "why": "only 0 and 1"})
    return pd.DataFrame(rows, columns=["column", "now", "suggest", "why"])


def clean(df: pd.DataFrame, opts: dict) -> tuple[pd.DataFrame, list[str]]:
    """Apply the selected fixes to a copy. Returns (clean frame, log of what changed)."""
    d, log = df.copy(), []
    if opts.get("drop_duplicates"):
        n = len(d)
        d = d.drop_duplicates()
        keys = key_columns(d)
        if keys:
            d = d.drop_duplicates(keys)
        log.append(f"removed {n - len(d):,} duplicate rows")
    for c in [c for c in d.columns if is_text(d[c]) and c != "patient_id"]:
        num = pd.to_numeric(d[c].astype(str).str.extract(r"^\s*(-?[\d.]+)")[0], errors="coerce")
        if num.notna().sum() >= 0.9 * d[c].notna().sum():           # a number column stored as text
            if opts.get("coerce_numeric"):
                d[c] = num.astype("float32")
                log.append(f"{c}: converted text to numbers (units stripped)")
        elif opts.get("fix_text"):
            s = d[c]
            fixed = s.where(s.isna(), s.astype(str).str.replace("HospitÃ¤l ", "", regex=False).str.strip().str.upper())
            n_fixed = int((fixed.astype(str) != s.astype(str)).sum())
            if n_fixed:
                log.append(f"{c}: normalised {n_fixed:,} text values (trim, case, mojibake)")
            d[c] = fixed
    if opts.get("fix_units"):
        if "Temp" in d:
            m = d["Temp"] > 50
            d.loc[m, "Temp"] = (d.loc[m, "Temp"] - 32) * 5 / 9
            log.append(f"Temp: converted {m.sum():,} °F readings to °C")
        if "FiO2" in d:
            m = (d["FiO2"] > 1) & (d["FiO2"] <= 100)
            d.loc[m, "FiO2"] = d.loc[m, "FiO2"] / 100
            log.append(f"FiO2: rescaled {m.sum():,} percentages to fractions")
    if opts.get("invalid_to_nan"):
        for c, (lo, hi) in VALID_RANGES.items():
            if c in d and c in numeric_cols(d):
                m = (d[c] < lo) | (d[c] > hi)
                d.loc[m, c] = np.nan
                if m.any():
                    log.append(f"{c}: set {m.sum():,} impossible values to missing")
    return d, log


TRANSFORMS = ["log(1+x)", "square root", "z-score", "min-max", "Yeo-Johnson", "quantile bins (5)"]


def transform(x: pd.Series, how: str) -> pd.Series:
    v = x.astype(float)
    if how == "log(1+x)":
        return np.log1p(v - min(v.min(), 0))
    if how == "square root":
        return np.sqrt(v - min(v.min(), 0))
    if how == "z-score":
        return (v - v.mean()) / v.std()
    if how == "min-max":
        return (v - v.min()) / (v.max() - v.min())
    if how == "Yeo-Johnson":
        out = pd.Series(np.nan, index=v.index)
        out[v.notna()] = stats.yeojohnson(v.dropna().to_numpy())[0]
        return out
    if how == "quantile bins (5)":
        return pd.qcut(v, 5, labels=False, duplicates="drop").astype(float)
    raise ValueError(how)


# ===================================================== 3. missing data
def missing_share(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].isna().mean().sort_values(ascending=False)


def nullity_correlation(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Correlation of missingness indicators: +1 = always missing together (missingno-style)."""
    m = df[cols].isna()
    m = m.loc[:, (m.mean() > 0) & (m.mean() < 1)]
    return m.astype(float).corr()


def missing_by(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    return df[cols].isna().groupby(df[by]).mean().T


def indicator_tests(df: pd.DataFrame, target: str, others: list[str]) -> pd.DataFrame:
    """For each other variable: does it differ between rows where `target` is missing and observed?
    Differences mean the missingness depends on observed data (evidence against MCAR, i.e. MAR)."""
    miss = df[target].isna()
    rows = []
    for c in others:
        a, b = df.loc[miss, c].dropna(), df.loc[~miss, c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"variable": c, f"mean | {target} missing": a.mean(), f"mean | {target} observed": b.mean(),
                     "SMD": (a.mean() - b.mean()) / sd if sd else 0.0,
                     "Welch p": stats.ttest_ind(a, b, equal_var=False).pvalue})
    r = pd.DataFrame(rows)
    return r.reindex(r.SMD.abs().sort_values(ascending=False).index) if len(r) else r


def missingness_auc(df: pd.DataFrame, target: str, others: list[str], seed: int = 42) -> float:
    """Cross-validated AUROC of predicting `target`'s missingness from the other variables
    (median-imputed, plus their own missing flags). ~0.5 is consistent with MCAR."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = cap_rows(df, 50_000, seed)
    y = d[target].isna().to_numpy()
    if y.all() or not y.any():
        return float("nan")
    X = pd.concat([d[others].fillna(d[others].median()), d[others].isna().add_suffix("_na")], axis=1)
    X = X.loc[:, X.std() > 0].to_numpy(float)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    return float(cross_val_score(clf, X, y, cv=3, scoring="roc_auc").mean())


def little_mcar(df: pd.DataFrame, cols: list[str], max_rows: int = 20_000, iters: int = 200,
                seed: int = 42) -> dict:
    """Little's (1988) MCAR test with EM estimates of the mean and covariance.

    d² = Σ_patterns n_j (ȳ_j − μ_j)ᵀ Σ_j⁻¹ (ȳ_j − μ_j) ~ χ²(Σ p_j − p) under MCAR."""
    X = cap_rows(df[cols].dropna(how="all"), max_rows, seed).to_numpy(float)
    n, p = X.shape
    obs = ~np.isnan(X)
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0) + 1e-9)
    patterns, inv = np.unique(obs, axis=0, return_inverse=True)
    inv = inv.ravel()
    for _ in range(iters):                       # EM for a multivariate normal with missing values
        Xh, C = np.where(obs, X, 0.0), np.zeros((p, p))
        for j, pat in enumerate(patterns):
            rows, o, m = inv == j, pat, ~pat
            if not m.any():
                continue
            if o.any():
                B = S[np.ix_(m, o)] @ np.linalg.pinv(S[np.ix_(o, o)])
                Xh[np.ix_(rows, m)] = mu[m] + (X[np.ix_(rows, o)] - mu[o]) @ B.T
                C[np.ix_(m, m)] += rows.sum() * (S[np.ix_(m, m)] - B @ S[np.ix_(o, m)])
            else:
                Xh[np.ix_(rows, m)] = mu[m]
                C[np.ix_(m, m)] += rows.sum() * S[np.ix_(m, m)]
        mu_new = Xh.mean(axis=0)
        S_new = ((Xh - mu_new).T @ (Xh - mu_new) + C) / n
        done = np.abs(mu_new - mu).max() < 1e-6 and np.abs(S_new - S).max() < 1e-6
        mu, S = mu_new, S_new
        if done:
            break
    d2, dof = 0.0, 0
    for j, o in enumerate(patterns):
        if not o.any():
            continue
        rows = inv == j
        diff = X[np.ix_(rows, o)].mean(axis=0) - mu[o]
        d2 += rows.sum() * diff @ np.linalg.pinv(S[np.ix_(o, o)]) @ diff
        dof += int(o.sum())
    dof -= p
    return {"d2": float(d2), "df": dof, "p": float(stats.chi2.sf(d2, dof)) if dof > 0 else float("nan"),
            "n": n, "patterns": len(patterns)}


def gap_vs_value(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Value of `var` by hours since its previous measurement in the same stay. If readings taken after
    long gaps are systematically different, measurement depended on the expected value (MNAR-like)."""
    d = df.loc[df[var].notna(), ["patient_id", "hour", var]].sort_values(["patient_id", "hour"])
    d["gap"] = d.groupby("patient_id")["hour"].diff()
    d["|z|"] = ((d[var] - d[var].mean()) / d[var].std()).abs()
    d["gap bin"] = pd.cut(d["gap"], [0, 1, 2, 4, 8, 24, np.inf], labels=["1h", "2h", "3–4h", "5–8h", "9–24h", ">24h"])
    return d.groupby("gap bin", observed=True).agg(n=(var, "size"), mean=(var, "mean"), mean_abs_z=("|z|", "mean")).reset_index()


IMPUTERS = ["Listwise deletion", "Mean", "Median", "Mode", "LOCF (carry forward in stay)", "LOCF + back-fill",
            "Linear interpolation (in stay)", "KNN (k = 5)", "Iterative (MICE-style)"]


def impute(df: pd.DataFrame, var: str, how: str, helpers: list[str] | None = None) -> pd.Series:
    """`var` after imputation (deletion returns the observed values only)."""
    x = df[var].astype(float)
    by = df["patient_id"] if "patient_id" in df else pd.Series(0, index=df.index)
    if how == "Listwise deletion":
        return x.dropna()
    if how == "Mean":
        return x.fillna(x.mean())
    if how == "Median":
        return x.fillna(x.median())
    if how == "Mode":
        return x.fillna(x.mode().iloc[0])
    if how.startswith("LOCF"):
        out = x.groupby(by).ffill()
        return out.groupby(by).bfill() if "back" in how else out
    if how.startswith("Linear"):              # vectorised: straight line between the readings either side
        pos = pd.Series(np.arange(len(x), dtype=float), index=x.index).where(x.notna())
        x0, x1 = x.groupby(by).ffill(), x.groupby(by).bfill()
        p0, p1 = pos.groupby(by).ffill(), pos.groupby(by).bfill()
        frac = ((np.arange(len(x)) - p0) / (p1 - p0)).fillna(0)
        return x.fillna(x0 + (x1 - x0) * frac)
    helpers = [h for h in (helpers or []) if h != var]
    X = cap_rows(df[[var] + helpers].astype(float), 5_000)      # model-based imputers: 5,000-row sample
    if how.startswith("KNN"):
        from sklearn.impute import KNNImputer

        return pd.Series(KNNImputer(n_neighbors=5).fit_transform(X)[:, 0], index=X.index)
    if how.startswith("Iterative"):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return pd.Series(IterativeImputer(random_state=0, max_iter=10).fit_transform(X)[:, 0], index=X.index)
    raise ValueError(how)


# ============================================== 4. descriptive statistics
def describe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Central tendency, variation, shape and the five-number summary per column."""
    out = {}
    for c in cols:
        x = df[c].dropna().astype(float)
        if x.empty:
            continue
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        mean, sd = x.mean(), x.std()
        out[c] = {"n": len(x), "missing": int(df[c].isna().sum()), "mean": mean, "median": med,
                  "mode": x.mode().iloc[0], "variance": x.var(), "std": sd, "IQR": q3 - q1,
                  "MAD": stats.median_abs_deviation(x), "CV %": 100 * sd / abs(mean) if mean else np.nan,
                  "range": x.max() - x.min(), "skewness": x.skew(), "kurtosis": x.kurt(),
                  "min": x.min(), "Q1": q1, "Q3": q3, "max": x.max(),
                  "outliers (1.5 IQR)": int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())}
    return pd.DataFrame(out).T


def insights(df: pd.DataFrame, max_items: int = 8) -> list[str]:
    """Plain-language headline facts about a frame."""
    num = numeric_cols(df)
    out = [f"{len(df):,} rows × {df.shape[1]} columns, {df.memory_usage(deep=True).sum() / 1e6:,.1f} MB in memory."]
    if "patient_id" in df:
        out.append(f"{df.patient_id.nunique():,} stays, median {df.groupby('patient_id').size().median():.0f} rows each.")
    if "is_septic" in df and "patient_id" in df:
        out.append(f"{df.groupby('patient_id').is_septic.first().mean():.1%} of stays develop sepsis "
                   "(heavy class imbalance).")
    miss = df.isna().mean()
    if (miss > 0.5).any():
        out.append(f"{(miss > 0.5).sum()} columns are more than half missing; worst: {miss.idxmax()} "
                   f"({miss.max():.1%}).")
    if "hour" in df and "ICULOS" in df:
        out.append(f"hour ≠ ICULOS in {(df.hour != df.ICULOS).mean():.1%} of rows: some records start after "
                   "ICU admission.")
    sk = df[num].skew(numeric_only=True).dropna()
    if len(sk) and sk.abs().max() > 2:
        out.append(f"Most skewed: {sk.abs().idxmax()} (skewness {sk[sk.abs().idxmax()]:.1f}); consider a log transform.")
    inv = invalid_entries(df)
    if len(inv) and inv.invalid.sum():
        out.append(f"{inv.invalid.sum():,} values are outside plausible ranges (most in {inv.iloc[0].column}).")
    if len(num) > 1:
        c = cap_rows(df[num], 50_000).corr().abs().where(lambda m: ~np.eye(len(m), dtype=bool)).stack()
        if len(c):
            (a, b), r = c.idxmax(), c.max()
            out.append(f"Strongest correlation: {a} ~ {b} (|r| = {r:.2f}).")
    return out[:max_items]


# ================================================ 5. bivariate analysis
def crosstab(df: pd.DataFrame, a: str, b: str, bins: int = 4) -> tuple[pd.DataFrame, dict]:
    """Contingency table (numeric columns binned by quantile) with a chi-square test and Cramér's V."""
    def as_cat(c):
        s = df[c]
        return pd.qcut(s, bins, duplicates="drop").astype(str) if c in numeric_cols(df) and s.nunique() > bins else s

    t = pd.crosstab(as_cat(a), as_cat(b))
    if t.shape[0] < 2 or t.shape[1] < 2:
        return t, {}
    chi2, p, dof, _ = stats.chi2_contingency(t)
    v = np.sqrt(chi2 / (t.to_numpy().sum() * (min(t.shape) - 1)))
    return t, {"chi2": chi2, "p": p, "dof": dof, "cramers_v": v}


# ====================================== 6. correlation and regression
def corr_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return cap_rows(df[cols]).corr(method=method)


def top_pairs(corr: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
    top = upper.reindex(upper.abs().sort_values(ascending=False).index)[:k]
    return pd.DataFrame({"pair": [f"{a} ~ {b}" for a, b in top.index], "r": top.to_numpy()})


def vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Variance inflation factors = diagonal of the inverse correlation matrix (complete cases)."""
    d = df[cols].dropna()
    d = d.loc[:, d.std() > 0]
    v = np.diag(np.linalg.pinv(d.corr().to_numpy()))
    return pd.DataFrame({"variable": d.columns, "VIF": v, "R² on the others": 1 - 1 / v}).sort_values(
        "VIF", ascending=False)


def ols(df: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Ordinary least squares on complete cases, with the usual inference and diagnostics."""
    d = df[[y] + xs].dropna().astype(float)
    Y = d[y].to_numpy()
    X = np.column_stack([np.ones(len(d)), d[xs].to_numpy()])
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    fitted = X @ beta
    resid = Y - fitted
    sse, sst = resid @ resid, ((Y - Y.mean()) ** 2).sum()
    s2 = sse / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    t = beta / se
    tcrit = stats.t.ppf(0.975, n - k)
    r2 = 1 - sse / sst
    f = (r2 / (k - 1)) / ((1 - r2) / (n - k)) if k > 1 else np.nan
    coef = pd.DataFrame({"term": ["intercept"] + xs, "coef": beta, "std err": se, "t": t,
                         "p": 2 * stats.t.sf(np.abs(t), n - k), "CI 2.5%": beta - tcrit * se,
                         "CI 97.5%": beta + tcrit * se})
    return {"coef": coef, "n": n, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k), "f": f,
            "f_p": stats.f.sf(f, k - 1, n - k) if k > 1 else np.nan, "rmse": np.sqrt(sse / n),
            "aic": n * np.log(sse / n) + 2 * k, "durbin_watson": np.sum(np.diff(resid) ** 2) / sse,
            "jarque_bera_p": stats.jarque_bera(resid).pvalue, "fitted": fitted, "resid": resid}


# ============================================================== 7. PCA
def pca(df: pd.DataFrame, cols: list[str], standardize: bool = True) -> dict:
    """PCA by SVD on median-imputed (and optionally standardised) columns (≤ 100,000 rows)."""
    X = cap_rows(df[cols], 100_000).astype(float)
    X = X.fillna(X.median())
    X = X.loc[:, X.std() > 0]
    Z = (X - X.mean()) / (X.std() if standardize else 1)
    U, s, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    var = s ** 2 / (len(Z) - 1)
    names = [f"PC{i + 1}" for i in range(len(s))]
    return {"explained": pd.DataFrame({"component": names, "explained": var / var.sum(),
                                       "cumulative": np.cumsum(var) / var.sum(), "eigenvalue": var}),
            "loadings": pd.DataFrame(Vt.T, index=X.columns, columns=names),
            "scores": pd.DataFrame(U * s, index=X.index, columns=names)}


# ===================================================== 8. time series
def series(df: pd.DataFrame, time_col: str, value_col: str, stat: str = "mean", by: str | None = None,
           t_range: tuple[float, float] | None = None) -> pd.DataFrame:
    """Aggregate `value_col` per time step: mean of observed values, or share of rows where it was measured."""
    d = df if t_range is None else df[df[time_col].between(*t_range)]
    v = d[value_col].notna().astype(float) if stat == "measured share" else d[value_col]
    keys = [d[time_col]] + ([d[by]] if by else [])
    out = v.groupby(keys, observed=True).agg(["mean", "count"]).reset_index()
    return out.rename(columns={time_col: "t", **({by: "group"} if by else {})})


def trend_test(t: np.ndarray, y: np.ndarray) -> dict:
    """Linear slope with its p-value, and the Mann-Kendall (Kendall τ vs time) monotonic-trend test."""
    ok = ~np.isnan(y)
    lr = stats.linregress(t[ok], y[ok])
    tau = stats.kendalltau(t[ok], y[ok])
    return {"slope": lr.slope, "slope_p": lr.pvalue, "r2": lr.rvalue ** 2, "kendall_tau": tau.statistic,
            "mk_p": tau.pvalue}


def acf(y: np.ndarray, nlags: int = 48) -> np.ndarray:
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy(float)
    y = y - y.mean()
    denom = (y ** 2).sum()
    return np.array([1.0] + [(y[:-k] * y[k:]).sum() / denom for k in range(1, min(nlags, len(y) - 1) + 1)])


def decompose(y: np.ndarray, period: int = 24) -> pd.DataFrame:
    """Classical additive decomposition: centred moving-average trend, mean seasonal profile, residual."""
    s = pd.Series(y, dtype=float).interpolate(limit_direction="both")
    if period % 2:
        trend = s.rolling(period, center=True).mean()
    else:                                           # 2×period moving average keeps an even window centred
        trend = s.rolling(period).mean().rolling(2).mean().shift(-(period // 2))
    detr = s - trend
    phase = np.arange(len(s)) % period
    prof = detr.groupby(phase).mean()
    seasonal = pd.Series(phase).map(prof - prof.mean()).to_numpy()
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal, "residual": s - trend - seasonal})


# ===================================================== 9. sampling
def sampling_experiment(y: np.ndarray, n: int, reps: int = 200, seed: int = 42, strata: np.ndarray | None = None,
                        clusters: np.ndarray | None = None, biased: np.ndarray | None = None) -> pd.DataFrame:
    """Repeated samples of size ~n under several designs; each estimates the population mean of y
    (NaNs ignored). Returns long format: design, rep, estimate."""
    rng = np.random.default_rng(seed)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)   # an all-missing draw -> NaN
    y = np.asarray(y, float)
    N = len(y)
    n = int(min(n, N))
    rows: list[tuple[str, int, float]] = []

    def run(name, draw):
        for r in range(reps):
            rows.append((name, r, float(draw())))

    run("Simple random", lambda: np.nanmean(y[rng.choice(N, n, replace=False)]))
    k = max(N // n, 1)
    run("Systematic (every k-th row)", lambda: np.nanmean(y[rng.integers(k)::k][:n]))
    if strata is not None:
        codes = pd.factorize(strata)[0]
        groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
        w = np.array([len(g) for g in groups]) / N
        alloc = np.maximum(1, np.round(n * w)).astype(int)

        def stratified():
            means = [np.nanmean(y[rng.choice(g, min(a, len(g)), replace=False)]) for g, a in zip(groups, alloc)]
            return np.nansum(np.array(means) * w) / w[~np.isnan(means)].sum()

        run("Stratified (proportional)", stratified)
    if clusters is not None:
        codes = pd.factorize(clusters)[0]
        ok = ~np.isnan(y)
        sums, cnt = np.bincount(codes, np.where(ok, y, 0)), np.bincount(codes, ok.astype(float))
        m = int(np.clip(round(n / np.bincount(codes).mean()), 1, len(sums)))
        run(f"Cluster ({m} whole clusters)", lambda: (lambda c: sums[c].sum() / max(cnt[c].sum(), 1))(
            rng.choice(len(sums), m, replace=False)))
    if biased is not None and biased.any():
        pool = np.flatnonzero(biased)
        run("Biased (convenience)", lambda: np.nanmean(y[rng.choice(pool, min(n, len(pool)), replace=False)]))
    return pd.DataFrame(rows, columns=["design", "rep", "estimate"])


def sampling_summary(est: pd.DataFrame, census: float) -> pd.DataFrame:
    g = est.groupby("design", sort=False).estimate
    out = pd.DataFrame({"mean estimate": g.mean(), "bias": g.mean() - census, "standard error": g.std(),
                        "RMSE": g.apply(lambda e: np.sqrt(np.mean((e - census) ** 2)))})
    out["design effect"] = out["standard error"] ** 2 / out.loc["Simple random", "standard error"] ** 2
    return out.reset_index()


def main_effects(grid: pd.DataFrame, metric: str, factors: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Design-of-experiments view of the model grid: mean response per factor level, and a one-way
    ANOVA F-test per factor."""
    eff = pd.concat([grid.groupby(f)[metric].mean().rename("mean").reset_index().rename(columns={f: "level"})
                     .assign(factor=f, level=lambda d: d.level.astype(str)) for f in factors])
    anova = []
    for f in factors:
        groups = [g[metric].to_numpy() for _, g in grid.groupby(f)]
        res = stats.f_oneway(*groups)
        anova.append({"factor": f, "levels": len(groups), "F": res.statistic, "p": res.pvalue,
                      "range of level means": np.ptp(eff.loc[eff.factor == f, "mean"].to_numpy())})
    return eff, pd.DataFrame(anova).sort_values("F", ascending=False)
