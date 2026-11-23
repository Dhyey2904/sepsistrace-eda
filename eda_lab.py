"""EDA lab page: the course's EDA syllabus, unit by unit, on the SepsisTRACE data or any uploaded table.

All computation lives in eda_core.py; this file only wires widgets and draws charts.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats

import core
import eda_core as E
from data import _read_one, attach_labels, list_files, load_patients
from plots import HOSP_A, HOSP_B, INK, MUTED, NONSEPTIC, SEPTIC

ss = st.session_state
PALETTE = [NONSEPTIC, SEPTIC, HOSP_A, HOSP_B, "#6A994E", "#8E6C8A", MUTED]
SECTIONS = ["1 · Sampling & experiment design", "2 · Import & cleaning", "3 · Missing data",
            "4 · Descriptive summary", "5 · Univariate & bivariate", "6 · Correlation & regression",
            "7 · Advanced EDA & dashboards"]
SOURCES = {"cohort": "PhysioNet cohort (data/patients.parquet)", "raw": "Raw .psv files (sample of stays)",
           "demo": "Demo patients (app data, 30 stays)", "upload": "Upload a file"}


# ===================================================================== data
@st.cache_resource(show_spinner="Loading the cohort…")
def cohort() -> pd.DataFrame | None:
    if not core.CFG.path("parquet").exists():
        return None
    return attach_labels(load_patients(core.CFG), core.CFG.data.label_shift_hours)


@st.cache_resource(show_spinner="Reading raw .psv files…")
def raw_files(n: int, seed: int) -> pd.DataFrame:
    files = list_files(core.CFG.path("raw"))
    pick = np.random.default_rng(seed).choice(len(files), min(n, len(files)), replace=False)
    df = pd.concat([_read_one(*files[i]) for i in sorted(pick)], ignore_index=True)
    return attach_labels(df.astype({"patient_id": str, "hospital": str}), core.CFG.data.label_shift_hours)


@st.cache_resource(show_spinner="Reading the upload…")
def uploaded(name: str, data: bytes) -> tuple[pd.DataFrame, dict]:
    return E.read_upload(name, data)


@st.cache_resource(show_spinner=False)
def stays_subset(n: int | str, seed: int) -> pd.DataFrame:
    df = cohort()
    all_ids = df.patient_id.unique()
    if n == "All" or int(n) >= len(all_ids):
        return df
    ids = np.random.default_rng(seed).choice(all_ids, int(n), replace=False)
    return df[df.patient_id.isin(set(ids))].reset_index(drop=True)


@st.cache_resource(show_spinner="Preparing the working table…")
def prepare(key: tuple, level: str, messy: bool, clean_opts: tuple, _base: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Working table = base (+ injected problems) (+ cleaning) (-> one row per stay). `key` names `_base`."""
    d = E.make_messy(_base) if messy else _base
    d, log = E.clean(d, dict(clean_opts)) if clean_opts else (d, [])
    return (stay_table(d) if level == "One row per stay" else d), log


def stay_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per stay: length, outcome, demographics, mean vitals and per-variable measurement rate."""
    g = df.groupby("patient_id", observed=True)
    out = g.agg(hospital=("hospital", "first"), stay_hours=("hour", "max"), age=("Age", "first"),
                gender=("Gender", "first"), septic=("is_septic", "first"))
    out = out.join(g[[v for v in E.VITALS if v in df]].mean().add_suffix("_mean"))
    rate = df[[v for v in E.TIME_VARYING if v in df]].notna().groupby(df.patient_id).mean()
    out = out.join(rate[["HR", "Temp", "WBC", "Lactate", "Glucose", "Creatinine"]].add_suffix("_meas_rate"))
    out["meas_per_hour"] = df[[v for v in E.TIME_VARYING if v in df]].notna().sum(axis=1).groupby(df.patient_id).mean()
    return out.reset_index()


@st.cache_data(show_spinner=False, max_entries=300)
def _calc(name: str, key: tuple, args: tuple, kwargs: tuple, _df: pd.DataFrame):
    return getattr(E, name)(_df, *args, **dict(kwargs))


def calc(fn, *args, **kwargs):
    """Memoised eda_core call on the working table; KEY identifies the table (source, sample, cleaning)."""
    return _calc(fn.__name__, KEY, args, tuple(kwargs.items()), _df=WORK)


# =================================================================== charts
def chart(c: alt.Chart, **kw) -> None:
    st.altair_chart(c, width="stretch", **kw)


def kde_frame(df: pd.DataFrame, col: str, group: str | None) -> pd.DataFrame:
    parts = []
    for name, x in (df.groupby(group, observed=True)[col] if group else [("all", df[col])]):
        x = x.dropna().astype(float)
        if len(x) < 5 or x.std() == 0:
            continue
        x = x.sample(min(len(x), 20_000), random_state=0)
        grid = np.linspace(*x.quantile([0.001, 0.999]), 200)
        parts.append(pd.DataFrame({col: grid, "density": stats.gaussian_kde(x)(grid), "group": str(name)}))
    return pd.concat(parts) if parts else pd.DataFrame(columns=[col, "density", "group"])


def color(group: str | None, title: str | None = None) -> alt.Color | alt.value:
    return alt.Color("group:N", title=title or group, scale=alt.Scale(range=PALETTE)) if group else alt.value(HOSP_A)


def histogram(df: pd.DataFrame, col: str, bins: int, group: str | None, share: bool) -> alt.Chart:
    x = df[col].astype(float)
    edges = np.histogram_bin_edges(x.dropna(), bins=bins)
    parts = []
    for name, s in (df.groupby(group, observed=True)[col] if group else [("all", x)]):
        cnt = np.histogram(s.dropna().astype(float), bins=edges)[0]
        parts.append(pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "count": cnt,
                                   "share": cnt / max(cnt.sum(), 1), "group": str(name)}))
    y = "share" if share else "count"
    return alt.Chart(pd.concat(parts)).mark_bar(opacity=0.6 if group else 0.9, binSpacing=0).encode(
        x=alt.X("lo:Q", title=col), x2="hi:Q", y=alt.Y(f"{y}:Q", stack=None, title="share within group" if share else "rows"),
        color=color(group), tooltip=["group", "lo", "hi", "count", alt.Tooltip("share:Q", format=".2%")]).properties(height=280)


def violin(df: pd.DataFrame, col: str, group: str | None) -> alt.Chart:
    d = kde_frame(df, col, group)
    levels = list(dict.fromkeys(d.group))
    d["pos"] = d.group.map({g: i for i, g in enumerate(levels)})
    d["w"] = 0.45 * d.density / d.groupby("group").density.transform("max")
    d["x0"], d["x1"] = d.pos - d.w, d.pos + d.w
    return alt.Chart(d).mark_area(orient="horizontal", opacity=0.75).encode(
        y=alt.Y(f"{col}:Q", title=col), x=alt.X("x0:Q", axis=alt.Axis(values=list(range(len(levels))),
                                                                     labelExpr=f"{levels}[datum.value]", title=group)),
        x2="x1:Q", color=color(group or "group"), detail="group:N").properties(height=300)


def heatmap(m: pd.DataFrame, title: str = "", fmt: str = ".2f", diverging: bool = True) -> alt.Chart:
    d = m.rename_axis(index="row", columns="col").stack().rename("value").reset_index()
    d["row"], d["col"] = d["row"].astype(str), d["col"].astype(str)
    order = [str(c) for c in m.columns]
    scale = alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True) if diverging else alt.Scale(scheme="blues")
    dark = "abs(datum.value) > 0.6" if diverging else f"datum.value > {0.6 * float(np.nanmax(m.to_numpy()))}"
    base = alt.Chart(d).encode(x=alt.X("col:N", sort=order, title=None),
                               y=alt.Y("row:N", sort=[str(r) for r in m.index], title=None, axis=alt.Axis(labelOverlap=False)))
    cells = base.mark_rect().encode(color=alt.Color("value:Q", scale=scale, title=title), tooltip=["row", "col",
                                                                                                    alt.Tooltip("value:Q", format=fmt)])
    text = base.mark_text(fontSize=9).encode(text=alt.Text("value:Q", format=fmt),
                                             color=alt.condition(dark, alt.value("white"), alt.value(INK)))
    return (cells + text if m.size <= 400 else cells).properties(height=max(240, 26 * len(m)))


def scatter(df: pd.DataFrame, x: str, y: str, group: str | None, fit: bool) -> alt.Chart:
    d = E.cap_rows(df[[x, y] + ([group] if group else [])].dropna(), 5_000).copy()
    if group:
        d["group"] = d[group].astype(str)
    pts = alt.Chart(d).mark_circle(size=14, opacity=0.45).encode(
        x=alt.X(f"{x}:Q", scale=alt.Scale(zero=False)), y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
        color=color(group), tooltip=[x, y])
    if fit:
        pts += pts.transform_regression(x, y).mark_line(color=INK, strokeWidth=2)
    return pts.properties(height=340)


def explain(text: str) -> None:
    st.caption(text)


# ================================================================= sections
def s_sampling() -> None:
    st.subheader("Census vs sample")
    explain("A **census** measures every unit of the population; a **sample** measures a subset and infers. Here the "
            "population is every row of the chosen source, so the census value is known exactly and each design can "
            "be judged by its **bias** (systematic error) and **standard error** (spread over repeated samples).")
    pop = POP
    nums = E.numeric_cols(pop)
    c1, c2, c3, c4 = st.columns(4)
    var = c1.selectbox("Variable", nums, index=nums.index("HR") if "HR" in nums else 0)
    stat = c2.radio("Estimate", ["mean of observed values", "share of rows where it was measured"])
    n = c3.select_slider("Sample size (rows)", [100, 250, 500, 1000, 2500, 5000, 10000], value=1000)
    reps = c4.select_slider("Repeated samples", [50, 100, 200, 500], value=200)
    cats = [c for c in E.categorical_cols(pop) if 1 < pop[c].nunique() <= 30]
    ids = [c for c in pop.columns if c not in nums and pop[c].nunique() > 30] or [None]
    c1, c2, c3 = st.columns(3)
    strata = c1.selectbox("Stratify by", cats or [None], index=cats.index("hospital") if "hospital" in cats else 0)
    clusters = c2.selectbox("Clusters (sample whole groups)", ids)
    biases = {"first n rows of the file (convenience)": None}
    if "Lactate" in pop:
        biases["only hours where Lactate was measured"] = pop.Lactate.notna().to_numpy()
    if "hospital" in pop:
        biases["only hospital A"] = (pop.hospital == "A").to_numpy()
    if "patient_id" in pop and "hour" in pop:
        biases["only stays ≥ 72 h"] = pop.groupby("patient_id").hour.transform("max").ge(72).to_numpy()
    bias = c3.selectbox("Biased design", list(biases))
    mask = biases[bias]
    if mask is None:
        mask = np.zeros(len(pop), bool)
        mask[:n] = True
    y = pop[var].notna().to_numpy(float) if stat.startswith("share") else pop[var].to_numpy(float)
    census = float(np.nanmean(y))
    est = sampling(KEY, var, stat, n, reps, strata, clusters, bias, _y=y, _pop=pop, _mask=mask)
    a, b = st.columns([1, 2])
    a.metric(f"Census value ({len(pop):,} rows)", f"{census:.4g}")
    summ = E.sampling_summary(est, census)
    a.dataframe(summ.set_index("design")[["bias", "standard error", "design effect"]].round(3))
    a.caption("Design effect = variance of the design ÷ variance of simple random sampling at the same n.")
    box = alt.Chart(est).mark_boxplot(extent="min-max", color=HOSP_A).encode(
        y=alt.Y("design:N", title=None, sort=list(summ.design), axis=alt.Axis(labelLimit=260)),
        x=alt.X("estimate:Q", scale=alt.Scale(zero=False)))
    rule = alt.Chart(pd.DataFrame({"x": [census]})).mark_rule(color=SEPTIC, strokeWidth=2).encode(x="x:Q")
    with b:
        chart((box + rule).properties(height=260, title=f"{reps} samples per design; red line = census"))
    explain("**Stratified** sampling fixes each stratum's share, so it is never worse than simple random and gains when "
            "strata differ. **Cluster** sampling is cheap in the field but, because rows within a stay are alike, its "
            "design effect is large. A **biased** design misses the census however large n grows: e.g. sampling only "
            "hours where a lab was drawn over-represents sicker moments — the same selection the project studies.")

    grid_path = core.CFG.path("results") / "grid.csv"
    st.subheader("Design of experiments: the model grid")
    if not grid_path.exists():
        st.info("Run `python main.py grid` to produce results/grid.csv.")
        return
    grid = pd.read_csv(grid_path)
    explain("The 144-model grid is a **full-factorial experiment**: factors model (4 levels) × feature set (3) × horizon "
            "(4), replicated over seeds, with a random patient-level split (**randomisation**) and hospital B held out "
            "(**blocking** by site). Main effects show how far the response moves across each factor's levels.")
    c1, c2 = st.columns(2)
    metric = c1.selectbox("Response", ["auprc_lift", "auroc", "auprc", "brier", "ece", "f1"])
    split = c2.radio("Evaluated on", sorted(grid.split.unique(), reverse=True), horizontal=True)
    g = grid[grid.split == split]
    eff, anova = E.main_effects(g, metric, ["model", "feature_set", "horizon"])
    left, right = st.columns([3, 2])
    left.altair_chart(alt.Chart(eff).mark_line(point=True, color=HOSP_A).encode(
        x=alt.X("level:N", title=None, sort=None), y=alt.Y("mean:Q", scale=alt.Scale(zero=False), title=metric)).properties(
        width=170, height=200).facet(column=alt.Column("factor:N", title=None)).resolve_scale(x="independent"))
    right.dataframe(anova.round(4), hide_index=True)
    inter = g.groupby(["model", "feature_set"])[metric].mean().reset_index()
    right.altair_chart(alt.Chart(inter).mark_line(point=True).encode(
        x=alt.X("feature_set:N", sort=["V", "M", "VM"]), y=alt.Y(f"{metric}:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", scale=alt.Scale(range=PALETTE))).properties(height=200, title="Interaction: model × feature set"),
        width="stretch")
    explain(f"{g.seed.nunique()} seeds give the **replication** that separates real effects from seed noise. "
            "Non-parallel lines in the interaction plot mean the best feature set depends on the model.")


@st.cache_data(show_spinner="Drawing repeated samples…", max_entries=50)
def sampling(key, var, stat, n, reps, strata, clusters, bias, _y, _pop, _mask) -> pd.DataFrame:
    return E.sampling_experiment(_y, n, reps, strata=_pop[strata].to_numpy() if strata else None,
                                 clusters=_pop[clusters].to_numpy() if clusters else None, biased=_mask)


def s_cleaning() -> None:
    st.subheader("Import")
    if SOURCE == "upload":
        st.json(IMPORT_INFO, expanded=False)
    st.code({"cohort": "pd.read_parquet('data/patients.parquet')           # columnar, typed, compressed",
             "raw": "pd.read_csv(path, sep='|')  # one pipe-separated file per stay, then pd.concat",
             "demo": "pd.read_parquet('data/demo_patients.parquet')",
             "upload": f"read_upload({IMPORT_INFO.get('file', '')!r})  # format from extension, "
                       "encoding + delimiter detected"}[SOURCE], language="python")
    # A widget's own key is dropped when the widget is not drawn, so the setting lives in a plain key.
    st.toggle("Practice mode: inject realistic data-quality problems (duplicates, padded/variant labels, °F, "
              "sentinel 999, numbers stored as text, mojibake)", value=ss.eda_messy, key="eda_messy_widget",
              on_change=lambda: ss.update(eda_messy=ss.eda_messy_widget))
    st.dataframe(E.arrow_safe(WORK.head(200)), height=220)

    st.subheader("Inspection and first insights")
    for line in calc(E.insights):
        st.markdown(f"- {line}")
    with st.expander("Column profile (type, completeness, cardinality, range)"):
        st.dataframe(calc(E.profile), hide_index=True)

    st.subheader("Data-quality checks")
    keys = E.key_columns(WORK)
    c1, c2, c3 = st.columns(3)
    c1.metric("Exact duplicate rows", f"{calc(E.duplicate_count):,}")
    if keys:
        c2.metric("Duplicate (patient_id, hour) keys", f"{int(WORK.duplicated(keys).sum()):,}")
    c3.metric("Text encoding", IMPORT_INFO.get("encoding", "binary (parquet)"))
    t1, t2, t3 = st.tabs(["Inconsistent formats", "Invalid entries", "Data types"])
    with t1:
        ti = calc(E.text_issues)
        if len(ti):
            st.dataframe(ti, hide_index=True)
        else:
            st.success("No text columns with format problems.")
        if "hospital" in WORK and E.is_text(WORK.hospital):
            st.caption("Raw labels in `hospital`: " + ", ".join(f"`{k!r}` ×{v:,}" for k, v in
                                                              WORK.hospital.value_counts().head(8).items()))
    with t2:
        inv = calc(E.invalid_entries)
        st.dataframe(inv, hide_index=True)
        explain("Ranges are clinical plausibility limits (eda_core.VALID_RANGES). In the real data FiO2 runs from −50 to "
                "4000: fractions mixed with percentages, a units inconsistency rather than a rare patient.")
    with t3:
        st.dataframe(calc(E.type_suggestions), hide_index=True)
        mem = WORK.memory_usage(deep=True).sum()
        st.caption(f"Memory now {mem / 1e6:,.1f} MB. Parquet keeps dtypes; CSV stores everything as text, which "
                   "is why the raw .psv import must convert types.")

    st.subheader("Clean")
    with st.form("clean"):
        cols = st.columns(5)
        opts = {k: cols[i].checkbox(lbl, value=True) for i, (k, lbl) in enumerate([
            ("drop_duplicates", "Drop duplicates"), ("fix_text", "Normalise text"),
            ("coerce_numeric", "Text → numbers"), ("fix_units", "Fix units (°F, FiO2 %)"),
            ("invalid_to_nan", "Impossible → missing")])}
        if st.form_submit_button("Apply to every section"):
            ss.eda_clean = tuple(sorted(opts.items()))
            st.rerun()
    if ss.get("eda_clean"):
        st.success("Cleaned data in use on every section:\n\n- " + "\n- ".join(CLEAN_LOG or ["nothing needed fixing"]))
        if st.button("Undo cleaning"):
            ss.eda_clean = ()
            st.rerun()
    st.download_button("Download the working table (.parquet)", calc(E.to_parquet_bytes), "eda_working_table.parquet")

    st.subheader("Transformation")
    nums = E.numeric_cols(WORK)
    c1, c2 = st.columns(2)
    var = c1.selectbox("Variable", nums, index=nums.index("Lactate") if "Lactate" in nums else 0, key="tr_var")
    how = c2.selectbox("Transform", E.TRANSFORMS)
    tr = E.transform(WORK[var], how)
    left, right = st.columns(2)
    left.altair_chart(histogram(WORK, var, 40, None, False).properties(title=f"{var}: skewness {WORK[var].skew():.2f}"),
                      width="stretch")
    right.altair_chart(histogram(pd.DataFrame({f"{how}({var})": tr}), f"{how}({var})", 40, None, False).properties(
        title=f"after {how}: skewness {tr.skew():.2f}"), width="stretch")
    explain("Log and Yeo-Johnson pull in long right tails (lab values); z-score and min-max only rescale, so shape and "
            "skewness are unchanged; binning trades detail for robustness to outliers.")


def s_missing() -> None:
    # onset / t_to_onset are labels, blank by definition for non-septic stays: not missing data.
    cols = [c for c in WORK.columns if WORK[c].isna().any() and c not in ("onset", "t_to_onset")]
    if not cols:
        st.success("No missing values in this table.")
        return
    st.subheader("How much, where and when")
    share = calc(E.missing_share, cols)
    left, right = st.columns(2)
    left.altair_chart(alt.Chart(share.rename_axis("column").reset_index(name="missing")).mark_bar(color=HOSP_A).encode(
        y=alt.Y("column:N", sort="-x", title=None), x=alt.X("missing:Q", axis=alt.Axis(format="%"), title="share missing"),
        tooltip=["column", alt.Tooltip("missing:Q", format=".1%")]).properties(height=max(200, 13 * len(share))),
        width="stretch")
    with right:
        keys = E.key_columns(WORK)
        sample = WORK.sort_values(keys).head(400) if keys else WORK.head(400)
        m = sample[cols].isna().reset_index(drop=True).rename_axis("row").reset_index().melt("row", var_name="column",
                                                                                          value_name="missing")
        chart(alt.Chart(m[m.missing]).mark_rect(color=INK).encode(
            x=alt.X("column:N", sort=list(share.index), axis=alt.Axis(labelAngle=-60), title=None),
            y=alt.Y("row:O", axis=None)).properties(height=max(200, 13 * len(share)),
                                                    title="Missingness matrix (black = missing, 400 rows)"))
    with st.expander("Nullity correlation: which variables go missing together"):
        nc = calc(E.nullity_correlation, cols)
        if len(nc) > 1:
            chart(heatmap(nc, "nullity r"))
        else:
            st.caption("Too few partially missing columns.")
        explain("Labs drawn in the same panel (e.g. BUN and Creatinine, Hct and Hgb) have nullity r near 1: their "
                "missingness is one decision, not independent gaps.")
    if "hour" in WORK and "is_septic" in WORK:
        vs = st.multiselect("Measured share by hour of stay", cols, default=[c for c in ["Lactate", "WBC", "Temp"] if c in cols])
        if vs:
            d = pd.concat([E.series(WORK, "hour", v, "measured share", "is_septic", (1, 72)).assign(variable=v) for v in vs])
            d["group"] = d.group.map({True: "septic", False: "non-septic"})
            chart(alt.Chart(d).mark_line().encode(x=alt.X("t:Q", title="hour of stay"),
                                                  y=alt.Y("mean:Q", title="share measured", axis=alt.Axis(format="%")),
                                                  color=alt.Color("group:N", scale=alt.Scale(range=[NONSEPTIC, SEPTIC])),
                                                  strokeDash="variable:N").properties(height=260))

    st.subheader("Mechanism: MCAR, MAR or MNAR?")
    explain("**MCAR**: missingness unrelated to anything. **MAR**: explained by *observed* variables. **MNAR**: depends on "
            "the unobserved value itself. MCAR is testable; MAR vs MNAR is not, so it is argued from evidence and design.")
    nums = E.numeric_cols(WORK)
    part = [c for c in nums if 0 < WORK[c].isna().mean() < 1]
    if not part:
        st.info("No numeric column is partially missing.")
        return
    c1, c2 = st.columns([1, 2])
    target = c1.selectbox("Variable with gaps", part, index=part.index("Lactate") if "Lactate" in part else 0)
    others = c2.multiselect("Compare against", [c for c in nums if c != target],
                            default=[c for c in ["HR", "MAP", "Temp", "Resp", "Age"] if c in nums and c != target][:5])
    if not others:
        return
    lt = calc(E.little_mcar, [target] + others)
    auc = calc(E.missingness_auc, target, others)
    ind = calc(E.indicator_tests, target, others)
    a, b, c = st.columns(3)
    a.metric("Little's MCAR test p", f"{lt['p']:.2g}", f"d² = {lt['d2']:.0f}, df = {lt['df']}", delta_color="off")
    b.metric(f"AUROC predicting '{target} missing'", f"{auc:.3f}")
    c.metric("Largest |SMD| (missing vs observed)", f"{ind.SMD.abs().max():.2f}" if len(ind) else "—")
    st.dataframe(ind.round(4), hide_index=True)
    verdict = ("consistent with **MCAR**" if lt["p"] > 0.05 and auc < 0.55 else
               "**not MCAR**: the observed variables predict where values are missing, so at least **MAR**")
    explain(f"Verdict for {target}: {verdict}. Little's test used {lt['n']:,} rows and {lt['patterns']} missingness "
            "patterns; with large n even tiny departures are significant, so read the effect sizes (SMD, AUROC) too.")
    if "patient_id" in WORK and "hour" in WORK:
        gv = calc(E.gap_vs_value, target)
        left, right = st.columns(2)
        left.altair_chart(alt.Chart(gv).mark_bar(color=SEPTIC).encode(
            x=alt.X("gap bin:N", sort=None, title="hours since previous measurement"),
            y=alt.Y("mean_abs_z:Q", title=f"mean |z| of {target}"), tooltip=["n", "mean"]).properties(height=220),
            width="stretch")
        right.caption(f"**MNAR evidence (informative missingness).** If {target} measured after short gaps is more "
                      "abnormal, clinicians re-measure *because* they expect an abnormal value: the gap depends on the "
                      "unobserved value. This is exactly the signal the project's measurement-only model (M) exploits.")

    st.subheader("Treatment")
    c1, c2 = st.columns([1, 2])
    var = c1.selectbox("Impute", part, index=part.index("Temp") if "Temp" in part else 0)
    methods = c2.multiselect("Methods", E.IMPUTERS, default=["Listwise deletion", "Mean", "LOCF (carry forward in stay)",
                                                             "Linear interpolation (in stay)", "KNN (k = 5)"])
    helpers = [c for c in ["HR", "MAP", "Resp", "O2Sat"] if c in nums and c != var]
    rows, dens = [], []
    for how in methods:
        x = calc(E.impute, var, how, helpers).dropna()
        rows.append({"method": how, "n": len(x), "mean": x.mean(), "std": x.std(), "skewness": x.skew()})
        dens.append(kde_frame(pd.DataFrame({var: x}), var, None).assign(group=how))
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    if dens:
        chart(alt.Chart(pd.concat(dens)).mark_line().encode(x=f"{var}:Q", y="density:Q", color=color("group", "method"))
              .properties(height=260))
    explain("Mean/median imputation keeps n but shrinks the standard deviation and spikes the density at one value. "
            "LOCF and interpolation respect each stay's own trajectory but cannot fill stays with no reading. KNN and "
            "iterative (MICE-style) borrow from correlated vitals (fitted on a 5,000-row sample).")


def s_descriptive() -> None:
    nums = E.numeric_cols(WORK)
    default = [c for c in ["HR", "MAP", "Temp", "Lactate", "Age", "stay_hours", "HR_mean", "age"] if c in nums][:5] or nums[:5]
    cols = st.multiselect("Variables", nums, default=default)
    if not cols:
        return
    d = calc(E.describe, cols)
    st.markdown("**Central tendency · variation · shape · five-number summary**")
    st.dataframe(d.round(3))
    explain("Mean vs median shows skew (Lactate: mean ≫ median). SD and variance are outlier-sensitive; IQR and MAD are "
            "robust. CV = SD / mean compares spread across units. Skewness > 0 = long right tail; kurtosis > 0 = "
            "heavier tails than normal.")
    sample = E.cap_rows(WORK[cols], 30_000)
    boxes = [alt.Chart(sample).mark_boxplot(color=HOSP_A, size=28).encode(y=alt.Y(f"{c}:Q", title=c, scale=alt.Scale(zero=False)))
             .properties(width=110, height=220) for c in cols]
    st.altair_chart(alt.hconcat(*boxes))
    cats = [c for c in E.categorical_cols(WORK) if WORK[c].nunique() <= 10]
    if cats:
        by = st.selectbox("Compare groups", cats, index=cats.index("is_septic") if "is_septic" in cats else
                          (cats.index("septic") if "septic" in cats else 0))
        st.dataframe(WORK.groupby(by, observed=True)[cols].agg(["mean", "median", "std"]).round(2))
    st.markdown("**Measures of relationship**")
    left, right = st.columns(2)
    left.caption("Covariance (units of x × y)")
    left.dataframe(WORK[cols].cov().round(2))
    right.caption("Pearson correlation (unit-free)")
    right.dataframe(WORK[cols].corr().round(3))


def s_univariate() -> None:
    nums, cats = E.numeric_cols(WORK), E.categorical_cols(WORK, 12)
    groups = [None] + [c for c in cats if WORK[c].nunique() <= 8]
    gdef = groups.index("is_septic") if "is_septic" in groups else (groups.index("septic") if "septic" in groups else 0)
    st.subheader("Univariate")
    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Plot", ["Histogram", "Density", "Boxplot", "Violin", "Bar chart"], horizontal=True)
    group = c3.selectbox("Split by", groups, index=gdef)
    if kind == "Bar chart":
        var = c2.selectbox("Categorical variable", cats)
        d = WORK.groupby([var] + ([group] if group and group != var else []), observed=True).size().rename("rows").reset_index()
        d[var] = d[var].astype(str)
        d["group"] = d[group].astype(str) if group and group != var else "all"
        chart(alt.Chart(d).mark_bar().encode(x=alt.X(f"{var}:N", sort="-y"), y="rows:Q", color=color(group if group != var else None),
                                             xOffset="group:N", tooltip=[var, "group", "rows"]).properties(height=300))
        return
    var = c2.selectbox("Numeric variable", nums, index=nums.index("HR") if "HR" in nums else 0)
    d = WORK[[var] + ([group] if group else [])].copy()
    if group:
        d["group"] = d[group].astype(str)
    if kind == "Histogram":
        a, b = st.columns(2)
        bins = a.slider("Bins", 5, 120, 40)
        share = b.toggle("Normalise within group", value=bool(group))
        chart(histogram(WORK, var, bins, group, share))
    elif kind == "Density":
        chart(alt.Chart(kde_frame(WORK, var, group)).mark_area(opacity=0.45, line=True).encode(
            x=f"{var}:Q", y="density:Q", color=color(group)).properties(height=300))
    elif kind == "Boxplot":
        enc = {"x": alt.X("group:N", title=group)} if group else {}
        chart(alt.Chart(E.cap_rows(d, 50_000)).mark_boxplot(size=40).encode(
            y=alt.Y(f"{var}:Q", scale=alt.Scale(zero=False)), color=color(group), **enc).properties(height=320))
    else:
        chart(violin(WORK, var, group))
    explain("Histogram: counts per bin (bin width changes the picture). Density: smoothed KDE. Boxplot: five-number "
            "summary + 1.5 IQR outliers. Violin: density mirrored, so multi-modality shows. Normalise within group "
            "when groups differ in size (septic stays are ~7%).")

    st.subheader("Bivariate")
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("x", nums, index=nums.index("SBP") if "SBP" in nums else 0)
    y = c2.selectbox("y", nums, index=nums.index("MAP") if "MAP" in nums else min(1, len(nums) - 1))
    fit = c3.toggle("Least-squares line", value=True)
    pair = WORK[[x, y]].dropna()
    a, b, c, e = st.columns(4)
    a.metric("Covariance", f"{pair[x].cov(pair[y]):.3g}")
    b.metric("Pearson r", f"{pair[x].corr(pair[y]):.3f}")
    c.metric("Spearman ρ", f"{pair[x].corr(pair[y], method='spearman'):.3f}")
    e.metric("Complete pairs", f"{len(pair):,}")
    left, right = st.columns(2)
    with left:
        chart(scatter(WORK, x, y, group, fit).properties(title="Scatter (≤ 5,000 points)"))
    with right:
        h = alt.Chart(E.cap_rows(pair, 100_000)).mark_rect().encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=40)), y=alt.Y(f"{y}:Q", bin=alt.Bin(maxbins=40)),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="blues", type="log"), title="rows"))
        chart(h.properties(height=340, title="2-D density heatmap (log colour)"))

    st.subheader("Crosstab")
    c1, c2, c3 = st.columns(3)
    a_ = c1.selectbox("Rows", cats + nums, index=(cats + nums).index("hospital") if "hospital" in cats else 0)
    b_ = c2.selectbox("Columns", cats + nums, index=(cats + nums).index("is_septic") if "is_septic" in cats else min(1, len(cats + nums) - 1))
    norm = c3.radio("Show", ["counts", "row %"], horizontal=True)
    t, test = E.crosstab(WORK, a_, b_)
    shown = t.div(t.sum(axis=1), axis=0) if norm == "row %" else t
    left, right = st.columns(2)
    left.dataframe(shown.round(3))
    if test:
        left.caption(f"χ² = {test['chi2']:.1f}, df = {test['dof']}, p = {test['p']:.2g}, Cramér's V = {test['cramers_v']:.3f} "
                     "(0 = independent, 1 = fully associated). Numeric variables are split into quartiles.")
    with right:
        chart(heatmap(shown.astype(float), norm, ".2f" if norm == "row %" else ".0f", diverging=False))


def s_correlation() -> None:
    nums = E.numeric_cols(WORK)
    default = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age", "Lactate", "WBC", "Hct", "Hgb"] if c in nums] \
        or nums[:8]
    cols = st.multiselect("Variables", nums, default=default)
    if len(cols) < 2:
        return
    method = st.radio("Correlation", ["pearson", "spearman"], horizontal=True,
                      format_func=lambda m: {"pearson": "Pearson (linear)", "spearman": "Spearman (rank, monotonic)"}[m])
    corr = calc(E.corr_matrix, cols, method)
    left, right = st.columns([3, 2])
    with left:
        chart(heatmap(corr, "r"))
    right.markdown("**Strongest pairs**")
    right.dataframe(E.top_pairs(corr).round(3), hide_index=True)
    other = calc(E.corr_matrix, cols, "spearman" if method == "pearson" else "pearson")
    gap = E.top_pairs(other - corr, 3)
    right.caption("Largest Pearson–Spearman gaps (non-linear or outlier-driven relations): "
                  + ", ".join(f"{p} ({v:+.2f})" for p, v in zip(gap.pair, gap.r)))

    st.subheader("Multicollinearity")
    v = calc(E.vif, cols)
    st.dataframe(v.round(2), hide_index=True)
    explain("VIF = 1 / (1 − R²) of each variable regressed on the others. VIF > 5–10 means its coefficient is unstable "
            "in a multiple regression: MAP is almost a weighted sum of SBP and DBP.")

    st.subheader("Regression and goodness of fit")
    c1, c2 = st.columns([1, 2])
    y = c1.selectbox("Response y", nums, index=nums.index("MAP") if "MAP" in nums else 0)
    xs = c2.multiselect("Predictors x", [c for c in nums if c != y],
                        default=[c for c in ["SBP", "DBP", "HR"] if c in nums and c != y][:3] or [c for c in nums if c != y][:1])
    if not xs:
        return
    r = calc(E.ols, y, xs)
    a, b, c, d, e = st.columns(5)
    a.metric("R²", f"{r['r2']:.3f}")
    b.metric("Adjusted R²", f"{r['adj_r2']:.3f}")
    c.metric("F-test p", f"{r['f_p']:.2g}")
    d.metric("RMSE", f"{r['rmse']:.3g}")
    e.metric("n (complete cases)", f"{r['n']:,}")
    st.dataframe(r["coef"].round(4), hide_index=True)
    res = E.cap_rows(pd.DataFrame({"fitted": r["fitted"], "residual": r["resid"]}), 4_000)
    q = np.sort(res.residual.to_numpy())
    qq = pd.DataFrame({"theoretical": stats.norm.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q)), "sample": q / q.std()})
    left, mid, right = st.columns(3)
    if len(xs) == 1:
        with left:
            chart(scatter(WORK, xs[0], y, None, True).properties(height=240, title="Simple linear fit"))
    mid.altair_chart(alt.Chart(res).mark_circle(size=10, opacity=0.4, color=HOSP_A).encode(x="fitted:Q", y="residual:Q")
                     .properties(height=240, title="Residuals vs fitted"), width="stretch")
    right.altair_chart(alt.Chart(qq).mark_circle(size=10, color=SEPTIC).encode(x="theoretical:Q", y="sample:Q")
                       .properties(height=240, title="Normal Q-Q of residuals"), width="stretch")
    explain(f"AIC {r['aic']:.0f} (lower = better trade-off of fit and size). Durbin-Watson {r['durbin_watson']:.2f} "
            f"(≈ 2 = no autocorrelation; hourly rows of one stay are autocorrelated). Jarque-Bera p {r['jarque_bera_p']:.2g} "
            "tests residual normality. A funnel in residuals vs fitted means heteroscedasticity.")


def s_advanced() -> None:
    nums = E.numeric_cols(WORK)
    cats = [c for c in E.categorical_cols(WORK, 8) if c in WORK]
    st.subheader("Principal component analysis")
    default = [c for c in ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age", "HR_mean", "MAP_mean", "Resp_mean",
                           "stay_hours", "meas_per_hour"] if c in nums] or nums[:6]
    c1, c2, c3 = st.columns([3, 1, 1])
    cols = c1.multiselect("Variables", nums, default=default, key="pca_cols")
    std = c2.toggle("Standardise", value=True)
    group = c3.selectbox("Colour by", [None] + cats, index=1 + cats.index("is_septic") if "is_septic" in cats else 0)
    if len(cols) >= 2:
        p = calc(E.pca, cols, std)
        left, mid, right = st.columns(3)
        ex = p["explained"]
        left.altair_chart((alt.Chart(ex).mark_bar(color=HOSP_A).encode(x=alt.X("component:N", sort=None), y="explained:Q")
                           + alt.Chart(ex).mark_line(point=True, color=SEPTIC).encode(x=alt.X("component:N", sort=None),
                                                                                      y="cumulative:Q"))
                          .properties(height=260, title="Scree plot (bar) and cumulative (line)"), width="stretch")
        with mid:
            chart(heatmap(p["loadings"].iloc[:, :3], "loading").properties(title="Loadings, PC1–PC3"))
        sc = p["scores"].iloc[:, :2].join(WORK[[group]] if group else pd.DataFrame(index=p["scores"].index))
        sc = E.cap_rows(sc, 3_000).rename(columns={group: "group"} if group else {})
        if group:
            sc["group"] = sc["group"].astype(str)
        right.altair_chart(alt.Chart(sc).mark_circle(size=12, opacity=0.5).encode(x="PC1:Q", y="PC2:Q", color=color(group))
                           .properties(height=260, title="Scores, PC1 vs PC2"), width="stretch")
        explain("Keep components up to the elbow or ~80% cumulative variance (eigenvalue > 1 = Kaiser rule when "
                "standardised). Loadings show which variables a component mixes: blood pressures usually load together.")

    st.subheader("Time-series EDA")
    times = [c for c in ["hour", "t_to_onset", "ICULOS"] if c in nums] or nums
    c1, c2, c3, c4 = st.columns(4)
    tcol = c1.selectbox("Time axis", times, format_func=lambda c: {"t_to_onset": "hours to onset (septic only)",
                                                                   "hour": "hour of stay"}.get(c, c))
    opts = [c for c in nums if c != tcol]
    vcol = c2.selectbox("Series", opts, index=opts.index("HR") if "HR" in opts else 0)
    stat = c3.radio("Statistic", ["mean", "measured share"], horizontal=True)
    tmin, tmax = float(WORK[tcol].min()), float(WORK[tcol].max())
    lo, hi = (-48, -1) if tcol == "t_to_onset" else (1, 96)
    span = c4.slider("Range", tmin, tmax, (float(np.clip(lo, tmin, tmax)), float(np.clip(hi, tmin, tmax))))
    by = "is_septic" if "is_septic" in WORK and tcol != "t_to_onset" else None
    s = calc(E.series, tcol, vcol, stat, by, span)
    if s.empty:
        st.info("No rows in this range.")
        return
    window = st.slider("Rolling-mean window (h)", 1, 24, 6)
    if by:
        s["group"] = s.group.map({True: "septic", False: "non-septic"})
    s["rolling"] = s.groupby("group")["mean"].transform(lambda v: v.rolling(window, center=True, min_periods=1).mean()) \
        if by else s["mean"].rolling(window, center=True, min_periods=1).mean()
    base = alt.Chart(s).encode(x=alt.X("t:Q", title=tcol), color=color(by, "") if by else alt.value(HOSP_A))
    chart((base.mark_line(opacity=0.35).encode(y=alt.Y("mean:Q", title=f"{stat} of {vcol}", scale=alt.Scale(zero=False)))
           + base.mark_line(strokeWidth=3).encode(y="rolling:Q")).properties(height=260))
    pooled = s.groupby("t")["mean"].mean() if not by else s[s.group == "septic"].set_index("t")["mean"]
    if len(pooled) < 8:
        st.info("Too few time steps for trend and seasonality tests; widen the range.")
        return
    tr = E.trend_test(pooled.index.to_numpy(float), pooled.to_numpy(float))
    a, b, c = st.columns(3)
    a.metric("Linear trend (per hour)", f"{tr['slope']:+.4g}", f"p = {tr['slope_p']:.2g}", delta_color="off")
    b.metric("Mann-Kendall τ", f"{tr['kendall_tau']:+.3f}", f"p = {tr['mk_p']:.2g}", delta_color="off")
    c.metric("R² of linear trend", f"{tr['r2']:.3f}")
    period = st.select_slider("Seasonal period (h)", [6, 8, 12, 24], value=24)
    ac = E.acf(pooled.to_numpy(float), 48)
    ci = 1.96 / np.sqrt(len(pooled))
    left, right = st.columns(2)
    left.altair_chart((alt.Chart(pd.DataFrame({"lag": range(len(ac)), "acf": ac})).mark_bar(color=HOSP_A).encode(x="lag:O", y="acf:Q")
                       + alt.Chart(pd.DataFrame({"y": [ci, -ci]})).mark_rule(strokeDash=[4, 4], color=SEPTIC).encode(y="y:Q"))
                      .properties(height=240, title=f"Autocorrelation ({'septic' if by else 'pooled'} series)"),
                      width="stretch")
    dec = E.decompose(pooled.to_numpy(float), period).assign(t=pooled.index.to_numpy())
    dm = dec.melt("t", var_name="component")
    right.altair_chart(alt.Chart(dm).mark_line(color=HOSP_A).encode(x="t:Q", y=alt.Y("value:Q", title=None, scale=alt.Scale(zero=False)))
                       .properties(height=60, width=320).facet(row=alt.Row("component:N", sort=["observed", "trend", "seasonal",
                                                                                               "residual"], title=None))
                       .resolve_scale(y="independent").properties(title=f"Additive decomposition, period {period} h"))
    explain("Trend: centred moving average (2×period for even periods); seasonal: mean detrended profile per phase; "
            "residual: the rest. Peaks of the ACF at the period confirm seasonality (e.g. daily lab rounds). Aligning to "
            "onset (`hours to onset`) shows the pre-sepsis trajectory the models learn from.")

    st.subheader("Dashboards and automatic profiling")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(WORK):,}")
    k2.metric("Columns", WORK.shape[1])
    k3.metric("Cells missing", f"{WORK.isna().mean().mean():.1%}")
    if "is_septic" in WORK or "septic" in WORK:
        sep = WORK["is_septic" if "is_septic" in WORK else "septic"]
        k4.metric("Septic share (rows)", f"{sep.mean():.1%}")
    tools = {"ydata_profiling": "ydata-profiling", "sweetviz": "sweetviz", "dtale": "dtale"}
    sample = E.cap_rows(WORK, 20_000)
    for mod, pkg in tools.items():
        with st.container(border=True):
            st.markdown(f"**{pkg}**")
            if importlib.util.find_spec(mod) is None:
                st.caption(f"Not installed. `.venv/bin/pip install {pkg}` and reload this page.")
            elif st.button(f"Generate {pkg} report (20,000-row sample)", key=f"prof-{mod}"):
                with st.spinner(f"Running {pkg}…"):
                    html = profile_report(mod, sample)
                if mod == "dtale":
                    st.markdown(f"D-Tale is running: [open it]({html})")
                else:
                    st.download_button("Download report (.html)", html, f"{pkg}_report.html")
                    components.html(html, height=700, scrolling=True)
    explain("These libraries automate sections 2–6 (types, missingness, distributions, correlations, interactions) "
            "into one report. They are optional; everything above is the same analysis built by hand.")


def profile_report(mod: str, df: pd.DataFrame) -> str:
    """HTML (or, for D-Tale, a URL) from an optional auto-EDA library."""
    if mod == "ydata_profiling":
        from ydata_profiling import ProfileReport

        return ProfileReport(df, minimal=True, progress_bar=False).to_html()
    if mod == "sweetviz":
        import sweetviz

        path = Path(tempfile.mkdtemp()) / "sweetviz_report.html"
        sweetviz.analyze(df).show_html(str(path), open_browser=False, layout="vertical")
        return path.read_text()
    import dtale

    return dtale.show(df, subprocess=True).main_url()


# ===================================================================== page
st.title("EDA lab")
st.caption("The course's EDA syllabus, unit by unit, on the SepsisTRACE data. Every number is computed live.")
available = {k: v for k, v in SOURCES.items() if k == "upload"
             or (k == "cohort" and core.CFG.path("parquet").exists())
             or (k == "raw" and core.CFG.path("raw").joinpath("training_setA").exists())
             or (k == "demo" and core.MODE != "missing")}
SOURCE = st.sidebar.radio("Data source", list(available), format_func=available.get)
seed = int(st.sidebar.number_input("Random seed", 0, 10_000, 42))
IMPORT_INFO: dict = {}
if SOURCE == "cohort":
    POP = cohort()
    n = st.sidebar.select_slider("Stays in the working sample", [500, 1000, 2000, 5000, 10000, "All"], value=2000)
    BASE, KEY = stays_subset(n, seed), ("cohort", core.MODE, n, seed)
elif SOURCE == "raw":
    n = st.sidebar.slider("Raw files to read", 20, 2000, 300, step=20)
    BASE = POP = raw_files(n, seed)
    KEY = ("raw", n, seed)
elif SOURCE == "demo":
    BASE = POP = core.load_demo()
    KEY = ("demo", core.MODE)
else:
    f = st.sidebar.file_uploader("CSV, TSV, PSV, TXT, JSON, Parquet, Feather or Excel",
                                 type=["csv", "tsv", "psv", "txt", "json", "parquet", "feather", "xlsx", "xls"])
    if f is None:
        st.info("Upload a table in the sidebar. A raw PhysioNet file, e.g. `training_setA/p000001.psv`, works as-is.")
        st.stop()
    try:
        BASE, IMPORT_INFO = uploaded(f.name, f.getvalue())
    except Exception as exc:          # unreadable file: show why instead of a traceback
        st.error(f"Could not read {f.name}: {exc}")
        st.stop()
    POP, KEY = BASE, ("upload", f.file_id)

stay_data = {"patient_id", "hour", "is_septic", "Age", "Gender", "hospital"} <= set(BASE.columns)
level = st.sidebar.radio("Unit of analysis", ["Hourly rows", "One row per stay"], horizontal=True) if stay_data else "Rows"
unit = st.query_params.get("unit", "1")          # deep link: ...?unit=3 opens "3 · Missing data"
section = st.sidebar.radio("Syllabus unit", SECTIONS,
                           index=int(unit) - 1 if unit.isdigit() and 1 <= int(unit) <= len(SECTIONS) else 0)
ss.setdefault("eda_messy", False)
ss.setdefault("eda_clean", ())
WORK, CLEAN_LOG = prepare(KEY, level, ss.eda_messy, ss.eda_clean, _base=BASE)
KEY = KEY + (level, ss.eda_messy, ss.eda_clean)
if ss.eda_messy or ss.eda_clean:
    st.sidebar.caption(("Practice problems injected. " if ss.eda_messy else "") + ("Cleaning applied." if ss.eda_clean else ""))
st.caption(f"Working table: **{len(WORK):,} rows × {WORK.shape[1]} columns** ({available[SOURCE]}, {level.lower()}).")

run = dict(zip(SECTIONS, [s_sampling, s_cleaning, s_missing, s_descriptive, s_univariate, s_correlation, s_advanced]))
try:
    run[section]()
except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:     # e.g. an uploaded column of the wrong kind
    st.error(f"This analysis cannot run on the current selection: {exc}")
