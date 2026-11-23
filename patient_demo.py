"""Patient demo page: the eight functions (1)-(8) on 30 internal-test patients."""
from __future__ import annotations

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core
from data import LABS, VITALS
from plots import HOSP_A, INK, MUTED, NONSEPTIC, SEPTIC

ss = st.session_state
precomputed = st.cache_data(core.precomputed)
console = st.cache_data(core.threshold_console)


@st.cache_data
def hidden_curve(pid: str, hide: tuple[str, ...], model: str, fset: str, h: int) -> pd.Series:
    return core.predict_with_hidden(core.patient(pid), list(hide), model, fset, h)


def fmt(x: float, spec: str = ".3f") -> str:
    return "—" if np.isnan(x) else format(x, spec)


# ------------------------------------------------------------------------ charts
def risk_chart(curves: dict[str, pd.Series], hour: int, thr: float, onset: float | None) -> alt.Chart:
    df = pd.concat([pd.DataFrame({"hour": s.index, "risk": s.values, "series": n}) for n, s in curves.items()])
    colors = {"risk": SEPTIC, "risk with hidden variables": NONSEPTIC}
    base = alt.Chart(df[df.hour <= hour]).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("hour:Q", title="Hour of stay", scale=alt.Scale(domain=[1, max(hour, 12)])),
        y=alt.Y("risk:Q", title="Calibrated risk"),
        color=alt.Color("series:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["hour", alt.Tooltip("risk:Q", format=".3f"), "series"])
    layers = [base, alt.Chart(pd.DataFrame({"y": [thr]})).mark_rule(strokeDash=[4, 4], color=INK).encode(y="y:Q")]
    if onset is not None and onset <= hour + 1:
        layers.append(alt.Chart(pd.DataFrame({"x": [onset]})).mark_rule(color=SEPTIC, strokeWidth=2).encode(x="x:Q"))
    return alt.layer(*layers).properties(height=260)


def vital_chart(p: pd.DataFrame, hour: int, var: str, greyed: bool) -> alt.Chart:
    """One vital: carried-forward step line plus observed dots. No fixed width, so it fits its column."""
    d = p.loc[p.hour <= hour, ["hour", var]].rename(columns={var: "value"})
    d["ffill"] = d["value"].ffill()
    title = alt.TitleParams(var + (" (not used by M)" if greyed else ""), fontSize=11,
                            color="#999999" if greyed else INK)
    line = alt.Chart(d).mark_line(color="#BBBBBB" if greyed else MUTED, interpolate="step-after").encode(
        x=alt.X("hour:Q", title=None, scale=alt.Scale(domain=[1, max(int(p.hour.max()), 12)])),
        y=alt.Y("ffill:Q", title=None, scale=alt.Scale(zero=False)))
    pts = alt.Chart(d.dropna(subset=["value"])).mark_circle(size=22, color="#BBBBBB" if greyed else SEPTIC).encode(
        x="hour:Q", y="value:Q", tooltip=["hour", alt.Tooltip("value:Q", title=var)])
    return alt.layer(line, pts).properties(height=140, title=title)


def vitals_grid(p: pd.DataFrame, hour: int, greyed: bool, per_row: int = 2) -> None:
    """The four panel vitals in a responsive grid of Streamlit columns."""
    vars_ = core.PANEL_VITALS
    for i in range(0, len(vars_), per_row):
        for col, var in zip(st.columns(per_row), vars_[i:i + per_row]):
            col.altair_chart(vital_chart(p, hour, var, greyed), width="stretch")


def mask_chart(p: pd.DataFrame, hour: int) -> alt.Chart:
    vars_ = VITALS + [v for v in LABS if p[v].notna().any()]
    d = p[p.hour <= hour][["hour"] + vars_].set_index("hour").notna().reset_index().melt(
        id_vars="hour", var_name="variable", value_name="measured")
    return alt.Chart(d[d.measured]).mark_rect(color="#3D4F66").encode(
        x=alt.X("hour:O", title="Hour", axis=alt.Axis(labelOverlap=True)),
        y=alt.Y("variable:N", sort=vars_, title=None)).properties(height=14 * len(vars_))


def gauge_chart(shares: list[float]) -> alt.Chart:
    """One horizontal bar split into V / M / statics shares, labelled in place."""
    blocks = ["values (V)", "measurement (M)", "statics"]
    starts = np.concatenate([[0], np.cumsum(shares)[:-1]])
    g = pd.DataFrame({"row": "share", "block": blocks, "share": shares, "x0": starts, "x1": starts + shares})
    g["mid"] = (g.x0 + g.x1) / 2
    g["label"] = [f"{k} {v:.0%}" if v >= 0.15 else "" for k, v in zip("VMS", shares)]
    color = alt.Color("block:N", sort=blocks, scale=alt.Scale(domain=blocks, range=[HOSP_A, SEPTIC, "#BBBBBB"]),
                      legend=None)
    bars = alt.Chart(g).mark_bar(height=34).encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=[0, 1]), axis=None), x2="x1:Q", y=alt.Y("row:N", axis=None),
        color=color, tooltip=["block", alt.Tooltip("share:Q", format=".0%")])
    text = alt.Chart(g).mark_text(color="white", fontWeight="bold", fontSize=12).encode(
        x="mid:Q", y=alt.Y("row:N", axis=None), text="label:N")
    return alt.layer(bars, text).properties(height=44)


# --------------------------------------------------------------------- sidebar
if core.MODE == "missing":
    st.title("SepsisTRACE")
    st.error("The demo data has not been built yet.")
    st.code(core.BUILD_HELP, language="text")
    st.stop()

patients = core.load_demo().groupby("patient_id").category.first().sort_values()
st.sidebar.caption("Does the *pattern* of measurement predict sepsis? Test-split patients only.")
if core.MODE == "quick-fallback":
    st.sidebar.warning("Showing the **quick** 2000-patient demo: full-data artefacts are not built yet. "
                       "Run `python main.py all` for the full version.")
elif core.MODE == "quick":
    st.sidebar.info("Quick mode (2000-patient subset).")
pid = st.sidebar.selectbox("Patient", patients.index, format_func=lambda x: f"{x} — {patients[x]}")
model = st.sidebar.radio("Model (2)", core.MODELS, index=core.MODELS.index("xgb"), horizontal=True)
fset = st.sidebar.radio("Feature set (3)", core.FEATURE_SETS, index=core.FEATURE_SETS.index("VM"), horizontal=True,
                        format_func=lambda s: {"V": "V values", "M": "M measurement", "VM": "V+M"}[s])
horizon = st.sidebar.select_slider("Horizon (hours ahead)", core.HORIZONS, value=6)
# Floor (not round) the validated threshold to the slider step so the default never sits above it.
val_thr = core.default_threshold(model, fset, horizon)
thr = st.sidebar.slider("Alert threshold (5)", 0.0, 0.5, float(np.floor(val_thr * 1e4) / 1e4), step=0.0001,
                        format="%.4f", key=f"thr-{model}-{fset}-{horizon}")
st.sidebar.caption(f"Default = best-F1 threshold on the validation split ({val_thr:.5f}).")

p = core.patient(pid)
onset = float(p.onset.iloc[0]) if p.is_septic.iloc[0] else None
max_hour = int(p.hour.max())

# Time-travel state is written here, before its widgets exist (Streamlit forbids writing a widget's key
# after the widget is created). A new patient starts at the end of the stay with playback off; each
# playback tick (requested at the very end of the script) advances one hour and stops at the end.
if ss.get("hour_pid") != pid:
    ss.hour, ss.hour_pid, ss.play = max_hour, pid, False
elif ss.pop("tick", False) and ss.get("play"):
    ss.hour = min(ss.hour + 1, max_hour)
    ss.play = ss.hour < max_hour


def on_play() -> None:
    if ss.play and ss.hour >= max_hour:   # Play at the end of the stay replays it from hour 1
        ss.hour = 1


tab_explore, tab_console, tab_challenge = st.tabs(["Patient explorer", "Threshold console", "Clinician challenge"])

# ------------------------------------------------------------- patient explorer
with tab_explore:
    st.subheader(f"Patient {pid} · {patients[pid]}" + (f" · onset at hour {onset:.0f}" if onset else ""))
    c1, c2 = st.columns([5, 1])
    c2.toggle("▶ Play", key="play", on_change=on_play)
    hour = c1.slider("Time-travel (1)", 1, max_hour, key="hour")

    hide = st.multiselect("Hide a measurement (4): treat as never measured, recompute live", VITALS + LABS)
    curves = {"risk": precomputed(pid, model, fset, horizon)}
    if hide:
        curves["risk with hidden variables"] = hidden_curve(pid, tuple(hide), model, fset, horizon)
    st.altair_chart(risk_chart(curves, hour, thr, onset), width="stretch")
    now = curves["risk"].get(hour, np.nan)
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk now", fmt(now))
    m2.metric("Alert", "YES" if now >= thr else "no")
    if hide:
        h2 = curves["risk with hidden variables"].get(hour, np.nan)
        m3.metric("Risk with hidden variables", fmt(h2), None if np.isnan(h2 - now) else f"{h2 - now:+.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Vitals**")
        if fset == "M":
            st.info("Feature set **M**: this model never sees values — only *when* things were measured. "
                    "The vitals below are greyed out because the model cannot use them.")
        vitals_grid(p, hour, greyed=(fset == "M"))
        st.markdown("**Measurement pattern** (what model M sees)")
        st.altair_chart(mask_chart(p, hour), width="stretch")
    with right:
        st.markdown("**Contribution gauge (7)** — share of |SHAP| by feature block, XGBoost V+M model")
        c = core.contribution_for(pid, hour, "xgb", horizon)   # SHAP is exact for XGBoost; see 05_prepare_demo
        if c is None:
            st.caption("No cached SHAP for this hour (predictions start at hour 3).")
        else:
            shares = [*c, max(0.0, 1 - sum(c))]
            st.altair_chart(gauge_chart(shares), width="stretch")
            st.markdown(" &nbsp; ".join(f"<span style='color:{col}'>■</span> {name} **{v:.0%}**" for col, name, v in
                                        zip([HOSP_A, SEPTIC, "#999999"], ["values (V)", "measurement (M)", "statics (S)"],
                                            shares)), unsafe_allow_html=True)
            st.caption(f"XGB V+M, {horizon}h. Top features this hour:")
            st.dataframe(core.top_features_for(pid, hour, "xgb", horizon).round(3), hide_index=True)

        st.markdown("**What-if (6)** — change values at the current hour")
        hist, cur = p[p.hour <= hour], p[p.hour == hour].iloc[0]
        edits = {}
        cols = st.columns(2)
        for i, v in enumerate(core.WHATIF_VARS):
            # Prefill: the value now, else the last known value; leaving the prefill untouched is not an edit.
            last = hist[v].ffill().iloc[-1]
            default = None if np.isnan(last) else float(round(last, 2))
            val = cols[i % 2].number_input(f"{v} (now: {fmt(cur[v], '.2f')})", value=default, key=f"wi-{v}-{pid}-{hour}")
            if val is not None and val != default:
                edits[v] = val
        if hour >= 3:
            before, after = (core.predict_with_edits(p, e, hour, model, fset, horizon, calibrated=False) for e in ({}, edits))
            st.metric("Raw model score at this hour", f"{after:.4f}", f"{after - before:+.4f}" if edits else None)
            if fset == "M" and edits:
                st.caption("Model M ignores values: only adding a measurement that was missing can move it.")
            st.caption("Blank or '—' = not measured at this hour; typing a value there adds a measurement.")
        else:
            st.caption("Predictions start at hour 3.")

# ------------------------------------------------------------ threshold console
with tab_console:
    st.subheader(f"Threshold console (5) — whole internal test split, {model} · {fset} · {horizon}h")
    r = console(model, fset, horizon, thr)
    a, b, c_, e = st.columns(4)
    a.metric("Septic patients alerted before onset", fmt(r["patients_detected_fraction"], ".0%"))
    b.metric("Hour-level sensitivity", f"{r['hour_sensitivity']:.0%}")
    c_.metric("False alarms / patient-day", fmt(r["false_alarms_per_patient_day"], ".2f"))
    lead = r["median_lead_time_hours"]
    e.metric("Median lead time (detected)", "—" if np.isnan(lead) else f"{lead:.0f} h")
    if not r["onsets_available"]:
        st.warning("Onset hours not found (`test_onsets.json` or `data/patients.parquet`): patient-level metrics "
                   "are hidden. Re-run `python main.py demo` to restore them.")
    cm = r["confusion"]
    st.markdown("Hour-level confusion matrix")
    st.table(pd.DataFrame([[cm["TP"], cm["FN"]], [cm["FP"], cm["TN"]]],
                          index=[f"label y_{horizon}=1", f"label y_{horizon}=0"], columns=["alert", "no alert"]))
    st.caption("Lead time is measured over detected septic patients only; read it together with the detected fraction.")

# ---------------------------------------------------------- clinician challenge
def new_challenge() -> None:
    """Draw a different patient from the one on screen."""
    ss.challenge_pid = str(np.random.default_rng().choice(patients.index.difference([ss.get("challenge_pid")])))
    ss.revealed, ss.draws = False, ss.get("draws", 0) + 1


def reveal(guess: str, truth: bool) -> None:
    ss.revealed = True
    ss.rounds += 1
    ss.score += int((guess == "Sepsis coming") == truth)


with tab_challenge:
    st.subheader("Clinician challenge (8)")
    ss.setdefault("score", 0)
    ss.setdefault("rounds", 0)
    if "challenge_pid" not in ss:
        new_challenge()
    cp = core.patient(ss.challenge_pid)
    truth = bool(cp.is_septic.iloc[0])
    st.caption("A hidden test patient. Look at the record and decide whether they develop sepsis.")
    vitals_grid(cp, int(cp.hour.max()), greyed=False, per_row=4)
    st.altair_chart(mask_chart(cp, int(cp.hour.max())), width="stretch")
    guess = st.radio("Your call", ["Sepsis coming", "No sepsis"], horizontal=True, key=f"guess-{ss.draws}",
                     disabled=ss.revealed)
    col1, col2 = st.columns(2)
    # Callbacks run before the next render, so Reveal is disabled as soon as it is pressed.
    col1.button("Reveal", disabled=ss.revealed, on_click=reveal, args=(guess, truth))
    col2.button("Next patient", on_click=new_challenge)
    if ss.revealed:
        st.markdown(f"**Truth:** {'septic, onset at hour %d' % cp.onset.iloc[0] if truth else 'not septic'}")
        rows = []
        for m in core.MODELS:
            s = precomputed(ss.challenge_pid, m, "VM", horizon)
            first = s.index[s.values >= core.default_threshold(m, "VM", horizon)]
            rows.append({"model": m, "max risk": round(float(s.max()), 3),
                         "alerted": bool(len(first)), "first alert hour": int(first[0]) if len(first) else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.metric("Session score", f"{ss.score} / {ss.rounds}")

# Playback runs after every tab has rendered, so the console and challenge stay live while playing.
if ss.get("play"):
    time.sleep(0.4)
    ss.tick = True
    st.rerun()
