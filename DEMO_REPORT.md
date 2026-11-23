# SepsisTRACE: Demo Report

PC-503 Programming Lab, M.Tech (ICT), DA-IICT. Streamlit app for the EDA course.

**Research question:** does the *pattern of clinical measurement* (when tests are taken) predict sepsis,
independently of the measured values?

---

## 1. Problem

Sepsis is a life-threatening reaction to infection and is common in ICU patients. Earlier detection improves
survival. ICU staff record vitals and labs hour by hour, and clinicians order more tests when they become worried,
so the *timing* of tests may itself be a warning signal. This project tests that idea and builds an app to explore it.

## 2. Data

| Item | Value |
|---|---|
| Source | PhysioNet/CinC Challenge 2019 (public) |
| ICU stays | 40,290 (hospital A: 20,336, hospital B: 19,954) |
| Granularity | One row per hour, 40 variables (vitals, labs, statics) |
| Variables used | 8 vitals, 8 labs, 6 statics (HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2; WBC, Lactate, Creatinine, Platelets, BUN, Bilirubin, Glucose, Hct; Age, Gender, Unit1/2, HospAdmTime, ICULOS) |
| Excluded | 426 patients already septic at hour 1 (rule L6), leaving 39,864 |
| Septic patients | 2,503 (6.3%) |
| Positive-hour rate | 0.16% (1 h), 0.49% (3 h), 0.97% (6 h), 1.89% (12 h) |
| Missingness | High: labs are measured rarely. This is the object of study. |

**Splits (by patient, never by row):** hospital A is split train 70 / val 10 / calibration 10 / test 10.
Hospital B is kept whole as the external test.

## 3. Method

1. **Window:** look back up to 12 hours at every hour. A 12 h window was kept because 87.7% of septic patients
   have at least 12 h of history before onset (rule: at least 80%).
2. **Feature sets**
   - **V:** values (mean, min, max, trend of the measurements)
   - **M:** measurement pattern only (counts, hours since last measured, never values)
   - **V+M:** both
3. **Models:** Logistic Regression, Random Forest, XGBoost, MLP.
4. **Horizons:** risk of sepsis onset within 1, 3, 6 or 12 hours.
5. **Calibration:** isotonic regression on a separate calibration split.
6. **Grid:** 4 models × 3 feature sets × 4 horizons × 3 seeds × 2 test sets = 288 evaluated rows.
7. **Leakage rules L1–L6** (assertions in `data.py`): patient-level splits; train-only preprocessing;
   no post-onset rows; causal windows; separate calibration split; septic-at-admission excluded.

Main metric: **lift** = AUPRC ÷ positive rate (1.0 = chance), because AUPRC alone is not comparable across horizons.

## 4. The app

Run: `pip install -r requirements.txt` then `streamlit run app.py` (opens at http://localhost:8501).
Light theme from `.streamlit/config.toml`. Two pages.

### 4.1 Patient demo (`patient_demo.py`)

Sidebar controls: patient (30 demo patients from the internal test split), **model (2)**, **feature set (3)**,
horizon (1/3/6/12 h), **alert threshold (5)** (defaults to the best-F1 threshold on the validation split).

| # | Function | What it does |
|---|---|---|
| 1 | Time-travel / Play | Replays a stay hour by hour; the risk curve and vitals grow with time |
| 2 | Model switch | LR / RF / XGB / MLP, predictions computed live |
| 3 | Feature-set switch | V, M or V+M |
| 4 | Hide a measurement | Treats chosen vitals/labs as never measured and recomputes the risk |
| 5 | Threshold console | Confusion matrix, alert rate, lead time and false alarms for the whole internal test split |
| 6 | What-if | Edit HR, BP, temperature, lactate at the current hour and see the new score |
| 7 | Contribution gauge | Share of |SHAP| from values vs measurement pattern (XGBoost V+M) |
| 8 | Clinician challenge | Guess sepsis from a hidden record, then compare with the four models |

Tabs: **Patient explorer**, **Threshold console**, **Clinician challenge**.

### 4.2 EDA lab (`eda_lab.py`)

Seven syllabus units, applied to the cohort, raw `.psv` samples, the demo patients or an uploaded file:

1. Sampling and experiment design (census vs sample, bias, the model grid as a designed experiment)
2. Import and cleaning (format checks, invalid entries, data types, transformation)
3. Missing data (how much, where and when; MCAR/MAR/MNAR; imputation including KNN and iterative)
4. Descriptive summary (central tendency, spread, covariance)
5. Univariate and bivariate (histograms, KDE, violins, scatter, crosstab)
6. Correlation and regression (heatmaps, multicollinearity, goodness of fit)
7. Advanced EDA and dashboards (PCA, time-series EDA, automatic profiling with ydata-profiling, Sweetviz, D-Tale)

## 5. Results

All numbers come from `results/*.csv` and `figures/stats.json` of the full run (10.8 h on Kaggle, 4 CPUs).

**Short answer:** inside hospital A, M predicts as well as V, but most of that is time-in-ICU, a dataset artefact.
With stay length controlled, a small short-horizon signal remains (lift 2.5 at 1 h, falling to 1.5 at 12 h), it does
not transfer to hospital B, and values transfer better than measurement patterns.

### 5.1 EDA findings

- Non-septic records stop near 59 h (99th percentile) while 39.2% of septic onsets come later, so ICU length alone
  separates the classes.
- After matching stay length (1,802 of 2,503 septic patients matched within ±6 h), septic patients still have
  more variables measured per hour: **+1.01** at −24..−19 h and **+1.45** over the last 6 h before onset.
- Hospital A never records EtCO2; DBP and unit fields differ in missingness between hospitals.

![Measurement rate, stay-length matched](figures/fig1b_measurement_rate_matched.png)

### 5.2 Full grid

All 288 cells beat chance (mean lift 1.9–5.6). At 6 h internal: RF 4.07 (M) vs 3.88 (V); XGB 3.85 vs 3.95;
LR 3.56 vs 3.54. V+M gives the best AUROC for tree models (0.80–0.82). The MLP is weakest (AUROC 0.68–0.73).

### 5.3 E1: measurement-only with controls (6 h)

| Condition | Internal lift |
|---|---|
| M_full | 3.59 |
| M_no_los | 3.54 |
| LOS_only | 3.85 |
| M_shuffled | 3.4–4.6 (did **not** collapse to chance) |
| M_matched (stay-length matched) | **2.50 / 2.20 / 1.70 / 1.47** at 1/3/6/12 h; ≈ 1.0 on hospital B |

The failed shuffled control was investigated: shuffling keeps each row's hour, and several M features depend on the
hour. The code passes the L1–L6 checks, so this is a data property, not leakage.

### 5.4 E2: cross-hospital transfer

Moving from A to B, **M loses +0.108 AUROC and +1.16 lift more than V** (averaged). V transfers better for every
model and horizon on AUROC. ECE also degrades more for M in 13 of 16 model–horizon pairs.

![Transfer A to B](figures/fig9_transfer.png)

### 5.5 E3: imputation ladder (XGBoost)

Internally flat (lift 3.8–4.3 at 3–12 h). On hospital B it decreases monotonically at 3 of 4 horizons
(1 h: 4.31 → 3.46 → 3.24 → 2.63). Adding measurement information does not help inside A and costs transfer.

### 5.6 Calibration

Isotonic calibration improved ECE in all 96 cases: internal 0.019–0.38 → ≤ 0.0049; hospital B 0.014–0.33 → ≤ 0.0087.
Outputs can be read as risks.

### 5.7 Clinical operating point (XGBoost, V+M, 6 h, threshold 0.056)

| | Internal test | Hospital B |
|---|---|---|
| Septic patients alerted before onset | 51.6% | 28.9% |
| False alarms per patient-day | 0.33 | 0.15 |
| False alarms per true alarm | 0.37 | 0.76 |
| Non-septic patients ever alerted | 7.8% | 1.7% |
| Median lead time (detected) | 36 h | 43 h |

![Threshold trade-off](figures/fig12_threshold_tradeoff.png)

### 5.8 PhysioNet 2019 utility (official unmodified scorer, V+M, 6 h)

| Model | Internal | Hospital B |
|---|---|---|
| XGBoost | **0.371** | 0.151 |
| Random forest | 0.358 | **0.301** |
| Logistic regression | 0.337 | 0.089 |
| MLP | 0.226 | 0.066 |

Not directly comparable with the challenge's best official score (about 0.36), which used a hidden test set.

### 5.9 SHAP (XGBoost, V+M, 6 h)

About 55% of the explanation comes from values, 28% from the measurement pattern, 17% from statics. The top feature
is **ICULOS** (time in ICU), the same artefact exposed by E1; second is `FiO2_dt_log`.

![SHAP summary](figures/fig13_shap_summary.png)

## 6. Conclusion

- *When* tests are taken carries real information inside one hospital, but much of it is time-in-ICU.
- A small matched signal remains at short horizons and does not transfer to another hospital.
- Values are the more reliable basis across hospitals; measurement patterns encode local habits.
- Negative results (the failed shuffled control) are reported, not hidden.

## 7. Limitations

- Dataset truncation of non-septic stays makes stay length predictive by itself.
- Hospital B transfer is weak for M and for most models.
- The demo uses 30 internal-test patients; the threshold console uses the full internal test split.
- Raw data, `patients.parquet` and trained models are not in git (see the README). The demo pages use the small
  files in `data/`; live model predictions need the `artifacts/` folder from the full pipeline.

## 8. Repository layout

| File | Purpose |
|---|---|
| `app.py` | Entry point, navigation between the two pages |
| `patient_demo.py` | Patient demo page |
| `eda_lab.py`, `eda_core.py` | EDA lab page and its calculations |
| `core.py`, `data.py`, `features.py`, `models.py`, `plots.py` | Shared pipeline modules |
| `config.py`, `config.yaml` | Settings and paths |
| `data/` | Small demo data (patients, predictions, thresholds, SHAP cache) |
| `figures/` | Figures used in this report |

## 9. Tools

Python 3.13, Streamlit, Altair, pandas, NumPy, SciPy, scikit-learn, XGBoost, SHAP, joblib, PyYAML.
