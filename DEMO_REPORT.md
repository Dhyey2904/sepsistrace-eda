# SepsisTRACE: Project Report

**Does the pattern of clinical measurement predict sepsis, independently of the measured values?**

PC-503 Programming Lab, M.Tech (ICT), DA-IICT · Early sepsis warning on the PhysioNet/CinC 2019 data

---

## Contents

1. Background and motivation
2. Research question and hypotheses
3. Data
4. Exploratory data analysis
5. Methodology and implementation
6. Experiments and results
7. Clinical evaluation
8. Interpretability (SHAP)
9. The demonstration application
10. Discussion
11. Limitations and threats to validity
12. Conclusion

---

## 1. Background and motivation

**Sepsis** is a life-threatening organ dysfunction caused by the body's response to infection. It is common in
intensive care units (ICUs), and every hour of delayed treatment worsens outcomes. A reliable early-warning model
could let clinicians intervene before onset.

ICU patients are monitored continuously. Vital signs are recorded hourly, but laboratory tests are ordered
selectively, so most values are missing in any given hour (median missingness across variables is 93%).
Most prediction work treats this missingness as a nuisance to be filled in. We take the opposite view:

> Clinicians order more tests, more often, when they become worried about a patient. The *timing and frequency of
> measurement* may therefore be an early signal in itself, before any single value turns abnormal.

If that is true, a model could use *what was measured and when* as information. It could also be a trap. Measurement
habits differ between hospitals and depend on how long a patient has stayed, so a model that leans on them may look
good in one hospital and fail in another. This project measures which of the two is the case.

## 2. Research question and hypotheses

**Main question:** does the pattern of clinical measurement predict sepsis, independently of the measured values?

| # | Question | How it is tested |
|---|---|---|
| Q1 | Do septic patients get measured differently before onset? | EDA with stay-length matching (Section 4) |
| Q2 | Does a measurement-only model (M) match a values-only model (V)? | Full grid, 4 models × 3 feature sets × 4 horizons |
| Q3 | Is that signal real, or an artefact such as time-in-ICU? | Experiment E1: shuffled, no-LOS, LOS-only and matched controls |
| Q4 | Does it transfer to a second hospital? | Experiment E2: train on A, test on B |
| Q5 | Does adding measurement information to imputed values help? | Experiment E3: imputation ladder |
| Q6 | Are the risk scores trustworthy and clinically usable? | Calibration, threshold analysis, official utility score |
| Q7 | What does the model actually rely on? | SHAP attribution by feature block |

## 3. Data

**Source:** PhysioNet/CinC Challenge 2019, the public ICU sepsis dataset from two hospital systems.

| Item | Value |
|---|---|
| ICU stays | 40,290 (hospital A: 20,336, hospital B: 19,954) |
| Hourly rows | 1,550,422 |
| Variables per hour | 40: 8 vitals, 26 labs, 6 statics (age, gender, units, admission time, ICU length of stay) |
| Vitals | HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 |
| Septic patients (raw) | 2,929 (7.3%) |
| Already septic at hour 1 | 426, excluded (rule L6) |
| Cohort after exclusion | 39,864 patients, 2,503 septic (6.3%) |
| Modelling rows | 1,537,508 (9,268 post-onset rows dropped) |
| Median stay | 39 h; 98.5% last at least 12 h, 76.8% at least 24 h |
| Median onset hour | 44 |

**Dataset size note:** the downloaded set B holds 19,954 files, so the total is 40,290 rather than the 40,336 in the
original challenge description.

**Labels.** The prediction target at horizon *h* is whether sepsis onset occurs within the next *h* hours
(h = 1, 3, 6, 12). The challenge labels turn positive 6 hours before onset, so a label shift of 6 h is applied. We
verified this empirically: in every septic patient the run of 1s continues unbroken to the end of the record and is never
longer than 10 hours. The positive-hour rate is low, at 0.16% (1 h), 0.49% (3 h), 0.97% (6 h) and 1.89% (12 h).

**Splits.** Hospital A is split by patient into train 70%, validation 10%, calibration 10% and test 10%. Hospital B is
held out entirely as the external test. "Internal" results mean A's test split; "external" means all of B.

## 4. Exploratory data analysis

The EDA fixed several design decisions and exposed the main pitfall of the dataset.

### 4.1 Choice of a 12-hour window

Rule: keep a 12-hour lookback if at least 80% of septic patients have 12 or more hours of record before onset;
otherwise fall back to 8 hours. Result: **87.7%** of septic patients have at least 12 h of history (97.0% have at least
8 h), so the 12-hour window is kept. Shorter histories use what exists, and a `hours_available` feature tells the model
how much history it saw.

### 4.2 Missingness

Median missingness is 93%. Vitals such as HR are mostly present (about 10% missing), while many labs are almost
never measured. Missingness also differs between hospitals: hospital A never records EtCO2, and DBP and the unit fields
differ in missingness. This is the first sign that measurement practice is hospital-specific.

### 4.3 The stay-length artefact

Non-septic records are effectively capped near 59 hours (99th percentile; maximum 336), while **39.2%** of septic onsets
occur later than that. Stay length (`ICULOS`) alone therefore separates the classes, which is a property of how the
dataset was built and not a clinical signal. This is why experiment E1 includes a no-stay-length condition and a
stay-length-matched cohort.

### 4.4 Measurement intensity survives matching

To remove the artefact, each septic patient was matched to a control with a similar stay length (within ±6 h):
1,802 of 2,503 septic patients matched (unmatched median onset 112 h; matched onset and control stay medians are both
29 h; KS statistic 0.033, so the stay-length distributions are close). Even after matching, septic patients have more
variables measured per hour before onset: **+1.01** at 24 to 19 hours before onset, rising to **+1.45** over the last
6 hours (7.39 vs 5.94 variables per hour). This supports Q1: a genuine intensity signal exists.

![Measurement rate before onset, stay-length matched](figures/fig1b_measurement_rate_matched.png)

## 5. Methodology and implementation

### 5.1 Leakage control

Medical prediction is easy to get wrong through leakage, so six rules are implemented as assertions that run after
every transformation (`data.py`):

| Rule | Meaning |
|---|---|
| L1 | Train, validation, calibration and test splits share no patients |
| L2 | Preprocessing is fitted on the training split only |
| L3 | No rows after sepsis onset are used |
| L4 | Windows are causal: the row at hour *t* only sees rows up to *t* |
| L5 | The calibration split is disjoint from training |
| L6 | Patients already septic at admission are excluded |

### 5.2 Features

A single function, `build_features`, is shared by training and the app. It first builds causal windows
(`causal_windows`: a sliding window over the last 12 hours, padded with NaN before the stay starts). Three feature
sets are derived. Statics are included in every set.

| Set | Contents | Purpose |
|---|---|---|
| **V** (values) | Per variable over the window: mean, std, min, max, last and slope | What a conventional model sees |
| **M** (measurement pattern) | Counts of measurements, hours since last measured, how many variables were ever measured; never the values | Isolates the measurement signal |
| **VM** | Both | Best-of-both comparison |

### 5.3 Models

| Model | Settings |
|---|---|
| Logistic regression | max_iter 2000, C = 1.0 (deterministic) |
| Random forest | 300 trees, min_samples_leaf 20 |
| XGBoost | 400 trees, depth 5, learning rate 0.05, subsample 0.8, colsample 0.8, metric AUCPR |
| MLP | hidden layers (64, 32), early stopping, max_iter 200 |

Class weighting handles the imbalance. Three seeds (42, 43, 44) quantify variation.

### 5.4 Calibration

Class weighting inflates raw probabilities, so scores are calibrated with **isotonic regression** fitted on the separate
calibration split (rule L5). If a case had fewer than 50 positives, a sigmoid fallback would be used; none needed it.
Calibration quality is measured with expected calibration error (ECE, 15 bins) and the Brier score.

### 5.5 Metrics

- **AUROC** and **AUPRC** for discrimination.
- **Lift** = AUPRC ÷ positive rate (1.0 = chance). This is the headline metric because AUPRC alone is not comparable
  across horizons with different base rates.
- **ECE, Brier** for calibration.
- **Clinical metrics:** share of septic patients alerted before onset, false alarms per patient-day, false alarms per
  true alarm, share of non-septic patients ever alerted, lead time.
- **PhysioNet/CinC 2019 utility**, computed with the organisers' unmodified scorer.

### 5.6 Experimental grid

The full grid is 4 models × 3 feature sets × 4 horizons × 3 seeds × 2 test sets (internal and external) = **288
evaluated rows**, covering 144 trained models. The full run took 10.8 hours on Kaggle (4 CPUs); the grid alone took
6.8 hours, about 4 of them for the random forests.

## 6. Experiments and results

All numbers come from the generated result tables of the full run.

### 6.1 Full grid

Every one of the 288 cells is above chance (mean lift 1.9 to 5.6). Inside hospital A, M is on par with V. At the 6 h
horizon: random forest 4.07 (M) vs 3.88 (V), XGBoost 3.85 vs 3.95, logistic regression 3.56 vs 3.54. V+M gives the best
AUROC for the tree models (0.80 to 0.82). On hospital B, V leads in most cells. The random forest transfers best:
V+M at 6 h reaches AUROC 0.800 on B against 0.802 on A. The MLP is the weakest (6 h internal AUROC 0.68 to 0.73). Seed
variation is small: the standard deviation is at most 0.2 lift in 80% of cells.

### 6.2 E1: measurement-only, with controls (6 h horizon)

Goal: check whether M's performance is a real measurement signal or an artefact. Conditions, with mean internal lift
over models:

| Condition | Idea | Internal lift |
|---|---|---|
| M_full | All measurement features | 3.59 |
| M_no_los | M without ICU hour and hours available | 3.54 |
| LOS_only | Only ICU hour and hours available | 3.85 |
| M_shuffled | Measurement blocks swapped between patients of similar stay length | 3.4 to 4.6 |
| M_shuffled_no_los | Shuffled and without the length features | 3.1 to 4.0 |
| **M_matched** | Stay-length-matched cohort | **2.50 / 2.20 / 1.70 / 1.47** at 1 / 3 / 6 / 12 h |

**The planned acceptance check failed:** a shuffled control should collapse to chance (lift near 1.0) if the signal were
real measurement behaviour, but it did not. This was investigated rather than hidden:

1. LOS_only already matches M_full, so time in ICU alone carries the signal, because of the 60 h truncation of
   non-septic records.
2. Shuffling keeps each row's hour, and several M features depend on the hour: counts, hours since a never-measured lab
   (equal to log(1 + t)), and the number of variables ever measured. M_shuffled therefore retains the time signal.
3. This is a property of the data, not code leakage: the L1 to L6 checks pass, splits are by patient, and tests confirm
   features are causal and M contains no values.
4. Only the matched cohort removes the artefact. There a **small, short-horizon signal remains** (lift 2.5 at 1 h
   falling to 1.5 at 12 h), and on hospital B it is about 1.0, i.e. chance.

**Finding:** most of M's headline performance is time-in-ICU. A modest genuine signal exists inside one hospital and
fades as the horizon grows.

![E1 controls](figures/fig8_e1_controls.png)

### 6.3 E2: cross-hospital transfer

Models trained on hospital A are tested on hospital B. V transfers better than M for every model and horizon on AUROC;
M loses 0.06 to 0.20 more AUROC than V. Averaged over models, horizons and seeds, **M loses +0.108 AUROC and +1.16
AUPRC-lift more than V**. The only exception on lift is logistic regression at 1 h (−0.16). ECE also degrades more for M
in 13 of 16 model-horizon pairs. The reason is consistent with the EDA: measurement practice differs between hospitals,
so a model of *what gets measured* learns hospital-specific habits.

![Transfer from hospital A to B](figures/fig9_transfer.png)

### 6.4 E3: imputation ladder (XGBoost, median imputer)

Four rungs add information step by step: R1 current values with median fill, R2 forward-filled values, R3 R2 plus
measurement counts, R4 full V+M.

- **Internally the ladder is flat** (lift 3.8 to 4.3 at 3 to 12 h). R3 is highest at 1 h (5.17) but with a seed standard
  deviation of 1.15.
- **On hospital B the ladder decreases monotonically at 3 of 4 horizons** (1 h: 4.31 → 3.46 → 3.24 → 2.63;
  6 h: 4.01 → 3.83 → 3.38 → 3.19). At 3 h, R1 and R2 tie.

Adding measurement information does not help inside hospital A and costs transfer, consistent with E2. All horizons are
reported; none was selected.

### 6.5 Calibration

Isotonic regression improved ECE in **all 96 cases**: internal 0.019 to 0.38 down to at most 0.0049, and hospital B
0.014 to 0.33 down to at most 0.0087. Brier scores also fall sharply, mostly because class weighting had inflated the raw
probabilities. Calibrated outputs can be read as actual risks.

![Reliability](figures/fig11_reliability.png)

## 7. Clinical evaluation

### 7.1 Operating point

XGBoost, V+M, 6 h horizon, calibrated, with threshold 0.056 chosen by best F1 on the validation split (never on the
test split):

| | Internal test | Hospital B |
|---|---|---|
| Septic patients alerted before onset | 51.6% | 28.9% |
| False alarms per patient-day | 0.33 | 0.15 |
| False alarms per true alarm | 0.37 | 0.76 |
| Non-septic patients ever alerted | 7.8% | 1.7% |
| Median lead time, detected patients | 36 h | 43 h |

Lead time is long because any alert before onset counts, including very early ones, so it should be read together with
the detected fraction. The threshold trades sensitivity against alarm burden:

![Threshold trade-off](figures/fig12_threshold_tradeoff.png)

### 7.2 PhysioNet 2019 utility score

Computed with the organisers' unmodified scorer (V+M, 6 h, threshold tuned on validation utility):

| Model | Internal test | Hospital B |
|---|---|---|
| XGBoost | **0.371** | 0.151 |
| Random forest | 0.358 | **0.301** |
| Logistic regression | 0.337 | 0.089 |
| MLP | 0.226 | 0.066 |

For context, the challenge's best official score was about 0.36, but on a different, hidden test set, so these numbers
are not directly comparable. The random forest is the most robust across hospitals.

## 8. Interpretability (SHAP)

SHAP values were computed for XGBoost, V+M, 6 h on the internal test split (200 background and up to 5,000 explained
rows). Additivity holds to 2 × 10⁻⁶.

- On average **55%** of the explanation comes from values, **28%** from the measurement pattern and about **17%** from
  statics.
- The most important single feature is **ICULOS (time in ICU)**, the same artefact E1 exposed.
- Second is `FiO2_dt_log`, hours since FiO2 was last recorded, a proxy for respiratory support.
- The measurement share is similar for septic and non-septic patients (median 0.26 vs 0.28).

The SHAP gauge uses XGBoost only: SHAP does not reproduce scikit-learn's native missing-value routing in the random
forest (an RF explanation summed to 52.05 while the model output was 0.88), so it is not used there.

![SHAP summary](figures/fig13_shap_summary.png)

## 9. The demonstration application

A Streamlit app presents the findings interactively, so the models can be probed rather than only read about. It has two
pages.

**Patient demo** (30 internal-test patients chosen to cover early-detected, late-detected, quiet, false-alarm and
heavily-missing cases) with eight functions:

| # | Function | What it demonstrates |
|---|---|---|
| 1 | Time-travel and Play | How the risk evolves hour by hour through a stay |
| 2 | Model switch | LR, RF, XGBoost and MLP give different risk curves on the same patient |
| 3 | Feature-set switch | V, M or V+M, so the effect of measurement pattern is visible |
| 4 | Hide a measurement | Treat chosen vitals or labs as never measured and recompute live, testing reliance on pattern |
| 5 | Threshold console | Confusion matrix, alert rate, lead time and false alarms on the whole internal test split |
| 6 | What-if | Edit HR, BP, temperature or lactate and see the new score |
| 7 | Contribution gauge | Share of |SHAP| from values vs measurement pattern |
| 8 | Clinician challenge | Guess sepsis from a hidden record, then compare with the four models |

**EDA lab** with seven syllabus units applied to the cohort, raw files, demo patients or an uploaded file: sampling and
experiment design, import and cleaning, missing data (including MCAR/MAR/MNAR and KNN or iterative imputation),
descriptive summaries, univariate and bivariate analysis, correlation and regression, and advanced EDA (PCA, time
series, automatic profiling).

Predictions are computed live from the trained models, calibrated before display, and the default alert threshold is the
validation-chosen one. All eight functions respond in under 0.2 s each.

## 10. Discussion

**Why the question matters.** If measurement behaviour were a robust signal it would be a cheap, always-available
feature, since it requires no extra tests. The results show it is not that simple.

**What the evidence says.**
1. Inside one hospital, measurement pattern predicts sepsis about as well as values (Section 6.1).
2. Most of that performance is time-in-ICU, an artefact of how non-septic records were truncated (Sections 4.3, 6.2).
3. After controlling stay length, a smaller genuine signal remains at short horizons (Section 6.2), consistent with the
   matched EDA result that septic patients are measured more intensely before onset (Section 4.4).
4. That signal does not transfer. Measurement habits are hospital-specific, so values are the more portable basis
   (Sections 6.3, 6.4).

**Practical reading.** A deployed early-warning model should be built primarily on values and validated externally.
Measurement features can add local, short-horizon information but should not be trusted across sites without
re-validation, and any evaluation must control for stay length.

**On the negative result.** The shuffled control did not collapse to chance. We report it as it is and explain its cause.
Reporting only the favourable conditions would have overstated the measurement-pattern effect.

## 11. Limitations and threats to validity

- **Dataset construction.** Truncated non-septic stays make stay length predictive on its own. Matching reduces but
  cannot fully remove this, and 701 septic patients (median onset 112 h) could not be matched.
- **Only two hospitals.** Transfer conclusions rest on a single A-to-B direction.
- **Retrospective data.** No prospective or real-time validation; alert timing assumes hourly availability of data.
- **Threshold selection.** Thresholds are chosen on validation data. Performance at other thresholds follows the
  trade-off curve, and external alert rates differ from internal ones.
- **Lead time.** Median lead time counts any pre-onset alert, so it overstates how early alerts are for most patients.
- **SHAP scope.** Contributions are computed for XGBoost only.
- **Demo scale.** The demo uses 30 patients for interactivity; the threshold console uses the full internal test split.
- **Compute.** The MLP is a small network; a tuned deep model might behave differently.

## 12. Conclusion

SepsisTRACE shows that **how often and when tests are taken carries real, but limited and local, information about
upcoming sepsis**. Inside a single hospital, measurement pattern alone predicts sepsis about as well as the values, but
most of that is time-in-ICU. With stay length controlled, a small short-horizon signal remains (lift 2.5 falling to
1.5), and it does not carry over to a second hospital, where value-based models transfer better. Calibrated XGBoost
reaches an official utility of 0.371 internally, and the random forest is the most robust across hospitals (0.301 on
hospital B). The study also documents a negative control result and its cause, giving an honest account of what
measurement-pattern features can and cannot do.
