# SepsisTRACE: EDA app (Streamlit)

Streamlit app for the EDA course, with two pages: the **patient demo** and the **EDA lab**.
Built on the PhysioNet/CinC 2019 early-sepsis data.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| `app.py` | Entry point and page navigation |
| `patient_demo.py` | Patient demo page |
| `eda_lab.py` / `eda_core.py` | EDA lab page and its calculations |
| `core.py`, `data.py`, `features.py`, `models.py`, `plots.py` | Shared pipeline modules the pages import |
| `config.py`, `config.yaml` | Settings and paths |
| `data/` | Small demo files (demo patients, predictions, SHAP cache) |
| `.streamlit/config.toml` | Light theme |

## Not included (too large)
Raw data (`training_setA/`, `training_setB/`), `data/patients.parquet` and trained models (`artifacts/`)
are git-ignored. Put the raw PhysioNet folders next to `app.py` and build them with the full
SepsisTRACE pipeline to use the EDA lab on the full data and the model-backed demo.
