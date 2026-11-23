"""SepsisTRACE app: two pages, the patient demo (default) and the EDA lab.

Start it with `python main.py app` (or `streamlit run eda/app.py` from the project folder).
SEPSISTRACE_QUICK=1 forces the quick-mode data. The light theme comes from .streamlit/config.toml.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):            # eda/ for eda_core; project root for core, data, models, ...
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="SepsisTRACE", layout="wide")
st.sidebar.title("SepsisTRACE")
st.navigation([
    st.Page("patient_demo.py", title="Patient demo", icon=":material/monitor_heart:", default=True),
    st.Page("eda_lab.py", title="EDA lab", icon=":material/query_stats:"),
]).run()
