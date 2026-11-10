"""Day 2 form: segment table and product neighbors. Does not touch the Day 1 form."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SEGMENTS = Path(os.environ.get("DAY2_SEGMENTS", ROOT / "models" / "segments.csv"))
NEIGHBORS = Path(os.environ.get("DAY2_NEIGHBORS", ROOT / "models" / "neighbors.csv"))

st.title("Fresh and Good, day 2")
st.caption("Synthetic classroom demo. Segments and product neighbors come from the notebooks.")

segments_tab, recs_tab = st.tabs(["Segments", "Recommendations"])

with segments_tab:
    if not SEGMENTS.is_file():
        st.error(f"Missing {SEGMENTS.name}. Run 01_clusters.ipynb and save the segment table.")
    else:
        st.dataframe(pd.read_csv(SEGMENTS), hide_index=True, width="stretch")

with recs_tab:
    if not NEIGHBORS.is_file():
        st.error(f"Missing {NEIGHBORS.name}. Run 03_recommend.ipynb and save the neighbor table.")
    else:
        table = pd.read_csv(NEIGHBORS)
        product = st.selectbox("Product in the basket", sorted(table["product"].unique()))
        st.dataframe(table[table["product"] == product], hide_index=True, width="stretch")
