import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in (ROOT / "lib", ROOT / "airflow" / "dags"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import streamlit as st

from schema import ALIAS, MODEL_NAME, PSI_LIMIT
from score import monitor_table, score_customer


def main() -> None:
    st.title("Fresh and Good pipeline demo")
    st.caption(
        "Synthetic classroom scores for a fictional Bangkok grocery. "
        "The model is the one the pipeline registered. "
        "The monitor page compares live features with the training reference."
    )
    page = st.radio("Page", ["Score", "Monitor"], horizontal=True)
    if page == "Score":
        st.write(f"Registry name `{MODEL_NAME}`, alias `{ALIAS}`.")
        customer_id = st.text_input("Customer id", value="FG0010")
        if not st.button("Score this customer"):
            st.info("Enter a customer id, then score them.")
            return
        try:
            probability = score_customer(customer_id)
        except Exception as exc:
            st.error(str(exc))
            return
        st.metric("Churn probability", f"{probability:.1%}")
        return
    try:
        table = monitor_table()
    except Exception as exc:
        st.error(str(exc))
        return
    st.write(f"A column with PSI at or above {PSI_LIMIT} needs a human decision.")
    st.dataframe(table)
    flagged = table.loc[table["check"], "column"].tolist()
    if flagged:
        st.error("Drift is above the line for " + ", ".join(flagged) + ".")
    else:
        st.success("No column is over the PSI line.")


main()
