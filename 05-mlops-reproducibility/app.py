import pandas as pd
import streamlit as st

from features import serving_frame
from registry import load_champion
from schema import ALIAS, MODEL_FEATURES, MODEL_NAME
from store import online_customer


def score_customer(customer_id: str) -> tuple[float, pd.DataFrame]:
    online = online_customer(customer_id)
    row = serving_frame(online)
    model = load_champion()
    probability = float(model.predict_proba(row.loc[:, list(MODEL_FEATURES)])[0, 1])
    return probability, row


def main() -> None:
    st.title("Fresh and Good registry demo")
    st.caption(
        "Synthetic classroom score for a fictional Bangkok grocery. "
        "The model is loaded by registry name and alias. "
        "Features come from the local Feast online store."
    )
    st.write(f"Registry name `{MODEL_NAME}`, alias `{ALIAS}`.")
    customer_id = st.text_input("Customer id", value="FG0010")
    if not st.button("Score this customer"):
        st.info("Enter a customer id, then score them. Ids look like FG0010.")
        return
    try:
        probability, row = score_customer(customer_id)
    except Exception as exc:
        st.error(str(exc))
        return
    st.metric("Churn probability", f"{probability:.1%}")
    st.write("Serving features for this customer")
    st.dataframe(row)


main()
