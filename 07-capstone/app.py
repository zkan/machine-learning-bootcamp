import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
DAY4 = ROOT.parent / "05-mlops-reproducibility"
DAY5 = ROOT.parent / "06-mlops-pipelines"


def _path(folder: Path) -> None:
    if folder.is_dir() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def score_and_monitor():
    _path(DAY5 / "lib")
    _path(DAY5 / "airflow" / "dags")
    from score import monitor_table, score_customer

    return score_customer, monitor_table


def feast_lookup(customer_id: str):
    import re

    identity = customer_id.strip()
    if re.fullmatch(r"FG\d{4}", identity) is None:
        raise ValueError("Use a customer id like FG0010.")
    repo = DAY4 / "feature_repo"
    if not (repo / "data" / "registry.db").is_file():
        raise FileNotFoundError(
            "Day 4 Feast registry is missing. From 05-mlops-reproducibility run make feast."
        )
    from feast import FeatureStore

    result = FeatureStore(repo_path=str(repo)).get_online_features(
        features=["customer_features:orders_90d", "customer_features:plan"],
        entity_rows=[{"customer_id": identity}],
    ).to_dict()
    return int(result["orders_90d"][0]), str(result["plan"][0])


def main() -> None:
    st.title("Fresh and Good capstone")
    st.caption(
        "Demo day form. The score and the monitor come from the Day 5 pipeline. "
        "Feast lookup is a stretch from Day 4. "
        "Synthetic scores for a fictional grocery."
    )
    page = st.radio("Page", ["Score", "Monitor", "Feast"], horizontal=True)
    customer_id = st.text_input("Customer id", value="FG0010")
    if page == "Score":
        if not st.button("Score this customer"):
            st.info("Score a customer from the pipeline alias, or open Monitor.")
            return
        try:
            score_customer, _monitor = score_and_monitor()
            probability = score_customer(customer_id)
        except Exception as exc:
            st.error(str(exc))
            return
        st.metric("Churn probability", f"{probability:.1%}")
        return
    if page == "Monitor":
        try:
            _score, monitor_table = score_and_monitor()
            table = monitor_table()
        except Exception as exc:
            st.error(str(exc))
            return
        st.dataframe(table)
        flagged = table.loc[table["check"], "column"].tolist()
        if flagged:
            st.error("Drift is above the line for " + ", ".join(flagged) + ".")
        else:
            st.success("No column is over the PSI line.")
        return
    if not st.button("Look up in Feast"):
        st.info("Feast is the stretch. Day 4 has to be materialized first.")
        return
    try:
        orders, plan = feast_lookup(customer_id)
    except Exception as exc:
        st.error(str(exc))
        return
    st.write(f"{customer_id} has plan {plan} and {orders} orders in the last 90 days.")


main()
