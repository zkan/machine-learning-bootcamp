import os
from pathlib import Path

import pandas as pd
import streamlit as st

from paths import churn_bundle_path, spend_bundle_path
from persist import TASK_CHURN, TASK_SPEND, load_bundle
from schema import AISLES, DISTRICTS, FEATURE_COLUMNS, PLANS
from spend import clip_spend
from validate import input_errors


def churn_decision(probability: float, threshold: float) -> str:
    """Return yes or no for a retention voucher.

    Exercise. Return yes when probability is at least threshold.
    Otherwise return no.
    """
    raise NotImplementedError(
        "Finish churn_decision in app.py. Return yes when probability is at least the threshold."
    )


def _path(env_name: str, default: Path) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    return default


def _load(path: Path, task: str, missing_notebook: str):
    if not path.is_file():
        return None, (
            f"No {task} file at {path}. Finish {missing_notebook}, then refresh this page. "
            "Load only a file you trust. A joblib file can run code when it is loaded."
        )
    try:
        return load_bundle(path, expected_task=task), None
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    st.title("Fresh and Good supervised demo")
    st.caption(
        "Synthetic classroom scores for a fictional grocery. This is not a production system. "
        "Churn is a yes or no. Next-month spend is an amount in THB. "
        "Load only a bundle you fit in class. A joblib file can run code when it is loaded."
    )
    churn_path = _path("FRESH_AND_GOOD_CHURN_BUNDLE", churn_bundle_path())
    spend_path = _path("FRESH_AND_GOOD_SPEND_BUNDLE", spend_bundle_path())
    churn_bundle, churn_error = _load(churn_path, TASK_CHURN, "notebook 05")
    spend_bundle, spend_error = _load(spend_path, TASK_SPEND, "notebook 06")
    if churn_error:
        st.error(churn_error)
    elif churn_bundle is not None:
        st.write(
            f"Churn threshold {churn_bundle.threshold:.2f}. "
            f"Missed churner {churn_bundle.cost_fn_thb:.0f} THB. "
            f"Wasted voucher {churn_bundle.cost_fp_thb:.0f} THB."
        )
    if spend_error:
        st.error(spend_error)
    if churn_bundle is None and spend_bundle is None:
        st.stop()

    with st.form("customer"):
        tenure_days = st.number_input("Tenure days", min_value=0, max_value=5000, value=180, step=1)
        orders_90d = st.number_input(
            "Orders in the last 90 days", min_value=0, max_value=80, value=4, step=1
        )
        spend_thb_90d = st.number_input(
            "Spend in the last 90 days, THB", min_value=0.0, value=1500.0, step=10.0
        )
        avg_basket_thb_90d = st.number_input(
            "Average basket, THB", min_value=0.0, value=380.0, step=10.0
        )
        days_since_last_order = st.number_input(
            "Days since last order", min_value=0, max_value=400, value=12, step=1
        )
        promo_order_share_90d = st.number_input(
            "Promo order share", min_value=0.0, max_value=1.0, value=0.2, step=0.05
        )
        app_sessions_90d = st.number_input(
            "App sessions in the last 90 days", min_value=0, max_value=200, value=8, step=1
        )
        support_tickets_90d = st.number_input(
            "Support tickets in the last 90 days", min_value=0, max_value=30, value=0, step=1
        )
        district = st.selectbox("District", DISTRICTS)
        plan = st.selectbox("Plan", PLANS)
        primary_aisle = st.selectbox("Primary aisle", AISLES)
        submitted = st.form_submit_button("Score this customer")

    with st.expander("Optional engineer stretch"):
        st.write(
            "input_errors already rejects spend when the order count is zero. "
            "Add one more rule in validate.py if the team wants it, and surface that message here."
        )

    if not submitted:
        st.info("Enter a customer, then score them.")
        return

    record = {
        "tenure_days": int(tenure_days),
        "orders_90d": int(orders_90d),
        "spend_thb_90d": float(spend_thb_90d),
        "avg_basket_thb_90d": float(avg_basket_thb_90d),
        "days_since_last_order": int(days_since_last_order),
        "promo_order_share_90d": float(promo_order_share_90d),
        "app_sessions_90d": int(app_sessions_90d),
        "support_tickets_90d": int(support_tickets_90d),
        "district": district,
        "plan": plan,
        "primary_aisle": primary_aisle,
    }
    errors = input_errors(record)
    if errors:
        for message in errors:
            st.error(message)
        return

    row = pd.DataFrame([record]).loc[:, list(FEATURE_COLUMNS)]
    if churn_bundle is not None:
        probability = float(churn_bundle.pipeline.predict_proba(row)[0, 1])
        st.metric("Churn probability", f"{probability:.1%}")
        try:
            decision = churn_decision(probability, churn_bundle.threshold)
        except NotImplementedError as exc:
            st.error(str(exc))
        else:
            if decision not in {"yes", "no"}:
                st.error("churn_decision must return yes or no.")
            else:
                st.write(f"Offer a voucher: {decision}")
    if spend_bundle is not None:
        raw = float(spend_bundle.pipeline.predict(row)[0])
        estimate = float(clip_spend([raw])[0])
        st.metric("Estimated next-month spend", f"{estimate:,.0f} THB")


main()
