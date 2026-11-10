"""Day 3 form: classic random forest or MLP. Does not import Day 1 or Day 2."""

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# Same columns as the Day 1 customer file. Copied here so this form does not import Day 1.
FEATURES = (
    "tenure_days",
    "orders_90d",
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "days_since_last_order",
    "promo_order_share_90d",
    "app_sessions_90d",
    "support_tickets_90d",
    "district",
    "plan",
    "primary_aisle",
)
DISTRICTS = ("Bang Kapi", "Chatuchak", "Lat Phrao", "Huai Khwang", "Bang Na")
PLANS = ("none", "weekly", "monthly")
AISLES = ("fresh", "pantry", "household", "ready_meal")
LABELS = {"classic": "Classic random forest", "mlp": "MLP"}

CLASSIC = Path(os.environ.get("DAY3_CLASSIC", ROOT / "models" / "classic.joblib"))
MLP = Path(os.environ.get("DAY3_MLP", ROOT / "models" / "mlp.joblib"))
CHOSEN = Path(os.environ.get("DAY3_CHOSEN", ROOT / "models" / "chosen.json"))
PATHS = {"classic": CLASSIC, "mlp": MLP}


def load_bundle(path: Path):
    if not path.is_file():
        return None, (
            f"No model file at {path.name}. Run 03_swap.ipynb, then refresh. "
            "Load only a file you fit in class. A joblib file can run code when it is loaded."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "display_name" not in bundle:
        return None, f"{path.name} is not a Day 3 model file."
    return bundle, None


def load_chosen(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


st.title("Fresh and Good, day 3")
st.caption(
    "Synthetic classroom scores for a fictional grocery. "
    "Classic is a random forest. MLP is a small neural net. This is not a production system."
)

choice = st.radio("Model", ["classic", "mlp"], format_func=lambda key: LABELS[key], horizontal=True)
bundle, error = load_bundle(PATHS[choice])
chosen = load_chosen(CHOSEN)
if error:
    st.error(error)
elif bundle is not None:
    if chosen and chosen.get("model") == choice:
        version = f"MLflow version {chosen.get('version')}"
    else:
        version = "not registered"
    st.write(f"In use: {bundle['display_name']}, {version}")
if chosen:
    st.caption(
        f"Registered model: {chosen.get('model')} version {chosen.get('version')} "
        f"({chosen.get('registered_name', 'fresh-and-good-churn')})."
    )
elif not error:
    st.caption("Nothing is registered yet. The register cell in 03_swap.ipynb writes chosen.json.")

if bundle is None:
    st.stop()

with st.form("customer"):
    tenure_days = st.number_input("Tenure days", min_value=0, max_value=5000, value=180, step=1)
    orders_90d = st.number_input("Orders in the last 90 days", min_value=0, max_value=80, value=4, step=1)
    spend_thb_90d = st.number_input("Spend in the last 90 days, THB", min_value=0.0, value=1500.0, step=10.0)
    avg_basket_thb_90d = st.number_input("Average basket, THB", min_value=0.0, value=380.0, step=10.0)
    days_since_last_order = st.number_input("Days since last order", min_value=0, max_value=400, value=12, step=1)
    promo_order_share_90d = st.number_input("Promo order share", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    app_sessions_90d = st.number_input("App sessions in the last 90 days", min_value=0, max_value=200, value=8, step=1)
    support_tickets_90d = st.number_input("Support tickets in the last 90 days", min_value=0, max_value=30, value=0, step=1)
    district = st.selectbox("District", DISTRICTS)
    plan = st.selectbox("Plan", PLANS)
    primary_aisle = st.selectbox("Primary aisle", AISLES)
    submitted = st.form_submit_button("Score this customer")

if not submitted:
    st.info("Enter a customer, then score them.")
    st.stop()

row = pd.DataFrame(
    [
        {
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
    ]
).loc[:, list(FEATURES)]
probability = float(bundle["pipeline"].predict_proba(row)[0, 1])
st.metric("Churn probability", f"{probability:.1%}")
