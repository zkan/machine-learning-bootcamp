"""Day 3 form: a churn tab and a product-image tab. Does not import Day 1 or Day 2."""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lib"))

from cnn import classify_upload

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
CHOICES = {"classic": "Classic random forest", "mlp": "MLP"}

CLASSIC = Path(os.environ.get("DAY3_CLASSIC", ROOT / "models" / "classic.joblib"))
MLP = Path(os.environ.get("DAY3_MLP", ROOT / "models" / "mlp.joblib"))
CNN = Path(os.environ.get("DAY3_CNN", ROOT / "models" / "cnn.pt"))
PATHS = {"classic": CLASSIC, "mlp": MLP}


def load_bundle(path: Path):
    if not path.is_file():
        return None, (
            f"No model file at {path.name}. Run 00_mlp_churn.ipynb, then refresh. "
            "Load only a file you fit in class. A joblib file can run code when it is loaded."
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "display_name" not in bundle:
        return None, f"{path.name} is not a Day 3 model file."
    return bundle, None


st.title("Fresh and Good, day 3")
st.caption(
    "Synthetic classroom scores for a fictional grocery. "
    "The churn tab is the morning net. The product-image tab is the small CNN. "
    "This is not a production system."
)

churn_tab, image_tab = st.tabs(["Churn", "Product image"])

with churn_tab:
    choice = st.radio(
        "Model",
        ["classic", "mlp"],
        format_func=lambda key: CHOICES[key],
        horizontal=True,
    )
    bundle, error = load_bundle(PATHS[choice])
    if error:
        st.error(error)
    else:
        st.write(f"In use: {bundle['display_name']}")
    if bundle is not None:
        with st.form("customer"):
            tenure_days = st.number_input("Tenure days", min_value=0, max_value=5000, value=180, step=1)
            orders_90d = st.number_input("Orders in the last 90 days", min_value=0, max_value=80, value=4, step=1)
            spend_thb_90d = st.number_input(
                "Spend in the last 90 days, THB", min_value=0.0, value=1500.0, step=10.0
            )
            avg_basket_thb_90d = st.number_input("Average basket, THB", min_value=0.0, value=380.0, step=10.0)
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
        if submitted:
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
        elif not error:
            st.info("Enter a customer, then score them.")

with image_tab:
    st.write("Upload a product photo. The CNN was trained on small grayscale images.")
    if not CNN.is_file():
        st.error(f"No model file at {CNN.name}. Run 03_cnn.ipynb, then refresh.")
    else:
        upload = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
        if upload is None:
            st.info("Choose an image file.")
        else:
            try:
                name, score = classify_upload(CNN, upload.getvalue())
            except Exception as exc:
                st.error(str(exc))
            else:
                st.write(f"In use: SmallCNN")
                st.metric("Class", name)
                st.metric("Score", f"{score:.1%}")
