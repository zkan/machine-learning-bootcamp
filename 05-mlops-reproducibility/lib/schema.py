import re
from datetime import datetime
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
DAY1_CSV = MODULE.parent / "01-supervised-learning" / "data" / "fresh_and_good_customers.csv"

CUSTOMERS_PARQUET = MODULE / "data" / "customers.parquet"
TRAIN_PARQUET = MODULE / "data" / "train_features.parquet"
FEAST_PARQUET = MODULE / "feature_repo" / "data" / "customer_features.parquet"
METRICS_PATH = MODULE / "data" / "metrics.json"
PARAMS_PATH = MODULE / "params.yaml"
FEATURE_REPO = MODULE / "feature_repo"
MLRUNS = MODULE / "mlruns"

SNAPSHOT = datetime(2026, 6, 30)
N_ROWS = 2400
CUSTOMER_ID = re.compile(r"FG(\d{4})")

SAFE_NUMERIC = (
    "tenure_days",
    "orders_90d",
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "days_since_last_order",
    "promo_order_share_90d",
    "app_sessions_90d",
    "support_tickets_90d",
)
INT_FEATURES = (
    "tenure_days",
    "orders_90d",
    "days_since_last_order",
    "app_sessions_90d",
    "support_tickets_90d",
)
FLOAT_FEATURES = (
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "promo_order_share_90d",
)
CATEGORICAL = (
    "district",
    "plan",
    "primary_aisle",
)
MOMENTUM = "basket_momentum_thb"
MODEL_FEATURES = SAFE_NUMERIC + (MOMENTUM,) + CATEGORICAL
TARGET = "churned"
JULY_SPEND = "next_month_spend_thb"

FEAST_FIELDS = tuple((name, "int") for name in INT_FEATURES) + tuple(
    (name, "float") for name in FLOAT_FEATURES
) + tuple((name, "str") for name in CATEGORICAL)

MODEL_NAME = "fresh-and-good-churn"
ALIAS = "champion"
EXPERIMENT = "fresh-and-good-churn"
FEATURE_VIEW = "customer_features"


def customer_id(index: int) -> str:
    return f"FG{index:04d}"


def parse_customer_id(value: str) -> str:
    match = CUSTOMER_ID.fullmatch(value.strip())
    if match is None:
        raise ValueError("Use a customer id like FG0010.")
    number = int(match.group(1))
    if number >= N_ROWS:
        raise ValueError(f"Customer ids run from FG0000 through FG{N_ROWS - 1:04d}.")
    return customer_id(number)
