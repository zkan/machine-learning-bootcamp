import re
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
DAY1_CSV = MODULE.parent / "01-supervised-learning" / "data" / "fresh_and_good_customers.csv"
DAGS = MODULE / "airflow" / "dags"
VENDOR_CSV = DAGS / "vendor" / "fresh_and_good_customers.csv"

CUSTOMERS = MODULE / "data" / "customers.parquet"
FEATURES = MODULE / "data" / "features.parquet"
REFERENCE = MODULE / "data" / "reference.parquet"
LIVE = MODULE / "data" / "live.parquet"
PREDICTIONS = MODULE / "data" / "predictions.parquet"
METRICS = MODULE / "data" / "metrics.json"
MLRUNS = MODULE / "mlruns"

N_ROWS = 2400
CUSTOMER_ID = re.compile(r"FG(\d{4})")
GATE = 0.70
PSI_LIMIT = 0.2
ATTEMPTS = 3

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
NUMERIC = INT_FEATURES + FLOAT_FEATURES
MODEL_FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"
MODEL_NAME = "fresh-and-good-churn"
ALIAS = "champion"
EXPERIMENT = "fresh-and-good-pipeline"


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


def source_csv() -> Path:
    if VENDOR_CSV.is_file():
        return VENDOR_CSV
    if DAY1_CSV.is_file():
        return DAY1_CSV
    raise FileNotFoundError(
        "Missing the Day 1 CSV. Run make vendor from 06-mlops-pipelines, or keep the Day 1 folder beside this one."
    )
