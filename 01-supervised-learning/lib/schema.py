from datetime import date

SNAPSHOT_DATE = date(2026, 6, 30)
FEATURE_WINDOW_START = date(2026, 4, 2)
FEATURE_WINDOW_END = SNAPSHOT_DATE
LABEL_WINDOW_START = date(2026, 7, 1)
LABEL_WINDOW_END = date(2026, 7, 30)

NUMERIC_FEATURES = (
    "tenure_days",
    "orders_90d",
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "days_since_last_order",
    "promo_order_share_90d",
    "app_sessions_90d",
    "support_tickets_90d",
)

CATEGORICAL_FEATURES = (
    "district",
    "plan",
    "primary_aisle",
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CHURN_TARGET = "churned"
SPEND_TARGET = "next_month_spend_thb"
TARGET = CHURN_TARGET
COLUMNS = FEATURE_COLUMNS + (CHURN_TARGET, SPEND_TARGET)

DISTRICTS = ("Bang Kapi", "Chatuchak", "Lat Phrao", "Huai Khwang", "Bang Na")
PLANS = ("none", "weekly", "monthly")
AISLES = ("fresh", "pantry", "household", "ready_meal")

RANDOM_STATE = 42
N_ROWS = 2400
TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_REST = 0.25

DEFAULT_COST_FN_THB = 500
DEFAULT_COST_FP_THB = 100

LEAKAGE_NAMES = (
    "customer_id",
    "email",
    "phone",
    "orders_next_30d",
    "days_until_next_order",
    "next_order_date",
    "churn_date",
)
