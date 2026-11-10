"""Two extra features, and a validation ROC-AUC for the leak comparison."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from leaks import LABELS

NEW_FEATURES = ("orders_per_week", "recency_over_45")
BASE_FEATURES = (
    "tenure_days",
    "orders_90d",
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "days_since_last_order",
    "promo_order_share_90d",
    "app_sessions_90d",
    "support_tickets_90d",
)


def add_two_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["orders_per_week"] = (out["orders_90d"] / 13).round(4)
    out["recency_over_45"] = (out["days_since_last_order"] >= 45).astype(int)
    return out


def validation_auc(frame: pd.DataFrame, columns, seed: int = 42) -> float:
    """ROC-AUC on a public validation split. This is not a hidden test."""
    use = list(columns)
    overlap = sorted(set(use) & set(LABELS))
    if overlap:
        raise ValueError(f"labels are not features: {overlap}")
    x_train, x_val, y_train, y_val = train_test_split(
        frame[use],
        frame["churned"].astype(int),
        test_size=0.25,
        random_state=seed,
        stratify=frame["churned"].astype(int),
    )
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(x_train, y_train)
    scores = pipe.predict_proba(x_val)[:, 1]
    return float(roc_auc_score(y_val, scores))
