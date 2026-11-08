import pandas as pd

from schema import CHURN_TARGET, FEATURE_COLUMNS, SPEND_TARGET


def features_and_target(frame: pd.DataFrame, target: str = CHURN_TARGET) -> tuple[pd.DataFrame, pd.Series]:
    if target not in (CHURN_TARGET, SPEND_TARGET):
        raise ValueError("target must be churned or next_month_spend_thb.")
    missing = [name for name in (*FEATURE_COLUMNS, target) if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    labels = frame[target]
    if target == CHURN_TARGET and not set(labels.unique()).issubset({0, 1}):
        raise ValueError("churned must contain only 0 and 1.")
    if target == SPEND_TARGET and (labels < 0).any():
        raise ValueError("next_month_spend_thb must be 0 or greater.")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    if target == CHURN_TARGET:
        return features, labels.astype(int).copy()
    return features, labels.astype(float).copy()
