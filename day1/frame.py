import pandas as pd

from day1.schema import FEATURE_COLUMNS, TARGET


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [name for name in (*FEATURE_COLUMNS, TARGET) if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    labels = frame[TARGET]
    if not set(labels.unique()).issubset({0, 1}):
        raise ValueError("churned must contain only 0 and 1.")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    return features, labels.astype(int).copy()
