"""Eight odd customers, then a z-score rule and an isolation forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from segments import RFM_COLUMNS

N_PLANTED = 8
TOP_N = 16
SEED = 42


def with_planted(frame: pd.DataFrame, n: int = N_PLANTED, seed: int = SEED) -> pd.DataFrame:
    """Append n synthetic odd rows. The seed only changes how far out they sit."""
    if n < 1:
        raise ValueError("n must be at least 1")
    out = frame.copy()
    out["planted_anomaly"] = 0
    template = out.iloc[0].copy()
    bump = int(np.random.default_rng(seed).integers(0, 500))
    rows = []
    for i in range(n):
        row = template.copy()
        row["planted_anomaly"] = 1
        if i % 2 == 0:
            row["orders_90d"] = 1
            row["spend_thb_90d"] = float(80000 + bump + i * 5000)
            row["avg_basket_thb_90d"] = float(row["spend_thb_90d"])
            row["days_since_last_order"] = 1
        else:
            row["orders_90d"] = 50 + i
            row["spend_thb_90d"] = 20.0
            row["avg_basket_thb_90d"] = 20.0 / float(row["orders_90d"])
            row["days_since_last_order"] = 300
        for column in ("churned", "orders_next_30d"):
            if column in row.index:
                row[column] = 0
        for column in ("next_month_spend_thb", "july_spend_copy_thb"):
            if column in row.index:
                row[column] = 0.0
        rows.append(row)
    return pd.concat([out, pd.DataFrame(rows)], ignore_index=True)


def _scaled(frame: pd.DataFrame) -> np.ndarray:
    columns = list(RFM_COLUMNS)
    normal = frame.loc[frame["planted_anomaly"] == 0, columns]
    scaler = StandardScaler().fit(normal)
    return scaler.transform(frame[columns])


def z_strength(frame: pd.DataFrame) -> np.ndarray:
    """Largest absolute z-score across recency, orders, and spend."""
    return np.max(np.abs(_scaled(frame)), axis=1)


def forest_decision(frame: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Lower is more anomalous. Outliers are negative when contamination is 'auto'."""
    scaled = _scaled(frame)
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(scaled)
    return model.decision_function(scaled)


def top_flags(scores: np.ndarray, top_n: int, *, low_is_anomaly: bool) -> np.ndarray:
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    order = np.argsort(scores if low_is_anomaly else -scores)
    return order[:top_n]


def planted_recall(frame: pd.DataFrame, flagged: np.ndarray) -> float:
    planted = set(np.flatnonzero(frame["planted_anomaly"].to_numpy() == 1))
    if not planted:
        raise ValueError("no planted rows")
    return len(planted & set(int(i) for i in flagged)) / len(planted)
