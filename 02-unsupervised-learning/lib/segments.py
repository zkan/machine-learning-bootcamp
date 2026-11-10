"""Scale recency, frequency, and spend, then cluster with k-means."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_COLUMNS = ("days_since_last_order", "orders_90d", "spend_thb_90d")
K_GRID = (2, 3, 4, 5, 6)
SEED = 42


def _scaled(frame: pd.DataFrame):
    return StandardScaler().fit_transform(frame[list(RFM_COLUMNS)])


def k_table(frame: pd.DataFrame, ks=K_GRID, seed: int = SEED) -> pd.DataFrame:
    scaled = _scaled(frame)
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(scaled)
        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(scaled, labels)),
            }
        )
    return pd.DataFrame(rows)


def fit_labels(frame: pd.DataFrame, k: int, seed: int = SEED):
    scaled = _scaled(frame)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return model.fit_predict(scaled)


def profile(frame: pd.DataFrame, labels) -> pd.DataFrame:
    out = frame.copy()
    out["segment"] = labels
    summary = (
        out.groupby("segment", as_index=False)
        .agg(
            customers=("churned", "size"),
            churn_rate=("churned", "mean"),
            recency_days=("days_since_last_order", "mean"),
            orders_90d=("orders_90d", "mean"),
            spend_thb_90d=("spend_thb_90d", "mean"),
        )
        .sort_values("segment")
    )
    summary["churn_rate"] = summary["churn_rate"].round(4)
    for column in ("recency_days", "orders_90d", "spend_thb_90d"):
        summary[column] = summary[column].round(2)
    return summary.reset_index(drop=True)


def best_k(table: pd.DataFrame) -> int:
    """Highest silhouette. The smaller k wins a tie."""
    ordered = table.sort_values(["silhouette", "k"], ascending=[False, True])
    return int(ordered.iloc[0]["k"])
