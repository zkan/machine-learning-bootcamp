from datetime import datetime

import pandas as pd
from feast import FeatureStore

from features import read_customers
from schema import FEATURE_REPO, FEATURE_VIEW, FEAST_FIELDS, parse_customer_id


def feature_names() -> list[str]:
    return [f"{FEATURE_VIEW}:{name}" for name, _kind in FEAST_FIELDS]


def open_store() -> FeatureStore:
    if not (FEATURE_REPO / "data" / "registry.db").is_file():
        raise FileNotFoundError(
            "Feast registry is missing. From this folder run make features and make feast."
        )
    return FeatureStore(repo_path=str(FEATURE_REPO))


def online_customer(customer_id: str) -> pd.DataFrame:
    identity = parse_customer_id(customer_id)
    result = open_store().get_online_features(
        features=feature_names(),
        entity_rows=[{"customer_id": identity}],
    )
    frame = pd.DataFrame(result.to_dict())
    if frame.empty or pd.isna(frame.loc[0, "orders_90d"]):
        raise LookupError(f"No online row for {identity}.")
    return frame


def historical_orders(customer_id: str) -> pd.DataFrame:
    identity = parse_customer_id(customer_id)
    entity_df = pd.DataFrame(
        {
            "customer_id": [identity, identity],
            "event_timestamp": [
                datetime(2026, 6, 1),
                datetime(2026, 7, 1),
            ],
        }
    )
    job = open_store().get_historical_features(
        entity_df=entity_df,
        features=[f"{FEATURE_VIEW}:orders_90d"],
    )
    return job.to_df()


def customer_source() -> pd.DataFrame:
    return read_customers()
