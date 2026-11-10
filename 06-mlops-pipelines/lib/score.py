import os

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import pyarrow.parquet as pq

from drift import drift_table
from schema import ALIAS, FEATURES, LIVE, MODEL_FEATURES, MODEL_NAME, MLRUNS, REFERENCE, parse_customer_id


def _open_store() -> None:
    uri = MLRUNS.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)


def score_customer(customer_id: str) -> float:
    identity = parse_customer_id(customer_id)
    if not FEATURES.is_file():
        raise FileNotFoundError("No feature table yet. Run make run first.")
    frame = pq.read_table(FEATURES).to_pandas()
    row = frame.loc[frame["customer_id"] == identity, list(MODEL_FEATURES)]
    if row.empty:
        raise LookupError(f"No feature row for {identity}.")
    _open_store()
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return float(model.predict_proba(row)[0, 1])


def monitor_table():
    if not REFERENCE.is_file() or not LIVE.is_file():
        raise FileNotFoundError("No drift tables yet. Run make run first.")
    reference = pq.read_table(REFERENCE).to_pandas()
    current = pq.read_table(LIVE).to_pandas()
    return drift_table(reference, current)
