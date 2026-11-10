import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

_LIB = Path(__file__).resolve().parents[2] / "lib"
if _LIB.is_dir() and str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"
os.environ["MLFLOW_LOG_UV_FILES"] = "false"

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schema import (
    ALIAS,
    CATEGORICAL,
    CUSTOMERS,
    EXPERIMENT,
    FEATURES,
    FLOAT_FEATURES,
    GATE,
    INT_FEATURES,
    LIVE,
    METRICS,
    MODEL_FEATURES,
    MODEL_NAME,
    MLRUNS,
    N_ROWS,
    NUMERIC,
    PREDICTIONS,
    REFERENCE,
    TARGET,
    customer_id,
    source_csv,
)

PIP_REQUIREMENTS = [
    "numpy==2.5.4",
    "pandas==2.3.3",
    "scikit-learn==1.9.1",
]


def _replace_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, path)


def _replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pandas()


def ingest() -> int:
    frame = pd.read_csv(source_csv())
    if len(frame) != N_ROWS:
        raise ValueError(f"Expected {N_ROWS} rows, found {len(frame)}.")
    columns = [pa.array([customer_id(i) for i in range(len(frame))], type=pa.string())]
    names = ["customer_id"]
    for name in INT_FEATURES:
        columns.append(pa.array(frame[name].to_numpy().astype("int64"), type=pa.int64()))
        names.append(name)
    for name in FLOAT_FEATURES:
        columns.append(pa.array(frame[name].to_numpy().astype("float64"), type=pa.float64()))
        names.append(name)
    for name in CATEGORICAL:
        columns.append(pa.array(frame[name].astype(str).tolist(), type=pa.string()))
        names.append(name)
    columns.append(pa.array(frame[TARGET].to_numpy().astype("int64"), type=pa.int64()))
    names.append(TARGET)
    table = pa.table(columns, names=names)
    _replace_table(table, CUSTOMERS)
    return table.num_rows


def validate(path: Path | None = None) -> None:
    frame = _read(CUSTOMERS if path is None else path)
    if len(frame) != N_ROWS:
        raise ValueError(f"validate expected {N_ROWS} rows, found {len(frame)}.")
    missing = [name for name in ("customer_id", *MODEL_FEATURES, TARGET) if name not in frame.columns]
    if missing:
        raise ValueError(f"validate is missing {missing}.")
    if frame["orders_90d"].isna().any():
        raise ValueError("validate found null orders_90d.")
    labels = set(frame[TARGET].astype(int).unique())
    if not labels <= {0, 1}:
        raise ValueError(f"validate found churned values {sorted(labels)}.")


def features() -> int:
    frame = _read(CUSTOMERS)
    keep = ["customer_id", *MODEL_FEATURES, TARGET]
    model_frame = frame.loc[:, keep].copy()
    for name in NUMERIC:
        model_frame[name] = model_frame[name].astype("float64")
    table = pa.Table.from_pandas(model_frame, preserve_index=False)
    _replace_table(table, FEATURES)
    reference = model_frame.loc[:, list(NUMERIC)]
    _replace_table(pa.Table.from_pandas(reference, preserve_index=False), REFERENCE)
    if not LIVE.is_file():
        shutil.copyfile(REFERENCE, LIVE)
    return len(model_frame)


def _pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def train() -> str:
    frame = _read(FEATURES)
    digest = _frame_hash(frame)
    marker = METRICS.with_name("train.json")
    if marker.is_file() and PREDICTIONS.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("features_sha256") == digest:
            print("train already matches this feature table")
            return saved["model_uri"]
    predictors = frame.loc[:, list(MODEL_FEATURES)]
    labels = frame[TARGET].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        predictors,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    pipeline = _pipeline()
    pipeline.fit(train_x, train_y)
    scores = pipeline.predict_proba(test_x)[:, 1]
    mlflow.set_tracking_uri(MLRUNS.resolve().as_uri())
    mlflow.set_registry_uri(MLRUNS.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_param("random_state", 42)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=train_x.head(2),
            pip_requirements=PIP_REQUIREMENTS,
        )
        run_id = run.info.run_id
    held = pd.DataFrame({"churned": test_y.to_numpy(), "score": scores})
    _replace_table(pa.Table.from_pandas(held, preserve_index=False), PREDICTIONS)
    _replace_text(
        marker,
        json.dumps(
            {"run_id": run_id, "model_uri": info.model_uri, "features_sha256": digest}
        )
        + "\n",
    )
    return info.model_uri


def evaluate() -> float:
    held = _read(PREDICTIONS)
    auc = float(roc_auc_score(held["churned"].astype(int), held["score"].astype(float)))
    trained = json.loads((METRICS.with_name("train.json")).read_text())
    payload = {
        "roc_auc": round(auc, 6),
        "gate": GATE,
        "passed": auc >= GATE,
        "model_uri": trained["model_uri"],
        "run_id": trained["run_id"],
    }
    _replace_text(METRICS, json.dumps(payload, indent=2) + "\n")
    return auc


def register() -> str:
    if not METRICS.is_file():
        raise FileNotFoundError("evaluate has not written metrics.json.")
    payload = json.loads(METRICS.read_text())
    auc = float(payload["roc_auc"])
    if auc < GATE:
        raise RuntimeError(f"roc_auc {auc:.4f} is below the gate {GATE:.2f}. Alias left in place.")
    mlflow.set_tracking_uri(MLRUNS.resolve().as_uri())
    mlflow.set_registry_uri(MLRUNS.resolve().as_uri())
    client = MlflowClient()
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        current = None
    if current is not None and current.run_id == payload["run_id"]:
        print(f"alias {ALIAS} already points at run {payload['run_id']}")
        return str(current.version)
    version = mlflow.register_model(payload["model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    print(f"alias {ALIAS} -> version {version.version}")
    return str(version.version)
