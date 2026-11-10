"""Local MLflow file store. Log a run. Do not register a model; that is Day 4."""

import logging
import os
from pathlib import Path

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow

logging.getLogger("mlflow").setLevel(logging.ERROR)

EXPERIMENT = "grocery-image"


def tracking_dir(module: Path) -> Path:
    path = module / "tracking"
    path.mkdir(parents=True, exist_ok=True)
    return path


def use_store(module: Path) -> str:
    uri = tracking_dir(module).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    return uri


def log_cnn_run(module: Path, params: dict, metrics: dict, artifact: Path) -> str:
    use_store(module)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="cnn") as run:
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        if artifact is not None:
            mlflow.log_artifact(str(artifact))
        return run.info.run_id
