"""Local MLflow file store for this folder. No server and no cloud.

MLflow 3.18 refuses the filesystem backend unless MLFLOW_ALLOW_FILE_STORE is
true. Set that before opening the store. The files live in ``tracking/``.
"""

import logging
import os
from pathlib import Path

# Read by MLflow when the store is opened. Set these before importing mlflow.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow

logging.getLogger("mlflow").setLevel(logging.ERROR)

EXPERIMENT_CUP = "network-tuning-cup"
EXPERIMENT_CHOICE = "churn-choice"
REGISTERED_NAME = "fresh-and-good-churn"


def tracking_dir(module: Path) -> Path:
    path = module / "tracking"
    path.mkdir(parents=True, exist_ok=True)
    return path


def use_store(module: Path) -> str:
    uri = tracking_dir(module).resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    return uri


def log_cup_run(module: Path, result: dict) -> str:
    """Log one ticket model. Returns the run id."""
    use_store(module)
    mlflow.set_experiment(EXPERIMENT_CUP)
    with mlflow.start_run() as run:
        mlflow.log_param("hidden_layer_sizes", str(result["hidden_layer_sizes"]))
        mlflow.log_param("learning_rate_init", result["learning_rate_init"])
        mlflow.log_metric("validation_accuracy", result["validation_accuracy"])
        mlflow.log_metric("training_seconds", result["training_seconds"])
        return run.info.run_id


def log_confusion_csv(module: Path, run_id: str, frame, path: Path) -> None:
    use_store(module)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    mlflow.log_artifact(str(path), run_id=run_id)


def leaderboard(module: Path):
    """Best validation accuracy first. A tie goes to the faster run."""
    import pandas as pd

    use_store(module)
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_CUP],
        order_by=["metrics.validation_accuracy DESC"],
    )
    if runs.empty:
        return runs
    board = pd.DataFrame(
        {
            "hidden_layer_sizes": runs["params.hidden_layer_sizes"],
            "learning_rate_init": runs["params.learning_rate_init"].astype(float),
            "validation_accuracy": runs["metrics.validation_accuracy"],
            "training_seconds": runs["metrics.training_seconds"],
            "run_id": runs["run_id"],
        }
    )
    return board.sort_values(
        ["validation_accuracy", "training_seconds"],
        ascending=[False, True],
    ).reset_index(drop=True)


def register_pipeline(module: Path, pipeline, model_key: str) -> str:
    """Log the pipeline and register it. Returns the new version number."""
    use_store(module)
    mlflow.set_experiment(EXPERIMENT_CHOICE)
    with mlflow.start_run(run_name=model_key):
        # MLflow 3.18 defaults to skops, which refuses MLPClassifier's Adam optimizer
        # unless that private type is listed. Pickle is a documented format for a model
        # this class just fit.
        info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=REGISTERED_NAME,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
    return str(info.registered_model_version)
