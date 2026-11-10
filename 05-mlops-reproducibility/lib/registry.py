import os

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_UV_AUTO_DETECT"] = "false"
os.environ["MLFLOW_LOG_UV_FILES"] = "false"

import mlflow
import yaml
from mlflow.tracking import MlflowClient

PIP_REQUIREMENTS = [
    "numpy==2.5.4",
    "pandas==2.3.3",
    "scikit-learn==1.9.1",
]

from schema import ALIAS, EXPERIMENT, MLRUNS, MODEL_NAME, PARAMS_PATH


def tracking_uri(path=None) -> str:
    chosen = MLRUNS if path is None else path
    return chosen.resolve().as_uri()


def open_store(path=None) -> str:
    uri = tracking_uri(path)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def load_params() -> dict:
    return yaml.safe_load(PARAMS_PATH.read_text())


def model_uri(name: str = MODEL_NAME, alias: str = ALIAS) -> str:
    return f"models:/{name}@{alias}"


def load_champion(path=None):
    open_store(path)
    return mlflow.sklearn.load_model(model_uri())


def register_champion(pipeline, sample, roc_auc: float, params: dict, path=None) -> str:
    open_store(path)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "C": params["train"]["C"],
                "test_size": params["train"]["test_size"],
                "random_state": params["train"]["random_state"],
            }
        )
        mlflow.log_metric("roc_auc", float(roc_auc))
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=sample,
            pip_requirements=PIP_REQUIREMENTS,
        )
    version = mlflow.register_model(info.model_uri, MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, ALIAS, version.version)
    return str(version.version)
