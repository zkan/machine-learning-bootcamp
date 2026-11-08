"""joblib uses pickle. Loading a file runs code in that file."""

from dataclasses import dataclass
from pathlib import Path

import joblib

from schema import CHURN_TARGET, FEATURE_COLUMNS, SPEND_TARGET

TASK_CHURN = "churn"
TASK_SPEND = "spend"


@dataclass(frozen=True)
class DecisionBundle:
    pipeline: object
    task: str
    target: str
    feature_columns: tuple[str, ...]
    synthetic: bool = True
    threshold: float | None = None
    cost_fn_thb: float | None = None
    cost_fp_thb: float | None = None


def save_bundle(path, pipeline, threshold: float, cost_fn_thb: float, cost_fp_thb: float) -> Path:
    if not 0 < float(threshold) < 1:
        raise ValueError("threshold must be between 0 and 1.")
    return _dump(
        path,
        DecisionBundle(
            pipeline=pipeline,
            task=TASK_CHURN,
            target=CHURN_TARGET,
            feature_columns=tuple(FEATURE_COLUMNS),
            synthetic=True,
            threshold=float(threshold),
            cost_fn_thb=float(cost_fn_thb),
            cost_fp_thb=float(cost_fp_thb),
        ),
    )


def save_spend_bundle(path, pipeline) -> Path:
    return _dump(
        path,
        DecisionBundle(
            pipeline=pipeline,
            task=TASK_SPEND,
            target=SPEND_TARGET,
            feature_columns=tuple(FEATURE_COLUMNS),
            synthetic=True,
        ),
    )


def load_bundle(path, expected_task: str | None = None) -> DecisionBundle:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"No bundle at {source}. Finish the matching notebook first. "
            "Load only a file you trust. A joblib file can run code when it is loaded."
        )
    bundle = joblib.load(source)
    if not isinstance(bundle, DecisionBundle):
        raise ValueError("That file is not a KlongCart decision bundle.")
    if bundle.task not in (TASK_CHURN, TASK_SPEND):
        raise ValueError("The bundle task must be churn or spend.")
    if expected_task is not None and bundle.task != expected_task:
        raise ValueError(f"This file is a {bundle.task} bundle, not a {expected_task} bundle.")
    if tuple(bundle.feature_columns) != tuple(FEATURE_COLUMNS):
        raise ValueError(
            "This bundle was fit on different columns. Refit it with the supervised feature list."
        )
    if bundle.task == TASK_CHURN and bundle.target != CHURN_TARGET:
        raise ValueError("A churn bundle must target churned.")
    if bundle.task == TASK_SPEND and bundle.target != SPEND_TARGET:
        raise ValueError("A spend bundle must target next_month_spend_thb.")
    if not bundle.synthetic:
        raise ValueError("This classroom app loads synthetic bundles only.")
    if bundle.task == TASK_CHURN and not 0 < float(bundle.threshold) < 1:
        raise ValueError("The churn threshold must be between 0 and 1.")
    return bundle


def _dump(path, bundle: DecisionBundle) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination
