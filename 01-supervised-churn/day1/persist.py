"""joblib uses pickle. Loading a file runs code in that file."""

from dataclasses import dataclass
from pathlib import Path

import joblib

from day1.schema import FEATURE_COLUMNS


@dataclass(frozen=True)
class DecisionBundle:
    pipeline: object
    threshold: float
    cost_fn_thb: float
    cost_fp_thb: float
    feature_columns: tuple[str, ...]
    synthetic: bool = True


def save_bundle(path, pipeline, threshold: float, cost_fn_thb: float, cost_fp_thb: float) -> Path:
    if not 0 < float(threshold) < 1:
        raise ValueError("threshold must be between 0 and 1.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = DecisionBundle(
        pipeline=pipeline,
        threshold=float(threshold),
        cost_fn_thb=float(cost_fn_thb),
        cost_fp_thb=float(cost_fp_thb),
        feature_columns=tuple(FEATURE_COLUMNS),
        synthetic=True,
    )
    joblib.dump(bundle, destination)
    return destination


def load_bundle(path) -> DecisionBundle:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"No bundle at {source}. Finish notebook 05 first. "
            "Load only a file you trust. A joblib file can run code when it is loaded."
        )
    bundle = joblib.load(source)
    if not isinstance(bundle, DecisionBundle):
        raise ValueError("That file is not a KlongCart decision bundle.")
    if tuple(bundle.feature_columns) != tuple(FEATURE_COLUMNS):
        raise ValueError(
            "This bundle was fit on different columns. Refit it with the Day 1 feature list."
        )
    if not bundle.synthetic:
        raise ValueError("This classroom app loads synthetic bundles only.")
    if not 0 < float(bundle.threshold) < 1:
        raise ValueError("The bundle threshold must be between 0 and 1.")
    return bundle
