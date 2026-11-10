"""Held-out cup scores. The learner notebook writes predictions. Labels stay out of git."""

from pathlib import Path

import pandas as pd

from cnn import FASHION_HOLD_LIMIT, FASHION_TRAIN_LIMIT, FASHION_VAL_LIMIT
from images import load_dataset, split_dataset, take


def class_parts(source: str = "auto"):
    """Train and validation keep their labels. The held-out return value is images only."""
    bundle = load_dataset(source)
    train, val, hold = split_dataset(bundle)
    if bundle["source"] == "fashion":
        train = take(*train, FASHION_TRAIN_LIMIT)
        val = take(*val, FASHION_VAL_LIMIT)
        hold = take(*hold, FASHION_HOLD_LIMIT)
    return bundle, train, val, hold[0]


def instructor_labels(source: str = "auto") -> pd.DataFrame:
    """Same held-out rows as class_parts. For the teaching team, not the notebook."""
    bundle = load_dataset(source)
    _train, _val, hold = split_dataset(bundle)
    if bundle["source"] == "fashion":
        hold = take(*hold, FASHION_HOLD_LIMIT)
    return label_frame(hold[1])


def prediction_frame(predicted) -> pd.DataFrame:
    return pd.DataFrame({"row": list(range(len(predicted))), "predicted": list(map(int, predicted))})


def label_frame(labels) -> pd.DataFrame:
    return pd.DataFrame({"row": list(range(len(labels))), "label": list(map(int, labels))})


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def score_files(predictions: Path, labels: Path) -> float:
    guessed = pd.read_csv(predictions)
    truth = pd.read_csv(labels)
    merged = guessed.merge(truth, on="row")
    if len(merged) != len(truth):
        raise ValueError("predictions and labels do not cover the same rows")
    return float((merged["predicted"] == merged["label"]).mean())
