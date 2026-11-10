"""Learning rate, scaling, early stopping, and a learning curve.

The morning comparison uses Adam. These tables use the same columns. The
learning-rate table uses SGD so the step size is the rate you set.
"""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import learning_curve

from customers import SEED, mlp_pipeline, roc_auc


def _row(name, pipeline, train, validation, **extra):
    row = {
        "setting": name,
        "train_roc_auc": roc_auc(pipeline, train),
        "validation_roc_auc": roc_auc(pipeline, validation),
        "iterations": int(pipeline.named_steps["model"].n_iter_),
    }
    row.update(extra)
    return row


def learning_rate_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """SGD on scaled columns. Three rates, same hidden layer."""
    rows = []
    for rate in (0.001, 0.01, 1.0):
        pipeline = mlp_pipeline(
            hidden_layer_sizes=(16,),
            learning_rate_init=rate,
            max_iter=40,
            solver="sgd",
            scale_numeric=True,
        )
        _fit(pipeline, train)
        rows.append(_row(f"sgd {rate}", pipeline, train, validation, learning_rate_init=rate))
    return pd.DataFrame(rows)


def scaling_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Same SGD rate. One run scales the numeric columns, one does not."""
    rows = []
    for scale in (True, False):
        pipeline = mlp_pipeline(
            hidden_layer_sizes=(16,),
            learning_rate_init=0.01,
            max_iter=40,
            solver="sgd",
            scale_numeric=scale,
        )
        _fit(pipeline, train)
        rows.append(_row("scaled" if scale else "not scaled", pipeline, train, validation))
    return pd.DataFrame(rows)


def stopping_table(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """A wider Adam net, with and without early stopping."""
    rows = []
    for stop in (False, True):
        pipeline = mlp_pipeline(
            hidden_layer_sizes=(64, 64),
            learning_rate_init=0.001,
            max_iter=80,
            early_stopping=stop,
            solver="adam",
            alpha=1e-5,
        )
        _fit(pipeline, train)
        rows.append(_row("early stopping" if stop else "no early stopping", pipeline, train, validation))
    return pd.DataFrame(rows)


def curve_table(train: pd.DataFrame) -> pd.DataFrame:
    """sklearn learning_curve on the train rows only. The test rows stay unused.

    The cv score is a 3-fold score inside the train rows, not the Day 3
    validation split.
    """
    from customers import xy

    features, labels = xy(train)
    pipeline = mlp_pipeline(hidden_layer_sizes=(16,), max_iter=30, solver="adam")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        sizes, train_scores, cv_scores = learning_curve(
            pipeline,
            features,
            labels,
            train_sizes=(0.25, 0.5, 0.75, 1.0),
            cv=3,
            scoring="roc_auc",
            shuffle=True,
            random_state=SEED,
            n_jobs=1,
        )
    return pd.DataFrame(
        {
            "train_rows": sizes.astype(int),
            "train_roc_auc": train_scores.mean(axis=1),
            "cv_roc_auc": cv_scores.mean(axis=1),
        }
    )


def _fit(pipeline, train: pd.DataFrame) -> None:
    from customers import xy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(*xy(train))
