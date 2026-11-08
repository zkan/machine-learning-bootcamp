import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE

FOREST_SEARCH = {
    "model__n_estimators": [40, 80, 120],
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20],
}

RIDGE_SEARCH = {"model__alpha": [0.1, 1.0, 10.0, 50.0]}


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )


def make_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("model", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )


def make_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def tune_forest(
    features: pd.DataFrame,
    labels: pd.Series,
    n_iter: int = 8,
    seed: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    search = RandomizedSearchCV(
        estimator=make_forest_pipeline(),
        param_distributions=FOREST_SEARCH,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        refit=True,
        random_state=seed,
        n_jobs=1,
    )
    search.fit(features, labels)
    return search


def make_dummy_spend_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )


def make_ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def tune_ridge(
    features: pd.DataFrame,
    labels: pd.Series,
    n_iter: int = 4,
    seed: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1.")
    search = RandomizedSearchCV(
        estimator=make_ridge_pipeline(),
        param_distributions=RIDGE_SEARCH,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=3,
        refit=True,
        random_state=seed,
        n_jobs=1,
    )
    search.fit(features, labels)
    return search
