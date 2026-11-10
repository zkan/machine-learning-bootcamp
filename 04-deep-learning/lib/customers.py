"""Day 1 customers, read only, and a random forest versus an MLP.

The forest settings match ``01-supervised-learning/lib/pipelines.py``. This
module does not import that file and does not write the Day 1 CSV.
"""

import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODULE = Path(__file__).resolve().parents[1]
DAY1_CSV = MODULE.parent / "01-supervised-learning" / "data" / "fresh_and_good_customers.csv"

NUMERIC = (
    "tenure_days",
    "orders_90d",
    "spend_thb_90d",
    "avg_basket_thb_90d",
    "days_since_last_order",
    "promo_order_share_90d",
    "app_sessions_90d",
    "support_tickets_90d",
)
CATEGORICAL = ("district", "plan", "primary_aisle")
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"
SEED = 42
# Same fractions as Day 1: 20% test, then 25% of the rest as validation.
TEST_FRACTION = 0.20
VALIDATION_FRACTION_OF_REST = 0.25

CLASSIC_KEY = "classic"
MLP_KEY = "mlp"


def load_customers(path: Path = DAY1_CSV) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [name for name in (*FEATURES, TARGET) if name not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return frame


def split_frame(frame: pd.DataFrame, seed: int = SEED):
    """Train, validation, and test. Do not fit on the test rows."""
    labels = frame[TARGET].astype(int)
    if labels.nunique() < 2:
        raise ValueError("churned needs both classes before the split.")
    rest, test = train_test_split(
        frame,
        test_size=TEST_FRACTION,
        random_state=seed,
        stratify=labels,
    )
    train, validation = train_test_split(
        rest,
        test_size=VALIDATION_FRACTION_OF_REST,
        random_state=seed,
        stratify=rest[TARGET].astype(int),
    )
    return train, validation, test


def xy(frame: pd.DataFrame):
    return frame.loc[:, list(FEATURES)], frame[TARGET].astype(int)


def preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(NUMERIC)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL),
            ),
        ]
    )


def forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    min_samples_leaf=5,
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )


def mlp_pipeline(
    hidden_layer_sizes=(32,),
    learning_rate_init: float = 0.001,
    max_iter: int = 200,
    early_stopping: bool = False,
    solver: str = "adam",
    scale_numeric: bool = True,
    alpha: float = 0.0001,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", preprocessor(scale_numeric=scale_numeric)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    solver=solver,
                    alpha=alpha,
                    random_state=SEED,
                ),
            ),
        ]
    )


def roc_auc(pipeline: Pipeline, frame: pd.DataFrame) -> float:
    features, labels = xy(frame)
    scores = pipeline.predict_proba(features)[:, 1]
    return float(roc_auc_score(labels, scores))


def fit_timed(pipeline: Pipeline, train: pd.DataFrame) -> float:
    features, labels = xy(train)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(features, labels)
    return time.perf_counter() - started


def compare_models(train: pd.DataFrame, validation: pd.DataFrame, hidden_layer_sizes=(32,)):
    """Fit the Day 1 forest and one MLP. Scores are validation ROC-AUC."""
    forest = forest_pipeline()
    forest_seconds = fit_timed(forest, train)
    network = mlp_pipeline(hidden_layer_sizes=hidden_layer_sizes)
    mlp_seconds = fit_timed(network, train)
    return {
        "classic_validation_roc_auc": roc_auc(forest, validation),
        "mlp_validation_roc_auc": roc_auc(network, validation),
        "classic_train_roc_auc": roc_auc(forest, train),
        "mlp_train_roc_auc": roc_auc(network, train),
        "classic_seconds": forest_seconds,
        "mlp_seconds": mlp_seconds,
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        CLASSIC_KEY: forest,
        MLP_KEY: network,
    }


def save_bundle(path: Path, pipeline: Pipeline, key: str, display_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"key": key, "display_name": display_name, "pipeline": pipeline},
        path,
    )
