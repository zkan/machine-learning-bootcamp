import json

import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from files import write_text
from registry import load_params, register_champion
from schema import CATEGORICAL, METRICS_PATH, MODEL_FEATURES, SAFE_NUMERIC, TARGET, TRAIN_PARQUET, MOMENTUM


def make_pipeline(C: float) -> Pipeline:
    numeric = list(SAFE_NUMERIC + (MOMENTUM,))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
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
            ("model", LogisticRegression(C=C, max_iter=500, solver="lbfgs")),
        ]
    )


def fit_churn(frame: pd.DataFrame, test_size: float, random_state: int, C: float):
    features = frame.loc[:, list(MODEL_FEATURES)]
    labels = frame[TARGET].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    pipeline = make_pipeline(C)
    pipeline.fit(train_x, train_y)
    scores = pipeline.predict_proba(test_x)[:, 1]
    auc = float(roc_auc_score(test_y, scores))
    return pipeline, auc, train_x.head(2)


def write_metrics(payload: dict) -> None:
    write_text(METRICS_PATH, json.dumps(payload, indent=2) + "\n")


def main() -> None:
    params = load_params()
    frame = pq.read_table(TRAIN_PARQUET).to_pandas()
    pipeline, auc, sample = fit_churn(
        frame,
        test_size=float(params["train"]["test_size"]),
        random_state=int(params["train"]["random_state"]),
        C=float(params["train"]["C"]),
    )
    version = register_champion(pipeline, sample, auc, params)
    payload = {
        "roc_auc": round(auc, 6),
        "rows": int(len(frame)),
        "test_size": float(params["train"]["test_size"]),
        "model_name": params["registry"]["model_name"],
        "alias": params["registry"]["alias"],
        "version": version,
    }
    write_metrics(payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
