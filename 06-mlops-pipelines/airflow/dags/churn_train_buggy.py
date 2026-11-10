from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

from churn_tasks import evaluate, features, register, train


@dag(
    schedule=None,
    start_date=pendulum.datetime(2026, 6, 30, tz="UTC"),
    catchup=False,
    tags=["fresh-and-good", "bounty"],
)
def churn_train_buggy():
    @task(retries=0)
    def ingest_task():
        from pathlib import Path

        import pandas as pd

        from schema import source_csv

        frame = pd.read_csv(source_csv())
        path = Path(__file__).resolve().parents[2] / "data" / "customers_buggy.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, mode="a", index=False)

    @task(retries=0)
    def validate_task():
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "data" / "customers.csv"
        frame = __import__("pandas").read_csv(path)
        if len(frame) == 0:
            raise ValueError("empty")

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def features_task():
        features()
        import pyarrow.parquet as pq

        from schema import FEATURES, TARGET

        frame = pq.read_table(FEATURES).to_pandas()
        leaked = frame.copy()
        leaked["label_copy"] = leaked[TARGET]
        leaked.to_parquet(FEATURES, index=False)

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def train_task():
        train()

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def evaluate_task():
        evaluate()

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def register_task():
        register()

    (
        ingest_task()
        >> validate_task()
        >> features_task()
        >> train_task()
        >> register_task()
        >> evaluate_task()
    )


churn_train_buggy()
