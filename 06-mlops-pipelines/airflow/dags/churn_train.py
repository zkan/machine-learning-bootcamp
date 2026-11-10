from datetime import timedelta

import pendulum
from airflow.sdk import dag, task

from churn_tasks import evaluate, features, ingest, register, train, validate


@dag(
    schedule="@daily",
    start_date=pendulum.datetime(2026, 6, 30, tz="UTC"),
    catchup=False,
    tags=["fresh-and-good"],
)
def churn_train():
    @task(retries=2, retry_delay=timedelta(minutes=2))
    def ingest_task():
        ingest()

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def validate_task():
        validate()

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def features_task():
        features()

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
        >> evaluate_task()
        >> register_task()
    )


churn_train()
