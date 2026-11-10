"""Clinic Dag. Three tasks, same functions as runner.py.

Airflow 3 loads this file from the dags folder. ``dag.test()`` needs
``airflow db migrate`` first, and this file has to sit in the dags folder.
"""

import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinic_tasks import count_open_tickets, publish, summarize


@dag(
    dag_id="fresh_and_good_clinic",
    schedule=None,
    start_date=pendulum.datetime(2026, 7, 1, tz="UTC"),
    catchup=False,
    tags=["clinic"],
)
def fresh_and_good_clinic():
    @task(retries=1)
    def count_open_tickets_task():
        return count_open_tickets()

    @task(retries=1)
    def summarize_task(count: int):
        return summarize(count)

    @task(retries=1)
    def publish_task(summary: dict):
        return publish(summary)

    publish_task(summarize_task(count_open_tickets_task()))


clinic = fresh_and_good_clinic()


if __name__ == "__main__":
    clinic.test()
