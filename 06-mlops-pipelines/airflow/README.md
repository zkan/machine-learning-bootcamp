# Airflow for the training dag

If you skipped Day 4's Airflow clinic, spend the first 15 minutes on that clinic before this dag. The steps and the memory floor are in `05-mlops-reproducibility/airflow/README.md`. This folder has its own copy of the same official Airflow 3.4.0 `docker-compose.yaml`, unmodified, from:

https://airflow.apache.org/docs/apache-airflow/3.4.0/docker-compose.yaml

Docker still needs at least 4 GB of memory, 2 CPUs, and 10 GB of disk. One machine per team is enough. The plain runner does not need Docker and does not install Airflow.

## Plain runner

From `06-mlops-pipelines`:

```sh
make run
```

That calls ingest, validate, features, train, evaluate, and register, in that order, with two retries per task (`retries=2`).

## UI path

The stock compose file mounts `dags`, `logs`, `plugins`, and `config`. It does not mount the rest of this repo, and the Airflow image does not include pandas, scikit-learn, or MLflow. Two limits follow.

The tasks read the Day 1 CSV. From this folder, copy it into the mounted dags folder:

```sh
make vendor
```

Packages use the official quick-check variable, which installs on every container start. Add this line to `.env` only if you accept that install cost:

```text
_PIP_ADDITIONAL_REQUIREMENTS=pandas==2.3.3 scikit-learn==1.9.1 mlflow==3.18.0 pyarrow==22.0.0
```

The image's Python has to be able to install those pins. If pip fails, use `make run` and skip the UI.

Then, from `06-mlops-pipelines/airflow`:

```sh
mkdir -p dags logs plugins config
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up airflow-init
docker compose up
```

Open http://localhost:8080 and sign in as `airflow` / `airflow`. Trigger `churn_train`. Leave `churn_train_buggy` for the bounty. Do not unpause the buggy dag during the demo.

Health check from the official file:

```sh
curl --fail http://localhost:8080/api/v2/monitor/health
```
