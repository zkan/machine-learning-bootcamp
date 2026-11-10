# Day 5: MLOps pipelines

Day 5 of the Machine Learning Bootcamp. You run the Fresh and Good training tasks in order, keep a weak model off the alias, and read a drift check before you decide to retrain or roll back.

If Day 4's Airflow clinic was cut, start with 15 minutes of [Airflow up](../05-mlops-reproducibility/airflow/README.md). Then come back here.

- 5.1 [Run the pipeline](00_pipeline.ipynb)
- 5.2 [Break the pipeline](01_bug_bounty.ipynb)
- 5.3 [Drift check](02_monitor.ipynb)
- 5.4 [On call](03_oncall.ipynb)
- 5.5 [Score and monitor](app.py)
- [Plain runner and the UI path](airflow/README.md)
- [Incident, severity, injects, runbook, postmortem](oncall/)
- [Data assumptions](DATA_ASSUMPTIONS.md)

Run the commands on this page from `06-mlops-pipelines` (this folder), unless a step says otherwise.

This folder uses Python 3.14.10, the same pin as Day 4. `uv sync` reads `.python-version`. You do not change the repo `mise.toml`.

## What you build

The form scores one customer id from the model the pipeline registered. The alias is `champion` on `fresh-and-good-churn`. A second page compares live features with the training reference.

The source table is still the Day 1 CSV. This day does not generate a new table. Ids are `FG0000` through `FG2399` in file order. Read `DATA_ASSUMPTIONS.md`.

## Create the environment

```sh
mise install
uv sync --frozen
uv run python -m ipykernel install --user --name fresh-and-good-day5 --display-name "Python (fresh-and-good-day5)"
```

```sh
uv run python -c "import sys; print(sys.version)"
```

Expect a version line that starts with `3.14.10`.

## Run the tasks

```sh
make run
```

The runner calls ingest, validate, features, train, evaluate, and register. Each task gets three attempts. Files are replaced in place, so a retry does not append a second copy. A second `make run` keeps the alias on the same MLflow run when the feature table has not changed.

Evaluate writes `data/metrics.json`. With the shipped code the held-out ROC-AUC is `0.793613`. The gate is `0.70`. The alias moves only when the score is at least the gate.

`make run` prints `register` and the alias version on the first success. The second run prints `alias champion already points at run`.

## The dag relay

`airflow/dags/churn_train.py` is the same six tasks, with two retries and a two-minute delay, on a daily schedule. Pairs can each own one task, then trigger `churn_train` in the UI. The plain runner is the path that works with no Docker. See `airflow/README.md` for the UI, the image pull, and the two limits of the stock compose file.

## Bug bounty

`airflow/dags/churn_train_buggy.py` has five bugs. Notebook `01_bug_bounty.ipynb` asks for five sentences. The answer key is with the instructor, not in this folder.

## Drift

Open the monitor page after `make run`. Every PSI is `0.0` while `data/live.parquet` is still the copy of the reference. A column at or above `0.2` needs a decision. Notebook `02_monitor.ipynb` asks you to choose `retrain` or `rollback`.

The engineer stretch is `validate`. It fails the run when the table has the wrong row count, a null `orders_90d`, or a `churned` value other than 0 or 1. Point it at a bad file and watch the task stop.

## On call

The incident is "Churn scores all zero since 03:00." Read `oncall/incident.md`, pick a level from `oncall/severity.md`, and take the inject cards in `oncall/injects.md`. Copy `oncall/runbook_template.md` and `oncall/postmortem_template.md` when you write them up. The postmortem is blameless.

For the shop, say what a customer would notice and what you will do until scores look right. That note is `customer_message` in the on-call notebook.

## Run the form

```sh
uv run python -m streamlit run app.py --server.port 8505
```

Score `FG0010`. The churn probability is `32.7%`. The health check is:

```sh
curl --fail http://127.0.0.1:8505/_stcore/health
```

Expect `ok`.

Switch the page control to Monitor. With the shipped live file, the page says no column is over the PSI line.

## Library versions

Checked on Python 3.14.10.

| Package | Version |
| --- | --- |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| pyarrow | 22.0.0 |
| scikit-learn | 1.9.1 |
| mlflow | 3.18.0 |
| streamlit | 1.67.0 |

## Troubleshooting

`ModuleNotFoundError` for `schema` or `churn_tasks`. Run commands as `make run` or with `PYTHONPATH=lib:airflow/dags`.

MLflow may print a hint about integer columns. Counts in this table are whole numbers. The form still scores.

Docker is missing, or the container cannot import pandas. Use `make run`. The UI is optional and one laptop per team is enough.
