# Day 4: MLOps reproducibility

Day 4 of the Machine Learning Bootcamp. You version the Fresh and Good churn check so a teammate can run it again. The form loads a model from the MLflow registry by name and alias, and it loads customer features from Feast by customer id.

- 4.1 [Reproduce one run](00_reproduce.ipynb)
- 4.2 [Find the model](01_registry.ipynb)
- 4.3 [Look up one customer](02_feast.ipynb)
- 4.4 [Training-serving skew hunt](03_skew_hunt.ipynb)
- 4.5 [Score one customer](app.py)
- 4.6 [Airflow up](airflow/README.md)
- [Data assumptions](DATA_ASSUMPTIONS.md)
- [Resume point](checkpoints/README.md)

Run the commands on this page from `05-mlops-reproducibility` (this folder), unless a step says otherwise.

## What you build

Build a form that scores one customer id. The model name is `fresh-and-good-churn`. The alias is `champion`. Feast returns the feature row for that id.

The source table is Day 1's CSV. This day does not generate a new table. Read `DATA_ASSUMPTIONS.md` before you model. The Day 1 file has no customer id. This day assigns `FG0000` through `FG2399` from the file's row order.

## Python 3.14.10

Days 1 to 3 use Python 3.15.2. This day uses Python 3.14.10.

Feast 0.67.0 requires `pandas<3`. The newest pandas 2 release with a wheel is pandas 2.3.3, and that wheel stops at Python 3.14. DVC 3.67.1 lists Python 3.14 as its newest classifier and does not list 3.15. MLflow 3.18.0 is the same release Day 3 of a longer course would use. It is a pure Python wheel and allows 3.10 and newer, but this folder follows the Feast and DVC pin so one environment can import all three.

`.python-version` in this folder is `3.14.10`. `uv sync` installs that interpreter. You do not change the repo `mise.toml`.

## Create the environment

Install mise 2026.11.3 the same way as Day 1, if it is not on your `PATH` yet. From this folder:

```sh
mise install
uv sync --frozen
uv run python -m ipykernel install --user --name fresh-and-good-day4 --display-name "Python (fresh-and-good-day4)"
```

Check the interpreter:

```sh
uv run python -c "import sys; print(sys.version)"
```

Expect a version line that starts with `3.14.10`.

## Reproduce the run

```sh
uv run dvc repro
uv run dvc repro
```

The first command runs `prepare`, `features`, and `train`. The second command is a no-op:

```text
Stage 'prepare' didn't change, skipping
Stage 'features' didn't change, skipping
Stage 'train' didn't change, skipping
Data and pipelines are up to date.
```

`train` prints a JSON object. With `params.yaml` left as it is shipped, the held-out ROC-AUC is `0.998132` on 2,400 rows and a test fraction of 0.2. A score that high still has to pass the skew hunt.

MLflow may print a hint about integer columns. Counts in this table are whole numbers. The form still scores a customer. You can ignore that hint today.

## Reproduce-my-run relay

Work in pairs.

Team A, from this folder, after `dvc repro`:

```sh
uv run dvc push
```

That fills `shared-dvc-remote` in this folder. Give that folder to Team B on a shared disk or a USB drive. Team B needs this git repo at the same commit, with the remote folder beside `.dvc`.

Team B:

```sh
uv run dvc pull
uv run dvc repro
```

Open MLflow only after both sides have a run. The comparison that matters is the data hash in `dvc.lock`, the params in `params.yaml`, and the package pins in `uv.lock`. Matching code with a different CSV, a different `C`, or a different Python is a different run.

The point of the relay is that reproducible means code, data, params, and the environment, all versioned.

## Auditor note

Notebook `00_reproduce.ipynb` asks you to name, for an auditor, where the code, the data, the params, and the environment live. Fill `auditor_note` before you leave that notebook.

## Find the model

`make repro` registers `fresh-and-good-churn` and points the alias `champion` at the new version. The form loads `models:/fresh-and-good-churn@champion`.

Notebook `01_registry.ipynb` is the model race. Three versions are on the page. Return the version with the highest `roc_auc`.

MLflow 3.18 keeps this registry in a local file store under `mlruns`. The file store is off unless `MLFLOW_ALLOW_FILE_STORE=true`. `lib/registry.py` sets that variable. Do not switch this classroom store to a database today.

## Feast

From this folder, after `dvc repro`:

```sh
make feast
```

That runs `feast apply` and `feast materialize` for 1 June 2026 through 2 July 2026. The entity is `customer`. The join key is `customer_id`. The offline store is the parquet file. The online store is sqlite.

In `02_feast.ipynb`, a point-in-time read for `FG0010` returns one row, dated 1 July 2026, with `orders_90d` 2. The 1 June request is before the snapshot, so it is absent. The online row for `FG0010` has plan `monthly` and `orders_90d` 2.

`store_roles` in that notebook is the exercise.

## Skew hunt

Training and serving build different rows for the same customer. One training column reads a July label. Feast stores the June snapshot columns, and the form uses the serving row. Notebook `03_skew_hunt.ipynb` asks you to name that column from `lib/features.py`.

## Run the form

```sh
uv run python -m streamlit run app.py
```

Open the URL Streamlit prints. Enter `FG0010` and score them. The churn probability is `95.5%`. The health check is:

```sh
curl --fail http://127.0.0.1:8501/_stcore/health
```

Expect `ok`.

If the registry or the Feast sqlite file is missing, the page tells you to run `make checkpoint`. That target is the wiring resume point in `checkpoints/README.md`. It does not answer the notebooks.

## Engineer stretch

Change `train.C` in `params.yaml` and run `dvc repro` again. Only the `train` stage runs. `prepare` and `features` stay cached. Put `C` back to `1.0` when you are done.

`mlruns` is not a DVC output. Changing `C` and changing it back can restore `data/metrics.json` from the DVC cache while the alias still points at the later version. If the form and `data/metrics.json` disagree, delete `mlruns` and run `uv run dvc repro --force`.

## Airflow clinic

The clinic is `airflow/README.md`. It uses the official Apache Airflow 3.4.0 compose file, unmodified. The UI is http://localhost:8080. Docker needs at least 4 GB of memory, 2 CPUs, and 10 GB of disk. One machine per team is enough. If Docker is not installed, skip the clinic. Day 5 has a plain Python runner that does not install Airflow.

## Library versions

Checked on Python 3.14.10 with `uv sync --frozen`.

| Package | Version |
| --- | --- |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| pyarrow | 22.0.0 |
| pyyaml | 6.0.3 |
| scikit-learn | 1.9.1 |
| mlflow | 3.18.0 |
| dvc | 3.67.1 |
| feast | 0.67.0 |
| streamlit | 1.67.0 |

## Troubleshooting

`ModuleNotFoundError` for `schema` or `registry`. Start Jupyter from the repo folder or from this folder. The first cell prints `cwd` and `module`.

`dvc repro` cannot find `uv`. Run it as `mise exec -- env PYTHONPATH=lib uv run dvc repro`, or use `make repro`.

`feast materialize` fails after you set `offline: True` on the feature view. Leave `offline` false. Feast 0.67's file writer rejects the table it just read. Historical reads still use the parquet file.

`No online row`. Run `make feast` after `dvc repro`.
