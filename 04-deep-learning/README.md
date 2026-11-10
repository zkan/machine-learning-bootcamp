# Day 3: Deep learning

Day 3 of the Machine Learning Bootcamp. You stay with Fresh and Good, the fictional Bangkok online grocery from Day 1. In the morning, fit a small neural net on churn and compare it with the random forest. After lunch, classify support-ticket text, log every run in MLflow, and put the chosen model on a form. At 15:30, share the memo and recap the day.

- 3.1 [Neural net on churn](00_mlp_churn.ipynb)
- 3.2 [Training behaviour](01_training.ipynb)
- 3.3 [Tickets and the tuning cup](02_tickets.ipynb)
- 3.4 [Swap the model](03_swap.ipynb)
- 3.5 [Model switch](app.py)
- 3.6 [Why this model](why_this_model.md)
- [Column notes](data/DATA_DICTIONARY.md)

Run the commands on this page from `04-deep-learning` (this folder). Tool versions are pinned in the repo `mise.toml`. This folder has its own `pyproject.toml`. Do not edit Day 1 or Day 2.

## Timetable

Bangkok time. Breaks are 10:30–10:45 and 14:30–14:45. Lunch is 12:00–13:00.

| Time | Block |
| --- | --- |
| 09:00–10:30 | Neural net, then `MLPClassifier` on churn. Did it beat the random forest? |
| 10:45–12:00 | Learning rate, scaling, early stopping, learning curve. Train score versus validation score. |
| 13:00–14:30 | TF-IDF plus MLP on tickets. Network Tuning Cup. Every run is logged. |
| 14:45–15:30 | Model switch on the form. Register one model. Fill in the memo. |
| 15:30–16:00 | Share the memo. Recap the day. |

## What you build

A validation ROC-AUC for the forest and for one MLP, a short note on the train-versus-validation gap, four logged ticket runs and a winner, a form that switches models and shows the MLflow version, and a memo.

## Objectives

By the end of this workshop, you can:

- Say what a layer, a loss, and a learning rate are doing in `MLPClassifier`.
- Compare that network with the Day 1 random forest on validation ROC-AUC.
- Point at a learning rate or a missing scaler that hurts the validation score.
- Read a train score and a validation score and say which gap is wider.
- Log several text models to a local MLflow folder and pick the best validation accuracy.
- Switch the form between the forest and the MLP and name the registered version.
- Say, in one sentence, which model you would ship and what you gave up.

## In the room

These three games have no kit in this repo. Say them out loud. They are the picture for the notebooks, not a handout.

Humans as a neural net. Three people stand in a line. The first hears a number of orders. Each person passes 1 if the number they received is 3 or more, and 0 otherwise. The last person says churn or stay. Then change the middle person's rule and send the same number through again.

Blindfold hill. One person is blindfolded on a line of tape. They take a step. A partner says only "higher" or "lower". A huge step jumps over the bottom of the hill. A tiny step takes all morning. That step size is the learning rate.

Memorize versus understand. Read three training customers and the label, and ask the room to recite them. Then describe a new customer the room has not heard. Reciting the three is the train score. The new customer is the validation score.

## The data

Churn uses the Day 1 file, `01-supervised-learning/data/fresh_and_good_customers.csv`. This folder reads it and does not write it. The split is the Day 1 split: seed 42, 1440 train, 480 validation, 480 test. Leave the test rows unused.

`data/support_tickets.csv` is 480 synthetic tickets, seed 42, 120 in each category. See [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md). There is no hidden test. The cup uses a 360/120 stratified split, seed 42.

## Expected outputs

Starter `hidden = (32,)` in `00_mlp_churn.ipynb`, validation ROC-AUC:

| Model | Train | Validation | Notes |
| --- | --- | --- | --- |
| Random forest | 0.959 | 0.776 | 80 trees, `min_samples_leaf` 5, seed 42 |
| MLP `(32,)` | 0.849 | 0.782 | Adam, 200 iterations, the loss has not settled |

The MLP is ahead on the validation rows by a little. The forest fits the train rows harder.

`01_training.ipynb`, validation ROC-AUC unless noted:

| Setting | Train | Validation |
| --- | --- | --- |
| SGD, learning rate 0.001, 40 iterations | 0.680 | 0.656 |
| SGD, learning rate 0.01, 40 iterations | 0.790 | 0.772 |
| SGD, learning rate 1.0, 40 iterations | 0.871 | 0.748 |
| SGD 0.01, numeric columns scaled | 0.790 | 0.772 |
| SGD 0.01, numeric columns not scaled | 0.540 | 0.542 |
| Adam `(64, 64)`, no early stopping, 80 iterations | 0.913 | 0.756 |
| Adam `(64, 64)`, early stopping | 0.812 | 0.787 |

The gap on the "no early stopping" row, using these three-decimal scores, is 0.913 − 0.756 = 0.157. Early stopping stops at 27 iterations. The unscaled run stops at 13 iterations.

Learning curve, train rows only, 3-fold ROC-AUC:

| Train rows | Train | CV |
| --- | --- | --- |
| 240 | 0.653 | 0.618 |
| 480 | 0.693 | 0.669 |
| 720 | 0.729 | 0.704 |
| 960 | 0.750 | 0.728 |

Network Tuning Cup, 40 iterations, validation accuracy. A majority guess on four balanced classes is 0.25.

| Hidden layers | Learning rate | Validation accuracy |
| --- | --- | --- |
| `(16,)` | 0.001 | 0.508 |
| `(32,)` | 0.001 | 0.700 |
| `(32, 16)` | 0.001 | 0.700 |
| `(32,)` | 0.01 | 0.758 |

The winner is `(32,)` at learning rate 0.01. On the latest Linux check of these notebooks the forest took 0.13 seconds and the MLP took 0.33 seconds. The four ticket fits took 0.033, 0.037, 0.038, and 0.031 seconds, in the order of the table. Your laptop will differ. The score is the number to match.

The stretch cell writes this confusion matrix for that winning run. Rows are the true category. Columns are the prediction. Each category has 30 validation rows.

|  | delivery | payment | product | account |
| --- | --- | --- | --- | --- |
| delivery | 23 | 0 | 3 | 4 |
| payment | 4 | 21 | 2 | 3 |
| product | 3 | 2 | 24 | 1 |
| account | 0 | 2 | 5 | 23 |

## Before class

Use the Day 1 install for mise 2026.11.3, uv 0.13.7, and Python 3.15.2. The Linux setup was tested for this day. The macOS and Windows commands for the notebook environment were not retested.

## Create the environment

From this folder, after mise is on your `PATH`:

```sh
mise install
uv sync --frozen
PYTHONPATH=lib uv run python -m ipykernel install --user --name fresh-and-good-day3 --display-name "Python (fresh-and-good-day3)"
```

The repo `mise.toml` sets `PYTHONPATH` to `01-supervised-learning/lib`. Leave that line alone so Day 1 imports still work. For Day 3 commands, set `PYTHONPATH=lib` as the Makefile does. The notebooks put this folder's `lib` first on `sys.path`. Day 3 modules are `customers`, `training`, `tickets`, and `tracking`.

MLflow 3.18 keeps the filesystem tracking store only if you opt in. The notebooks and the Makefile set `MLFLOW_ALLOW_FILE_STORE=true`. Runs and the model registry are files under `tracking/`. Nothing is uploaded. `tracking/` is gitignored.

`uv.lock` installs PyYAML 6.0.3 from source because that release has no cp315 wheel. If the build fails, check that your laptop has a C compiler.

Check the ticket table:

```sh
PYTHONPATH=lib uv run python -m tickets --check
```

`make setup`, `make check`, `make notebook`, `make app`, and `make test` run the same steps.

## Open the notebooks

From this folder:

```sh
uv run python -m jupyter notebook
```

If the kernel menu lists more than one Python, choose Python (fresh-and-good-day3). Run the notebooks in order. `None` and an empty `debrief` are the blanks. The code around them still runs.

1. `00_mlp_churn.ipynb`
2. `01_training.ipynb`
3. `02_tickets.ipynb`
4. `03_swap.ipynb`

## MLflow

After the cup notebook has logged runs, the leaderboard is also a local page. From this folder:

```sh
MLFLOW_ALLOW_FILE_STORE=true uv run python -m mlflow ui --backend-store-uri "file://${PWD}/tracking" --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. The experiment is `network-tuning-cup`. The registered model is `fresh-and-good-churn`. This is a local UI on the same files. It is not a shared server.

The file store is in maintenance in MLflow 3.18. The project sets `MLFLOW_ALLOW_FILE_STORE` because this class keeps a folder of files and does not run a database. See the [backend store](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/) page. The APIs used here were checked on MLflow 3.18.0: `mlflow.set_tracking_uri`, `mlflow.sklearn.log_model(..., name="model", registered_model_name=..., serialization_format="pickle")`, `mlflow.log_metric`, `mlflow.log_artifact`, and `mlflow.search_runs`. The default format in this release is skops, which refuses the Adam optimizer object inside `MLPClassifier` unless that private type is listed. The notebook logs the pipeline with pickle instead.

## Open the form

After `03_swap.ipynb` has written `models/` (that folder is gitignored):

```sh
uv run python -m streamlit run app.py
```

The radio says Classic random forest or MLP. The line under it names the model and, for the one you registered, the MLflow version. Score a customer. The probability is a classroom score on synthetic rows.

Load only a file you fit in class. A joblib file can run code when it is loaded.

## Fill in the memo

`why_this_model.md` has three blanks: accuracy, explainability, and cost. The forest can be talked through tree by tree. The MLP is a stack of weights. On this table both fits take well under a second, so the cost conversation is the voucher, not the GPU. Write one sentence on which model you would ship.

## Wrap-up

At 15:30, each pair reads the memo. Say which model you would ship, and one trade-off you accepted. Then recap the day: the MLP and the forest on the validation rows, the train-versus-validation gap, and the ticket run you kept in MLflow.

## For the teaching team

Completed notebooks and a filled memo stay with the teaching team. They are not in this repo.

```sh
make test
```

## Library versions

| Package | Version | Where |
| --- | --- | --- |
| mise | 2026.11.3 | repo `mise.toml` |
| uv | 0.13.7 | repo `mise.toml` |
| Python | 3.15.2 | this folder |
| numpy | 2.5.4 | notebooks |
| pandas | 3.1.1 | notebooks |
| scikit-learn | 1.9.1 | [MLPClassifier](https://scikit-learn.org/1.9/modules/generated/sklearn.neural_network.MLPClassifier.html), [TfidfVectorizer](https://scikit-learn.org/1.9/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [learning_curve](https://scikit-learn.org/1.9/modules/generated/sklearn.model_selection.learning_curve.html) |
| mlflow | 3.18.0 | local file store, checked on Python 3.15.2 |
| joblib | 1.6.0 | model files for the form |
| streamlit | 1.67.0 | form |
| jupyter | 1.1.1 | notebook server |
| ipykernel | 7.4.0 | kernel `fresh-and-good-day3` |

The automated checks also used pytest 9.2.0, nbclient 0.11.0, and nbformat 5.10.4. MLflow pulls matplotlib 3.11.2 and pyarrow 26.0.1. The notebooks do not call them.
