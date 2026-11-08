# Day 1: Supervised learning

Day 1 of the Machine Learning Bootcamp. You build two local checks for Fresh and Good, a fictional Bangkok online grocery. Classification asks who places no paid order next month. Regression asks how many baht they spend in that same month.

- 1.1 [Start and Python](00_start.ipynb)
- 1.2 [Look at the customers](01_eda.ipynb)
- 1.3 [Split the rows](02_split.ipynb)
- 1.4 [Fit on the train rows](03_train_cv.ipynb)
- 1.5 [Compare a small search](04_tune.ipynb)
- 1.6 [Choose a threshold and check the test rows once](05_evaluate.ipynb)
- 1.7 [Estimate next-month spend](06_regression.ipynb)
- 1.8 [Score one customer](app.py)
- [Column notes](data/DATA_DICTIONARY.md)

The class runs 09:00 to 16:00 Bangkok time. Work in pairs inside five teams. The session is taught in Thai. These notes are in English.

The score is a classroom demo. It is not a production retention system. The rows were drawn by `lib/generate.py`. They are not real shoppers.

Run the commands on this page from `01-supervised-learning` (this folder), unless a step says otherwise. Tool pins live in the repo `mise.toml`.

## What you build

Build a form that shows churn probability, a yes or no voucher decision, and estimated July spend in THB. Fit both models locally. Choose the voucher threshold from your team's cost card. Yes means send a small voucher. No means leave the customer alone.

## The questions

Which customers will place no paid order from 1 July 2026 through 30 July 2026? That is classification. `churned` is 0 or 1.

How many baht will each customer pay in that same month? That is regression. `next_month_spend_thb` is an amount. It is 0 when `churned` is 1.

Features stop on 30 June 2026. That window is 2 April 2026 through 30 June 2026, 90 days. Both July columns are labels, not features. Read `data/DATA_DICTIONARY.md` before you model. The table has no customer id, email, or phone.

## Objectives

By 16:00, you can:

- Say what the train rows, the validation rows, and the test rows are for.
- Compute a churn rate and name three facts from the table.
- Fit logistic regression and a random forest inside a preprocessing pipeline.
- Compare those models with a majority-class baseline on validation ROC-AUC.
- Choose a threshold on the validation rows from the team's cost card.
- Score the churn test rows once, then save the churn bundle.
- Compare a mean spend baseline with ridge regression using MAE and RMSE in THB, then save the spend bundle.
- Show the churn probability, the yes or no decision, and the spend estimate in the form.

## Timetable

| Bangkok time | What you do | Notebook |
| --- | --- | --- |
| 09:00 to 09:10 | The question, supervised learning, and the loop | `00_start.ipynb` |
| 09:10 to 09:50 | Environment, kernel, smoke check | this page |
| 09:50 to 10:30 | Dicts, functions, f-strings, three exercises | `00_start.ipynb` |
| 10:30 to 10:45 | Break | |
| 10:45 to 11:25 | Read, filter, group, plot, churn rate, three observations | `01_eda.ipynb` |
| 11:25 to 12:00 | Majority baseline and the three-way split. Optional paper rules | `02_split.ipynb` |
| 12:00 to 13:00 | Lunch | |
| 13:00 to 13:40 | Logistic regression, random forest, pipeline, cross-validation on train | `03_train_cv.ipynb` |
| 13:40 to 14:30 | Small random search, compare with the baseline, look at the train gap | `04_tune.ipynb` |
| 14:30 to 14:45 | Break | |
| 14:45 to 15:05 | Confusion matrix, precision, recall, ROC-AUC, threshold, one churn test check, save | `05_evaluate.ipynb` |
| 15:05 to 15:25 | Mean spend baseline, ridge, validation MAE, one test MAE and RMSE | `06_regression.ipynb` |
| 15:25 to 15:45 | Form, both bundles, yes or no line, spend in THB | `app.py` |
| 15:45 to 16:00 | Demo. Say the data is synthetic | |

Classification is the main task. The 15:05 block compares spend models on the same rows. A ridge search larger than `n_iter=4` is optional and outside the timetable.

The business learner in each pair sets the cost card. Defaults are 500 THB for a missed churner and 100 THB for a wasted voucher. If your team changes these costs, edit them in notebook 05 before running the test cell.

## Before class

Install mise 2026.11.3, then use it to install uv 0.13.7 and Python 3.15.2. Do not mix in Python from python.org, Homebrew, or `py -3`. The Linux setup was tested. The macOS and Windows commands follow the mise documentation but were not tested for this course.

## Install mise

Use `MISE_VERSION` to install the course version.

Linux and macOS, from https://mise.jdx.dev/installing-mise.html:

```sh
curl -fsSL https://mise.run | MISE_VERSION=v2026.11.3 sh
```

For Zsh, use the activation command from https://mise.jdx.dev/cli/activate.html:

```sh
eval "$(~/.local/bin/mise activate zsh)"
```

For Bash:

```sh
eval "$(~/.local/bin/mise activate bash)"
```

For Windows PowerShell, use https://mise.run/powershell:

```powershell
$env:MISE_VERSION = "2026.11.3"
irm https://mise.run/powershell | iex
```

Open a new PowerShell window so `Path` includes mise, then activate it with the documented command:

```powershell
(&mise activate pwsh) | Out-String | Invoke-Expression
```

`mise.toml` requires at least mise 2026.11.3. Use the installer above for the exact course version.

Check it:

```sh
mise --version
```

Check that the version is 2026.11.3.

## Create the environment

From this folder (`01-supervised-learning`), after mise is on your `PATH`:

```sh
mise install
uv sync --frozen
uv run python -m ipykernel install --user --name fresh-and-good-day1 --display-name "Python (fresh-and-good-day1)"
```

These commands install Python 3.15.2 and uv 0.13.7 from the repo `mise.toml`, then the packages in this folder's `uv.lock`, and the notebook kernel. The `dev` group is not needed for class.

Run project commands with `uv run` to use the project environment.

Check the interpreter:

```sh
uv run python -c "import sys; print(sys.version); print(sys.executable)"
```

Check for Python 3.15.2 and an executable in this folder's `.venv`, not a system Python. On Windows, the path ends in `.venv\Scripts\python.exe`.

Run `mise install` once from the repo root if the tool pins are not installed yet. The kernel install prints `Installed kernelspec fresh-and-good-day1`. The location depends on your home directory. Use the installed packages for class. Do not add MLflow, DVC, or Feast today.

## Check the table

From this folder, with the environment active:

```sh
uv run python -m generate --check
```

Expect a `match` line ending in `rows=2400`. Do not edit the CSV. It contains 2,400 synthetic rows drawn with seed 42.

```sh
uv run python -m smoke
```

Expect the package versions, `rows 2400`, the column names, and `smoke ok`. See [Library versions](#library-versions) for the expected versions.

## Open the notebooks

From this folder:

```sh
uv run python -m jupyter notebook
```

Open `00_start.ipynb`. If the kernel menu lists more than one Python, choose Python (fresh-and-good-day1).

The first code cell prints `cwd` and `module`. Start Jupyter from the repo folder or from `01-supervised-learning`. Other folders fail that cell on purpose.

Run the notebooks in order.

1. `00_start.ipynb`
2. `01_eda.ipynb`
3. `02_split.ipynb`
4. `03_train_cv.ipynb`
5. `04_tune.ipynb`
6. `05_evaluate.ipynb`
7. `06_regression.ipynb`

Each notebook loads the CSV itself. You can restart the kernel between files.

The completed setup cells confirm the table and split:

```text
(2400, 13)
train 1440 validation 480 test 480
overlap train and test 0
churned is a label, not a feature: True
```

## Complete the exercises

Fill each exercise before running the notebook from top to bottom. `None` and `NotImplementedError` mark intentional blanks, not broken files.

| File | You fill in |
| --- | --- |
| `00_start.ipynb` | `soda_shopper`, `format_customer`, `quiet_customer` |
| `01_eda.ipynb` | `no_plan`, `churn_by_plan`, `churn_rate`, and three sentences |
| `02_split.ipynb` | `majority_accuracy`, `threshold_set` |
| `03_train_cv.ipynb` | the forest cross-validation mean and the logistic validation ROC-AUC |
| `04_tune.ipynb` | `beats_baseline`, `gap`, and one sentence |
| `05_evaluate.ipynb` | test precision, test recall, and test ROC-AUC |
| `app.py` | `churn_decision` |

If you are still in an earlier notebook at 15:25, ask the instructor for both checkpoint bundles. Follow [Resume points](checkpoints/README.md) to use them.

A joblib file can run code when loaded. Load only a model you fit in class or receive from the instructor in the room. Do not download a model from the internet or commit a saved model.

## Run the form

At 15:25, after notebook 05 and notebook 06 have saved their bundles, from this folder:

```sh
uv run python -m streamlit run app.py
```

If a bundle is missing, the page shows its path. Finish notebook 05 and notebook 06, or use both instructor checkpoint bundles, then refresh.

Enter the model's feature values. The form rejects spend with zero orders. Complete `churn_decision` in `app.py` to show `Offer a voucher` alongside the probability. Return `yes` when the probability is at least the threshold, and `no` otherwise.

The optional expander under the form is an engineer stretch for one more input rule. Notebook 05 has an optional stretch for gradient boosting and feature importance. Skip both during the timed demo.

## Compare teams without a hidden test

Write each team's validation ROC-AUC on the board. That set is for choosing the model and the threshold.

Run the test cell in notebook 05 once. Do not change the model after you see it.

The public CSV contains `churned`. Test labels are not hidden, so this is not a sealed exam.

## If you are behind

Use the first cut that fits the clock.

- 10:25 and still on Python: skip `quiet_customer`.
- 11:20 and still on the table: skip the bar chart. Keep the plan rates and the three sentences.
- 11:55: skip the paper rules in notebook 02.
- 13:35: skip the forest exercise. Keep the logistic print and move to notebook 04.
- 14:20 and the search is still running: stop it and call `tune_forest` with `n_iter=4`.
- 15:05: save the churn bundle if notebook 05 already finished, then start notebook 06. Do not open the form yet.
- 15:25 and a bundle is missing: take both instructor checkpoint bundles and start the form. Do not refit during the demo.

The random search in notebook 04 and notebook 05 uses 8 draws and 3 folds, with at most 120 trees.

## Troubleshooting

`ModuleNotFoundError` for `schema` or `pipelines`. Start Jupyter from the repo folder or from `01-supervised-learning`. The first cell prints `cwd` and `module` when it works. Run `uv` through mise so `PYTHONPATH` includes `lib`.

`uv run python -m generate --check` prints `mismatch`. Leave the CSV alone. Run the command from this folder.

The kernel list has no `fresh-and-good-day1`. From this folder, run the `uv run python -m ipykernel install` command again.

The form says there is no model file. Finish notebook 05 and notebook 06, or ask for both instructor files, then refresh.

The form says `Finish churn_decision`. That is the 15:25 exercise. Fill in `churn_decision` in `app.py` and save the file. Streamlit reruns the script.

`streamlit` says the port is in use. Run `uv run python -m streamlit run app.py --server.port 8502`.

## Checks for the teaching team

The `dev` group is pytest, nbclient, and nbformat. From this folder:

```sh
uv sync --frozen --group dev
uv run --group dev python -m pytest tests -q --tb=line
```

These optional checks cover the table, split, pipelines, saved bundles, starter form, and exercise blanks. They do not contain exercise answers.

## Library versions

The versions below were resolved by `uv lock` on CPython 3.15.2. Use `uv.lock`. Do not add a second requirements file.

| Package | Version | Documentation used for this day |
| --- | --- | --- |
| mise | 2026.11.3 | [Installing mise](https://mise.jdx.dev/installing-mise.html), [activate](https://mise.jdx.dev/cli/activate.html), [uv projects](https://mise.jdx.dev/lang/python.html) |
| uv | 0.13.7 | [Python versions](https://docs.astral.sh/uv/concepts/python-versions/), [release 0.13.7](https://github.com/astral-sh/uv/releases/tag/0.13.7) |
| Python | 3.15.2 | [Python 3.15.2](https://www.python.org/downloads/release/python-3152/) |
| numpy | 2.5.4 | generator |
| pandas | 3.1.1 | [read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| matplotlib | 3.11.2 | bar chart in notebook 01 |
| scikit-learn | 1.9.1 | [Pipelines](https://scikit-learn.org/stable/modules/compose.html), [RandomizedSearchCV](https://scikit-learn.org/1.9/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), [MAE](https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.mean_absolute_error.html), [RMSE](https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.root_mean_squared_error.html) |
| joblib | 1.6.0 | save and load the bundle |
| streamlit | 1.67.0 | [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) |
| jupyter | 1.1.1 | notebook server |
| ipykernel | 7.4.0 | kernel `fresh-and-good-day1` |

The automated checks also used pytest 9.2.0, nbclient 0.11.0, and nbformat 5.10.4.

`uv.lock` installs PyYAML 6.0.3 from source because that release has no cp315 wheel. If the build fails, check that your laptop has a C compiler.
