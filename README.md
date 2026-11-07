# Machine Learning Bootcamp

Day 1 builds a local churn check for KlongCart, a fictional Bangkok online grocery. You start from a business question, look at a synthetic customer table, fit a scikit-learn model, choose a voucher threshold, and type one customer into a Streamlit form.

The class runs 09:00 to 16:00 Bangkok time. Work in pairs inside five teams. The session is taught in Thai. These notes are in English.

The score is a classroom demo. It is not a production retention system. The rows were drawn by `day1/generate.py`. They are not real shoppers.

## What you will have at 16:00

A form that shows a churn probability and the word yes or no. Yes means send a small voucher. No means leave the customer alone. The probability comes from a model you fit on this laptop. The yes or no line uses a threshold your team chose from a cost card.

## The question

Which customers will place no paid order from 1 July 2026 through 30 July 2026?

Features stop on 30 June 2026. That window is 2 April 2026 through 30 June 2026, 90 days. `churned` is the later window. It is the label, not a feature. Read `data/DATA_DICTIONARY.md` before you model. The table has no customer id, email, or phone, and no column measured in July.

## Objectives

You can do these by the end of the day.

- Say what the train rows, the validation rows, and the test rows are for.
- Compute a churn rate and name three facts from the table.
- Fit logistic regression and a random forest inside a preprocessing pipeline.
- Compare those models with a majority-class baseline on validation ROC-AUC.
- Choose a threshold on the validation rows from the team's cost card.
- Score the test rows once, then save the bundle.
- Show the probability and the yes or no decision in the form.

## Timetable

| Bangkok time | What you do | Notebook |
| --- | --- | --- |
| 09:00 to 09:10 | The question, supervised learning, and the loop | `notebooks/00_start.ipynb` |
| 09:10 to 09:50 | Environment, kernel, smoke check | this page |
| 09:50 to 10:30 | Dicts, functions, f-strings, three exercises | `notebooks/00_start.ipynb` |
| 10:30 to 10:45 | Break | |
| 10:45 to 11:25 | Read, filter, group, plot, churn rate, three observations | `notebooks/01_eda.ipynb` |
| 11:25 to 12:00 | Majority baseline and the three-way split. Optional paper rules | `notebooks/02_split.ipynb` |
| 12:00 to 13:00 | Lunch | |
| 13:00 to 13:40 | Logistic regression, random forest, pipeline, cross-validation on train | `notebooks/03_train_cv.ipynb` |
| 13:40 to 14:30 | Small random search, compare with the baseline, look at the train gap | `notebooks/04_tune.ipynb` |
| 14:30 to 14:45 | Break | |
| 14:45 to 15:10 | Confusion matrix, precision, recall, ROC-AUC, threshold, one test check, save | `notebooks/05_evaluate.ipynb` |
| 15:10 to 15:45 | Form, load the bundle, fill in the yes or no line | `app.py` |
| 15:45 to 16:00 | Demo. Say the data is synthetic | |

The 09:00 block can stay in discussion while `00_start.ipynb` is open. The cost card belongs to the business half of the pair. The default is 500 THB for a missed churner and 100 THB for a wasted voucher. Change those two names in notebook 05 if the team disagrees, and do it before the test cell.

## Before class

Install Python 3.12. This package was run with Python 3.12.3 on Linux. Windows and macOS steps are listed below and were not run here.

Create the environment from the repo folder. The morning block is the backup for people who could not finish this the night before.

## Create the environment

Linux and macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name klongcart-day1 --display-name "Python (klongcart-day1)"
```

Windows, from the repo folder, in PowerShell or Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name klongcart-day1 --display-name "Python (klongcart-day1)"
```

The learner install is `requirements.txt` only. It pins numpy, pandas, matplotlib, scikit-learn, joblib, streamlit, jupyter, and ipykernel. joblib is how the afternoon saves the model. scikit-learn uses it. Do not add MLflow, DVC, or Feast today.

`requirements-dev.txt` is for the automated checks. You do not need it to follow the timetable.

On this Linux machine the kernel command printed:

```text
Installed kernelspec klongcart-day1 in /home/ubuntu/.local/share/jupyter/kernels/klongcart-day1
```

The folder under your home directory will differ. The important line is `Installed kernelspec klongcart-day1`.

## Check the table

From the repo folder, with the environment active:

```bash
python -m day1.generate --check
```

Linux result from this prep:

```text
match /workspace/data/klongcart_churn.csv rows=2400
```

Your path will be the repo you cloned. The line still ends in `rows=2400`. A mismatch means the CSV was edited. Do not edit it. The generator and the file are the same draw, seed 42, 2,400 rows.

```bash
python -m day1.smoke
```

Linux result:

```text
numpy 2.5.4
pandas 3.1.1
matplotlib 3.11.2
scikit-learn 1.9.1
streamlit 1.67.0
rows 2400
columns tenure_days, orders_90d, spend_thb_90d, avg_basket_thb_90d, days_since_last_order, promo_order_share_90d, app_sessions_90d, support_tickets_90d, district, plan, primary_aisle, churned
smoke ok
```

## Open the notebooks

From the repo folder:

```bash
python -m jupyter notebook
```

Open `notebooks/00_start.ipynb`. If the kernel menu lists more than one Python, choose Python (klongcart-day1).

The first code cell prints `cwd` and `repo`. Start Jupyter from the repo folder or from `notebooks`. Both work. Other folders fail that cell on purpose.

Run the notebooks in order.

1. `notebooks/00_start.ipynb`
2. `notebooks/01_eda.ipynb`
3. `notebooks/02_split.ipynb`
4. `notebooks/03_train_cv.ipynb`
5. `notebooks/04_tune.ipynb`
6. `notebooks/05_evaluate.ipynb`

Each notebook loads the CSV itself. You can restart the kernel between files.

These prints come from the cells that are already filled in. They are how you know the table and the split loaded.

```text
(2400, 12)
train 1440 validation 480 test 480
overlap train and test 0
churned is a label, not a feature: True
```

## Exercises that are supposed to be blank

The notebooks in git do not run from top to bottom until you fill the blanks. A blank left as `None`, or a function that still raises `NotImplementedError`, stops that notebook. That is the assignment, not a broken file.

| File | You fill in |
| --- | --- |
| `00_start.ipynb` | `soda_shopper`, `format_customer`, `quiet_customer` |
| `01_eda.ipynb` | `no_plan`, `churn_by_plan`, `churn_rate`, and three sentences |
| `02_split.ipynb` | `majority_accuracy`, `threshold_set` |
| `03_train_cv.ipynb` | the forest cross-validation mean and the logistic validation ROC-AUC |
| `04_tune.ipynb` | `beats_baseline`, `gap`, and one sentence |
| `05_evaluate.ipynb` | test precision, test recall, and test ROC-AUC |
| `app.py` | `churn_decision` |

Finished notebooks, the filled `churn_decision`, and a saved model are not in this repo. Kan keeps those in a private archive for the teaching team. If a pair is still in an earlier notebook at 15:10, the instructor can hand them `models/churn_bundle.joblib` from that archive. Put the file in `models/` on the student laptop and start the form. Do not commit that file. Do not download a model from the internet. A joblib file can run code when it is loaded. Load only a file you fit in class, or a file the instructor hands you in the room.

`checkpoints/README.md` is the short version of the same resume rules.

## Run the form

After notebook 05 saves the bundle:

```bash
python -m streamlit run app.py
```

Before that file exists, the page stops with an error. The error names the missing path, tells you to finish notebook 05, and warns you about joblib. Refresh after the file is there.

The form asks for the same columns the model was fit on. Spend with zero orders is rejected. The probability is shown by the starter. The line `Offer a voucher` appears after you replace the `NotImplementedError` in `churn_decision`. Return `yes` when the probability is at least the threshold. Return `no` otherwise.

The optional expander under the form is an engineer stretch for one more input rule. Notebook 05 has an optional stretch for gradient boosting and feature importance. Skip both during the timed demo.

## Compare teams without a hidden test

Write each team's validation ROC-AUC on the board. That set is for choosing the model and the threshold.

Run the test cell in notebook 05 once. Do not change the model after you see it.

The public CSV already contains `churned`, so this repo cannot hide a test label. There is no leaderboard of secret answers. The instructor archive has a local scorer with a cap of two bundles. The cap is a file the instructor can delete. It is a turn-in limit, not a sealed exam.

## If you are behind

Use the first cut that fits the clock.

- 10:25 and still on Python: skip `quiet_customer`.
- 11:20 and still on the table: skip the bar chart. Keep the plan rates and the three sentences.
- 11:55: skip the paper rules in notebook 02.
- 13:35: skip the forest exercise. Keep the logistic print and move to notebook 04.
- 14:20 and the search is still running: stop it and call `tune_forest` with `n_iter=4`.
- 15:05: save the bundle and open the form. Say the test numbers out loud only if the cell already finished.
- 15:40 and the form has no file: take the instructor checkpoint. Do not refit during the demo.

The random search in notebook 04 and notebook 05 uses 8 draws and 3 folds, with at most 120 trees. On the Linux machine used for this prep, notebook 04 finished in 4.01 seconds, including kernel startup.

## Troubleshooting

`ModuleNotFoundError: day1`. Start Jupyter from the repo folder or from `notebooks`. The first cell prints both paths when it works.

`python -m day1.generate --check` prints `mismatch`. Leave the CSV alone. Run the command from the repo folder.

The kernel list has no `klongcart-day1`. Activate `.venv` and run the `ipykernel install` command again.

The form says there is no model file. Finish notebook 05, or ask for the instructor file, then refresh.

The form says `Finish churn_decision`. That is the 15:10 exercise. Fill in `churn_decision` in `app.py` and save the file. Streamlit reruns the script.

`streamlit` says the port is in use. Run `python -m streamlit run app.py --server.port 8502`.

## Checks for the teaching team

From the repo folder, after `pip install -r requirements-dev.txt`:

```bash
python -m pytest tests -q --tb=line
```

Linux result during this prep:

```text
23 passed in 2.20s
```

Those tests check the table, the split, the pipelines, the bundle round trip, the starter form, and that the public notebooks still contain the blanks. They do not contain the exercise answers.

## Library versions

Pinned in `requirements.txt` and imported successfully on Linux with Python 3.12.3.

| Package | Version | Documentation used for this day |
| --- | --- | --- |
| numpy | 2.5.4 | generator |
| pandas | 3.1.1 | [read_csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) |
| matplotlib | 3.11.2 | bar chart in notebook 01 |
| scikit-learn | 1.9.1 | [Pipelines](https://scikit-learn.org/stable/modules/compose.html), [RandomizedSearchCV](https://scikit-learn.org/1.9/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), [tuning and held-out evaluation](https://scikit-learn.org/stable/modules/grid_search.html) |
| joblib | 1.6.0 | save and load the bundle |
| streamlit | 1.67.0 | [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) |
| jupyter | 1.1.1 | notebook server |
| ipykernel | 7.4.0 | kernel `klongcart-day1` |

The automated checks also used pytest 9.2.0, nbclient 0.11.0, and nbformat 5.10.4.
