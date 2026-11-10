# Day 2: Segments and recommendations

Day 2 of the Machine Learning Bootcamp. You stay with Fresh and Good, the fictional Bangkok online grocery from Day 1. Morning work finds two leaky columns and groups customers. Afternoon work suggests products, opens a second form, and plays a small bandit.

- 2.1 [Leaks, then two features](00_leaks.ipynb)
- 2.2 [Customer segments](01_clusters.ipynb)
- 2.3 [Product neighbors](02_recommend.ipynb)
- 2.4 [Four snack machines](03_bandit.ipynb)
- 2.5 [Segments and recommendations form](app.py)
- [Column notes](data/DATA_DICTIONARY.md)

Run the commands on this page from `02-segments-and-recommendations` (this folder). Tool pins live in the repo `mise.toml`. This folder has its own `pyproject.toml`. Day 1 is a different folder. Do not edit it.

## Timetable

Bangkok time. Breaks are 10:30–10:45 and 14:30–14:45. Lunch is 12:00–13:00.

| Time | Block |
| --- | --- |
| 09:00–10:00 | Leak hunt and two new features. Compare on validation. |
| 10:00–10:30 | Start k-means. Scale recency, frequency, and spend. |
| 10:45–12:00 | Choose k, profile the segments, write a persona and one action. |
| 13:00–14:30 | Baskets, popularity versus cosine similarity, precision@k. |
| 14:45–15:10 | Open the Day 2 form. Segments tab, then Recommendations tab. |
| 15:10–16:00 | Epsilon-greedy bandit, 4 machines, 20 pulls. |

## What you build

A short list of columns you will not use, two June features, a segment table with one action per segment, a product-neighbor table, and a form that reads those two tables.

## Objectives

By the end of this workshop, you can:

- Point at a column that uses July and leave it out of the model.
- Add two features that are known on 30 June and compare validation ROC-AUC.
- Scale three columns, run k-means, and pick k with silhouette.
- Describe a segment and name one action.
- Compare a popularity list with cosine similarity using precision@k on held-out baskets.
- Pull a four-arm bandit with epsilon-greedy and a seed.

## The data

`data/fresh_and_good_day2.csv` is the Day 1 customer file plus two planted July columns. `lib/leaks.py` reads `01-supervised-learning/data/fresh_and_good_customers.csv` and writes this copy. It does not change the Day 1 file.

The July labels are in the public file. There is no hidden test and no hidden basket file. Validation scores and precision@k are scores on held-out rows of these public tables.

Baskets are drawn in `lib/baskets.py` (800 baskets, seed 7). They are not real orders.

## Before class

Use the Day 1 install for mise 2026.11.3, uv 0.13.7, and Python 3.15.2. The Linux setup was tested. The macOS and Windows commands were not retested for Day 2.

## Create the environment

From this folder, after mise is on your `PATH`:

```sh
mise install
uv sync --frozen
PYTHONPATH=lib uv run python -m ipykernel install --user --name fresh-and-good-day2 --display-name "Python (fresh-and-good-day2)"
```

The repo `mise.toml` sets `PYTHONPATH` to `01-supervised-learning/lib`. Leave that line alone. Day 1 imports stay on Day 1. This Makefile sets `PYTHONPATH=lib` only for Day 2 commands. The notebooks also insert this folder's `lib` at the front of `sys.path`. Day 2 modules are `leaks`, `rfm`, `segments`, `baskets`, and `bandit`. Those names are not Day 1 modules.

`uv.lock` installs PyYAML 6.0.3 from source because that release has no cp315 wheel. If the build fails, check that your laptop has a C compiler.

Check the table:

```sh
PYTHONPATH=lib uv run python -m leaks --check
```

`make setup`, `make check`, `make notebook`, `make app`, and `make test` are the same steps. `make` uses `env PYTHONPATH=lib` so Day 2's modules win for that command.

## Open the form

The notebooks write `models/segments.csv` and `models/neighbors.csv`. That folder is gitignored. Then:

```sh
uv run python -m streamlit run app.py
```

The form does not import the Day 2 helpers, so this command can leave `PYTHONPATH` on Day 1.

The form has a Segments tab and a Recommendations tab. If a table is missing, the tab says which notebook to run.

## Bandit on paper

From 15:10, teams can play the Bandit Casino on paper: 4 machines, odds hidden by the instructor, 20 pulls per team. This repo does not include a physical kit. `03_bandit.ipynb` is the same game in NumPy.

## If you finish early

Business: sketch a campaign cost and one retained-customer value for each segment. Engineer: use the segment id when you pick the product list. Neither is required. There is no gridworld notebook.

## Checks for the teaching team

```sh
make test
```

Completed notebooks stay with the teaching team. They are not in this repo.
