# Day 6: Capstone

Demo day. You pull the week into one form and one 8-minute story. The form scores a customer from the Day 5 alias, shows the drift table, and can look a customer up in Feast when Day 4 was materialized.

- [Plan](plan/one_page.md) and [30-day plan](plan/thirty_day.md)
- [Rubric cards](rubric/cards.md)
- [Presenter guide](business/presenter_guide.md)
- [Checkpoints](checkpoints/README.md)
- [Opening sentence](00_capstone.ipynb)
- [Demo form](app.py)
- [Data assumptions](DATA_ASSUMPTIONS.md)

Run the commands on this page from `07-capstone`. This folder uses Python 3.14.10 so it can import the Day 4 Feast code and the Day 5 pipeline code. `uv sync` reads `.python-version`. You do not change the repo `mise.toml`.

## Definition of done

You are done when a classmate can do all of these from the READMEs, without you typing for them.

- Day 4 `dvc repro` a second time prints that the pipeline is up to date.
- The form on this page loads `fresh-and-good-churn` at alias `champion`.
- The Day 5 runner, or the Airflow dag, registered a model that passed the 0.70 gate.
- The monitor page opens and you can say retrain or rollback if a column crosses 0.2 PSI.
- Feast lookup works, or you marked it as a stretch you are not demoing.
- `plan/one_page.md` is filled and signed.
- You have rehearsed the demo once at 8 minutes.

## Checkpoints

Three stops. The person who approves the scope is the TA or the instructor. TA availability is not settled. If no TA is in the room, the instructor signs. The tag commands are in [checkpoints](checkpoints/README.md). Create them on your branch. Do not push them to `main`.

1. Scope. The one-page plan, including what you will not build.
2. Pipeline. A passing model on the alias, and this form scoring `FG0010`.
3. Rehearsal. One full demo, including the monitor page.

## Demo day

8 minutes of demo, then 4 minutes of questions.

Suggested split for the 8 minutes:

1. Two minutes for the business story. Problem, cost, payoff, and one risk. Use [the presenter guide](business/presenter_guide.md).
2. Four minutes for the form. Score `FG0010`, then show the monitor page.
3. Two minutes for how you would rerun it. Name code, data, params, and the environment.

Peer votes. Each person has 3 votes. You cannot vote for your own team. Write the team name once per vote. The instructor tallies them in the room. The tally is a classroom signal, not a grade. The rubric cards are the grade.

## Create the environment

Day 5 has to have been run once so `data/features.parquet` and `mlruns` exist. Day 4 Feast is only required for the third page.

```sh
mise install
uv sync --frozen
uv run python -m ipykernel install --user --name fresh-and-good-day6 --display-name "Python (fresh-and-good-day6)"
```

## Run the form

```sh
uv run python -m streamlit run app.py --server.port 8506
```

Score `FG0010`. Expect a churn probability of `32.7%`, the same number as Day 5, because this form loads that alias. The health check is:

```sh
curl --fail http://127.0.0.1:8506/_stcore/health
```

Expect `ok`.

The Feast page for `FG0010` shows plan `monthly` and `2` orders when Day 4 was materialized.

## After the demo

Fill [plan/thirty_day.md](plan/thirty_day.md) before you leave.

## Library versions

Same pins as Day 4 and Day 5, on Python 3.14.10.

| Package | Version |
| --- | --- |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| pyarrow | 22.0.0 |
| scikit-learn | 1.9.1 |
| mlflow | 3.18.0 |
| feast | 0.67.0 |
| streamlit | 1.67.0 |
