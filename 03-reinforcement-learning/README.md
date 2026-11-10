# Reinforcement learning

This is the 15:10–16:00 block on Day 2. It is not part of `02-unsupervised-learning`. You play four snack machines with an epsilon-greedy rule.

- 3.1 [Four snack machines](00_bandit.ipynb)
- 3.2 [A small grid](01_q_learning.ipynb), extra if the bandit is finished

Run the commands on this page from `03-reinforcement-learning` (this folder). Tool pins live in the repo `mise.toml`. This folder has its own `pyproject.toml`. It uses the same package versions as Day 2 and adds none. Do not edit Day 1.

## Steps

1. Create the environment below and open `00_bandit.ipynb`.
2. Set epsilon between 0 and 1. The notebook uses 20 pulls and seed 42.
3. Run the cell. Read the choices, the rewards, and the mean reward.
4. Run the next cell and check that the same seed repeats the choices.
5. If you have time, try epsilon 0 and epsilon 0.3 with the same seed. Twenty pulls is a small sample.

The 15:10–16:00 block is the bandit. The grid notebook is extra after that. Do not start it until the bandit cell has run. It does not change the 20 pulls.

## On paper

The Bandit Casino uses 4 machines, odds hidden by the instructor, and 20 pulls per team. This repo does not include a physical kit. The notebook is that game in NumPy. The odds live in `lib/bandit.py`. Leave them unread until you have run the cell.

## Expected output

Twenty choices, twenty rewards of 0 or 1, and one mean reward. Running the same epsilon, the same pull count, and seed 42 again prints the same choices.

If you reach the grid: an arrow in each open cell, the mean reward of the last episodes higher than the first, and a short note on gamma 0 versus gamma 0.9. Gamma 0.9 should reach the goal and stay off the pit. Gamma 0 does not plan that far.

## Create the environment

From this folder, after mise is on your `PATH`:

```sh
mise install
uv sync --frozen
PYTHONPATH=lib uv run python -m ipykernel install --user --name fresh-and-good-rl --display-name "Python (fresh-and-good-rl)"
```

The repo `mise.toml` sets `PYTHONPATH` to `01-supervised-learning/lib`. Leave that line alone. This Makefile sets `PYTHONPATH=lib` only for these commands. The notebook inserts this folder's `lib` at the front of `sys.path`.

`uv.lock` installs PyYAML 6.0.3 from source because that release has no cp315 wheel. If the build fails, check that your laptop has a C compiler.

`make setup`, `make notebook`, and `make test` are the same steps.

## Checks for the teaching team

```sh
make test
```

The completed notebook stays with the teaching team. It is not in this repo.
