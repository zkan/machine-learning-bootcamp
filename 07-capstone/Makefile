# mise.toml still points PYTHONPATH at Day 1's lib. The app inserts Day 4 and Day 5 on sys.path itself.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- uv

.DEFAULT_GOAL := help

.PHONY: help setup sync kernel app test

help:
	@printf '%s\n' \
		'Day 6: Capstone' \
		'' \
		'  make setup  locked packages and the notebook kernel' \
		'  make sync   uv sync --frozen' \
		'  make kernel register the fresh-and-good-day6 kernel' \
		'  make app    start the demo form' \
		'  make test   teaching-team checks'

setup: sync kernel

sync:
	cd $(HERE) && $(UV) sync --frozen

kernel:
	cd $(HERE) && $(UV) run python -m ipykernel install --user --name fresh-and-good-day6 --display-name "Python (fresh-and-good-day6)"

app:
	cd $(HERE) && $(UV) run --no-sync python -m streamlit run app.py --server.headless true --server.port 8506

test:
	cd $(HERE) && $(UV) sync --frozen --group dev
	cd $(HERE) && $(UV) run --group dev python -m pytest tests -q --tb=line
