# mise.toml still points PYTHONPATH at Day 1's lib. env overrides it for these commands only.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- env PYTHONPATH=lib uv

.DEFAULT_GOAL := help

.PHONY: help setup sync kernel repro feast checkpoint app test

help:
	@printf '%s\n' \
		'Day 4: MLOps reproducibility' \
		'' \
		'  make setup      locked packages and the notebook kernel' \
		'  make sync       uv sync --frozen' \
		'  make kernel     register the fresh-and-good-day4 kernel' \
		'  make repro      dvc repro' \
		'  make feast      feast apply and materialize' \
		'  make checkpoint repro, then feast' \
		'  make app        start the scoring form' \
		'  make test       teaching-team checks'

setup: sync kernel

sync:
	cd $(HERE) && $(UV) sync --frozen

kernel:
	cd $(HERE) && $(UV) run python -m ipykernel install --user --name fresh-and-good-day4 --display-name "Python (fresh-and-good-day4)"

repro:
	cd $(HERE) && $(UV) run --no-sync dvc repro

feast:
	cd $(HERE)/feature_repo && $(UV) run --no-sync feast apply
	cd $(HERE)/feature_repo && $(UV) run --no-sync feast materialize 2026-06-01T00:00:00 2026-07-02T00:00:00

checkpoint: repro feast

app:
	cd $(HERE) && $(UV) run --no-sync python -m streamlit run app.py --server.headless true

test:
	cd $(HERE) && $(UV) sync --frozen --group dev
	cd $(HERE) && $(UV) run --group dev python -m pytest tests -q --tb=line
