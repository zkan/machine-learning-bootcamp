# Day 2 workshop commands. Run make from this folder.
# mise.toml still points PYTHONPATH at Day 1's lib. env overrides it for these commands only.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- env PYTHONPATH=lib uv

.DEFAULT_GOAL := help

.PHONY: help setup sync kernel check notebook app test

help:
	@printf '%s\n' \
		'Day 2: Segments and recommendations' \
		'' \
		'  make setup    locked packages and the notebook kernel' \
		'  make sync     uv sync --frozen' \
		'  make kernel   register the fresh-and-good-day2 kernel' \
		'  make check    confirm the planted-leak table' \
		'  make notebook start Jupyter' \
		'  make app      start the day 2 form' \
		'  make test     teaching-team checks'

setup: sync kernel

sync:
	cd $(HERE) && $(UV) sync --frozen

kernel:
	cd $(HERE) && $(UV) run python -m ipykernel install --user --name fresh-and-good-day2 --display-name "Python (fresh-and-good-day2)"

check:
	cd $(HERE) && $(UV) run python -m leaks --check

notebook:
	cd $(HERE) && $(UV) run python -m jupyter notebook

app:
	cd $(HERE) && $(UV) run python -m streamlit run app.py

test:
	cd $(HERE) && $(UV) sync --frozen --group dev
	cd $(HERE) && $(UV) run --group dev python -m pytest tests -q --tb=line
