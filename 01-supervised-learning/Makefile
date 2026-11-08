# Day 1 workshop commands. Run make from this folder (pyproject.toml lives here).
# mise exec selects the uv pin and PYTHONPATH from the repo mise.toml.
# See README.md.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- uv

.DEFAULT_GOAL := help

.PHONY: help setup mise-version tools sync kernel python check smoke notebook app app-8502 sync-dev test

help:
	@printf '%s\n' \
		'Day 1: Supervised learning' \
		'' \
		'  make setup         mise install, locked packages, and the notebook kernel' \
		'  make mise-version  print the mise version' \
		'  make tools         mise install' \
		'  make sync          uv sync --frozen' \
		'  make kernel        register the fresh-and-good-day1 kernel' \
		'  make python        print the project interpreter' \
		'  make check         confirm the synthetic table' \
		'  make smoke         print versions, rows, and columns' \
		'  make notebook      start Jupyter' \
		'  make app           start the scoring form' \
		'  make app-8502      start the form on port 8502' \
		'  make sync-dev      install the teaching-team packages' \
		'  make test          run the teaching-team checks'

setup: tools sync kernel

mise-version:
	cd $(HERE) && mise --version

tools:
	cd $(HERE) && mise install

sync:
	cd $(HERE) && $(UV) sync --frozen

kernel:
	cd $(HERE) && $(UV) run python -m ipykernel install --user --name fresh-and-good-day1 --display-name "Python (fresh-and-good-day1)"

python:
	cd $(HERE) && $(UV) run python -c "import sys; print(sys.version); print(sys.executable)"

check:
	cd $(HERE) && $(UV) run python -m generate --check

smoke:
	cd $(HERE) && $(UV) run python -m smoke

notebook:
	cd $(HERE) && $(UV) run python -m jupyter notebook

app:
	cd $(HERE) && $(UV) run python -m streamlit run app.py

app-8502:
	cd $(HERE) && $(UV) run python -m streamlit run app.py --server.port 8502

sync-dev:
	cd $(HERE) && $(UV) sync --frozen --group dev

test: sync-dev
	cd $(HERE) && $(UV) run --group dev python -m pytest tests -q --tb=line
