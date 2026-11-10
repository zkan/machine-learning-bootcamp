# mise.toml still points PYTHONPATH at Day 1's lib. env overrides it for these commands only.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- env PYTHONPATH=lib:airflow/dags uv

.DEFAULT_GOAL := help

.PHONY: help setup sync kernel run vendor app test

help:
	@printf '%s\n' \
		'Day 5: MLOps pipelines' \
		'' \
		'  make setup   locked packages and the notebook kernel' \
		'  make sync    uv sync --frozen' \
		'  make kernel  register the fresh-and-good-day5 kernel' \
		'  make run     plain Python pipeline, no Airflow' \
		'  make vendor  copy the Day 1 CSV into the Airflow dags folder' \
		'  make app     start the form and the monitor page' \
		'  make test    teaching-team checks'

setup: sync kernel

sync:
	cd $(HERE) && $(UV) sync --frozen

kernel:
	cd $(HERE) && $(UV) run python -m ipykernel install --user --name fresh-and-good-day5 --display-name "Python (fresh-and-good-day5)"

run:
	cd $(HERE) && $(UV) run --no-sync python -m runner

vendor:
	mkdir -p $(HERE)/airflow/dags/vendor
	cp $(HERE)/../01-supervised-learning/data/fresh_and_good_customers.csv $(HERE)/airflow/dags/vendor/fresh_and_good_customers.csv

app:
	cd $(HERE) && $(UV) run --no-sync python -m streamlit run app.py --server.headless true --server.port 8505

test:
	cd $(HERE) && $(UV) sync --frozen --group dev
	cd $(HERE) && $(UV) run --group dev python -m pytest tests -q --tb=line
