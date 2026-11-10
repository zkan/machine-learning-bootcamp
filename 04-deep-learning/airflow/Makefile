# Airflow clinic. Run make from 04-deep-learning/airflow.
# The repo mise.toml points UV_PYTHON at Python 3.15.2. This folder needs Python 3.12.16.
# env -u UV_PYTHON drops that pin for these commands only.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- env -u UV_PYTHON uv

export AIRFLOW_HOME := $(HERE)/home
export AIRFLOW__CORE__LOAD_EXAMPLES := false
export AIRFLOW__CORE__DAGS_FOLDER := $(HERE)/dags

.DEFAULT_GOAL := help

.PHONY: help setup sync migrate dag runner standalone

help:
	@printf '%s\n' \
		'Day 3 Airflow clinic' \
		'' \
		'  make setup       locked Airflow packages on Python 3.12.16' \
		'  make sync        uv sync --frozen --python 3.12.16' \
		'  make migrate     airflow db migrate' \
		'  make dag         airflow dags test fresh_and_good_clinic' \
		'  make runner      plain-Python run of the same tasks' \
		'  make standalone  airflow standalone (UI on localhost:8080)'

setup: sync

sync:
	cd $(HERE) && $(UV) sync --frozen --python 3.12.16

migrate:
	cd $(HERE) && $(UV) run --python 3.12.16 airflow db migrate

dag: migrate
	cd $(HERE) && $(UV) run --python 3.12.16 airflow dags test fresh_and_good_clinic 2026-07-01

runner:
	cd $(HERE) && $(UV) run --python 3.12.16 python runner.py

standalone:
	cd $(HERE) && $(UV) run --python 3.12.16 airflow standalone
