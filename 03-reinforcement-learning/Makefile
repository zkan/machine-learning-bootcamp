# Reinforcement learning commands. Run make from this folder.
# mise.toml still points PYTHONPATH at Day 1's lib. env overrides it for these commands only.

HERE := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
UV := mise exec -- env PYTHONPATH=lib uv

.DEFAULT_GOAL := help

.PHONY: help setup sync kernel notebook test

help:
	@printf '%s\n' \
		'Reinforcement learning' \
		'' \
		'  make setup    locked packages and the notebook kernel' \
		'  make sync     uv sync --frozen' \
		'  make kernel   register the fresh-and-good-rl kernel' \
		'  make notebook start Jupyter' \
		'  make test     teaching-team checks'

setup: sync kernel

sync:
	cd $(HERE) && $(UV) sync --frozen

kernel:
	cd $(HERE) && $(UV) run python -m ipykernel install --user --name fresh-and-good-rl --display-name "Python (fresh-and-good-rl)"

notebook:
	cd $(HERE) && $(UV) run python -m jupyter notebook

test:
	cd $(HERE) && $(UV) sync --frozen --group dev
	cd $(HERE) && $(UV) run --group dev python -m pytest tests -q --tb=line
