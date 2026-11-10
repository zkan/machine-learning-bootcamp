import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from features import feast_table, serving_frame, training_frame
from prepare import customers_from, load_day1
from registry import load_champion, register_champion
from schema import (
    ALIAS,
    DAY1_CSV,
    MODEL_FEATURES,
    MODEL_NAME,
    N_ROWS,
    TARGET,
    customer_id,
    parse_customer_id,
)
from train import fit_churn


def test_day1_table_is_the_only_source():
    frame = load_day1()
    assert len(frame) == N_ROWS
    assert DAY1_CSV.name == "fresh_and_good_customers.csv"
    assert "customer_id" not in frame.columns


def test_customer_ids_follow_file_order():
    table = customers_from(load_day1())
    ids = table.column("customer_id").to_pylist()
    assert ids[0] == "FG0000"
    assert ids[-1] == customer_id(N_ROWS - 1)
    assert len(set(ids)) == N_ROWS
    with pytest.raises(ValueError):
        parse_customer_id("customer-1")


def test_training_and_serving_rows_differ():
    customers = customers_from(load_day1()).to_pandas()
    trained = training_frame(customers).drop(columns=[TARGET]).iloc[[4]].reset_index(drop=True)
    served = serving_frame(customers).iloc[[4]].reset_index(drop=True)
    assert list(trained.columns) == list(MODEL_FEATURES)
    assert not trained.equals(served)


def test_feast_table_column_order():
    customers = customers_from(load_day1()).to_pandas()
    table = feast_table(customers)
    assert table.column_names[:2] == ["customer_id", "event_timestamp"]
    assert TARGET not in table.column_names
    assert "next_month_spend_thb" not in table.column_names


def test_register_and_load_by_alias(tmp_path):
    customers = customers_from(load_day1()).to_pandas().iloc[:400]
    pipeline, auc, sample = fit_churn(training_frame(customers), 0.25, 42, 1.0)
    assert 0.5 < auc <= 1.0
    version = register_champion(
        pipeline,
        sample,
        auc,
        {"train": {"C": 1.0, "test_size": 0.25, "random_state": 42}},
        path=tmp_path / "mlruns",
    )
    loaded = load_champion(path=tmp_path / "mlruns")
    scored = loaded.predict_proba(sample.loc[:, list(MODEL_FEATURES)])
    assert version == "1"
    assert scored.shape == (len(sample), 2)
    assert MODEL_NAME
    assert ALIAS == "champion"


def test_learner_notebooks_keep_placeholders():
    folder = DAY1_CSV.parents[2] / "05-mlops-reproducibility"
    names = (
        "00_reproduce.ipynb",
        "01_registry.ipynb",
        "02_feast.ipynb",
        "03_skew_hunt.ipynb",
    )
    for name in names:
        text = (folder / name).read_text()
        assert "NotImplementedError" in text
        assert "Exercise" in text
    skew = (folder / "03_skew_hunt.ipynb").read_text()
    assert "basket_momentum_thb" not in skew
    assert "next_month_spend_thb" not in skew
