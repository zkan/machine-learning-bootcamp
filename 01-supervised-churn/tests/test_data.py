from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from day1.frame import features_and_target
from day1.generate import customer_bytes, generate_customers, main
from day1.paths import data_csv, module_root_from
from day1.schema import (
    COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_WINDOW_END,
    FEATURE_WINDOW_START,
    LABEL_WINDOW_END,
    LABEL_WINDOW_START,
    LEAKAGE_NAMES,
    N_ROWS,
    SNAPSHOT_DATE,
    TARGET,
)


def test_windows_do_not_overlap():
    feature_days = (FEATURE_WINDOW_END - FEATURE_WINDOW_START).days + 1
    label_days = (LABEL_WINDOW_END - LABEL_WINDOW_START).days + 1
    assert feature_days == 90
    assert label_days == 30
    assert FEATURE_WINDOW_END == SNAPSHOT_DATE
    assert LABEL_WINDOW_START == SNAPSHOT_DATE + timedelta(days=1)
    assert LABEL_WINDOW_START > FEATURE_WINDOW_END


def test_generator_bytes_are_stable(tmp_path: Path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_bytes(customer_bytes())
    second.write_bytes(customer_bytes())
    assert first.read_bytes() == second.read_bytes()


def test_committed_table_matches_generator():
    assert data_csv().read_bytes() == customer_bytes()


def test_check_command_accepts_the_committed_table():
    assert main(["--check"]) == 0


def test_table_shape_and_banned_columns():
    frame = pd.read_csv(data_csv())
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == N_ROWS
    assert set(frame[TARGET].unique()) == {0, 1}
    assert set(frame.columns).isdisjoint(LEAKAGE_NAMES)
    features, labels = features_and_target(frame)
    assert list(features.columns) == list(FEATURE_COLUMNS)
    assert TARGET not in features.columns
    assert labels.tolist() == frame[TARGET].tolist()


def test_extra_future_column_stays_out_of_features():
    frame = pd.read_csv(data_csv())
    frame["orders_next_30d"] = 3
    features, _labels = features_and_target(frame)
    assert "orders_next_30d" not in features.columns


def test_ranges_and_churn_rate():
    frame = pd.read_csv(data_csv())
    assert (frame["tenure_days"] >= 0).all()
    assert (frame["orders_90d"] >= 0).all()
    assert (frame["spend_thb_90d"] >= 0).all()
    assert (frame["days_since_last_order"] >= 0).all()
    assert frame["promo_order_share_90d"].between(0, 1).all()
    zero_orders = frame["orders_90d"] == 0
    assert (frame.loc[zero_orders, "spend_thb_90d"] == 0).all()
    assert int(frame[TARGET].sum()) == 714
    dictionary = (data_csv().parent / "DATA_DICTIONARY.md").read_text()
    for name in COLUMNS:
        assert name in dictionary
    assert "30 June 2026" in dictionary
    assert "next 30 days" in dictionary


def test_module_root_from_repo_module_and_notebooks():
    module = data_csv().parents[1]
    assert module_root_from(module) == module
    assert module_root_from(module / "notebooks") == module
    assert module_root_from(module.parent) == module
    assert date(2026, 6, 30) == SNAPSHOT_DATE
