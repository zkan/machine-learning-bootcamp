import pandas as pd

from day1.frame import features_and_target
from day1.paths import data_csv
from day1.schema import N_ROWS, TARGET
from day1.split import split_customers


def _customers() -> pd.DataFrame:
    return pd.read_csv(data_csv())


def test_split_is_disjoint_and_stratified():
    frame = _customers()
    train, validation, test = split_customers(frame)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert len(train) + len(validation) + len(test) == N_ROWS
    assert abs(len(train) - 1440) <= 2
    assert abs(len(validation) - 480) <= 2
    assert abs(len(test) - 480) <= 2
    full_rate = frame[TARGET].mean()
    for part in (train, validation, test):
        assert abs(part[TARGET].mean() - full_rate) < 0.03
        features, _labels = features_and_target(part)
        assert TARGET not in features.columns


def test_same_seed_returns_the_same_rows():
    frame = _customers()
    first = split_customers(frame)
    second = split_customers(frame)
    for left, right in zip(first, second):
        assert left.index.tolist() == right.index.tolist()
