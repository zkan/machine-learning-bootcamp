from customers import load_customers, split_frame
from training import curve_table, learning_rate_table, scaling_table, stopping_table


def _frames():
    train, validation, _test = split_frame(load_customers())
    return train, validation


def test_learning_rate_and_scaling():
    train, validation = _frames()
    rates = learning_rate_table(train, validation).set_index("setting")
    assert round(rates.loc["sgd 0.001", "validation_roc_auc"], 3) == 0.656
    assert round(rates.loc["sgd 0.01", "validation_roc_auc"], 3) == 0.772
    assert round(rates.loc["sgd 1.0", "train_roc_auc"], 3) == 0.871
    assert round(rates.loc["sgd 1.0", "validation_roc_auc"], 3) == 0.748
    assert rates.loc["sgd 1.0", "validation_roc_auc"] < rates.loc["sgd 0.01", "validation_roc_auc"]
    scaled = scaling_table(train, validation).set_index("setting")
    assert round(scaled.loc["scaled", "validation_roc_auc"], 3) == 0.772
    assert round(scaled.loc["not scaled", "validation_roc_auc"], 3) == 0.542
    assert int(scaled.loc["not scaled", "iterations"]) == 13


def test_early_stopping_shrinks_the_gap_and_the_curve_rises():
    train, validation = _frames()
    stop = stopping_table(train, validation).set_index("setting")
    plain = stop.loc["no early stopping"]
    early = stop.loc["early stopping"]
    assert round(plain["train_roc_auc"], 3) == 0.913
    assert round(plain["validation_roc_auc"], 3) == 0.756
    assert round(early["train_roc_auc"], 3) == 0.812
    assert round(early["validation_roc_auc"], 3) == 0.787
    assert int(plain["iterations"]) == 80
    assert int(early["iterations"]) == 27
    assert round(plain["train_roc_auc"] - plain["validation_roc_auc"], 3) == 0.157
    assert (plain["train_roc_auc"] - plain["validation_roc_auc"]) > (
        early["train_roc_auc"] - early["validation_roc_auc"]
    )
    curve = curve_table(train)
    assert curve["train_rows"].tolist() == [240, 480, 720, 960]
    assert [round(value, 3) for value in curve["train_roc_auc"]] == [0.653, 0.693, 0.729, 0.750]
    assert [round(value, 3) for value in curve["cv_roc_auc"]] == [0.618, 0.669, 0.704, 0.728]
    assert curve["cv_roc_auc"].is_monotonic_increasing
