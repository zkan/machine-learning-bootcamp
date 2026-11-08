import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frame import features_and_target
from paths import data_csv
from persist import TASK_CHURN, DecisionBundle, load_bundle, save_bundle, save_spend_bundle
from pipelines import make_logistic_pipeline, tune_forest
from schema import CHURN_TARGET, FEATURE_COLUMNS, SPEND_TARGET, TARGET
from split import split_customers
from threshold import choose_threshold, labels_at, total_cost


def test_threshold_on_a_known_card():
    actual = [0, 0, 1, 1]
    proba = [0.2, 0.55, 0.45, 0.8]
    low = total_cost(actual, labels_at(proba, 0.25), 5, 1)
    high = total_cost(actual, labels_at(proba, 0.45), 5, 1)
    assert low.cost_thb == high.cost_thb == 1
    assert low.false_negatives == 0
    assert low.false_positives == 1
    chosen = choose_threshold(actual, proba, cost_fn_thb=5, cost_fp_thb=1)
    assert chosen == 0.25


def test_higher_miss_cost_does_not_raise_the_threshold():
    actual = [0, 0, 1, 1]
    proba = [0.2, 0.55, 0.45, 0.8]
    cautious = choose_threshold(actual, proba, cost_fn_thb=5, cost_fp_thb=1)
    relaxed = choose_threshold(actual, proba, cost_fn_thb=1, cost_fp_thb=5)
    assert cautious <= relaxed


def test_split_fit_does_not_use_test_rows_for_scaling():
    frame = pd.read_csv(data_csv())
    train, validation, test = split_customers(frame)
    x_train, y_train = features_and_target(train)
    x_test, y_test = features_and_target(test)
    assert set(train.index).isdisjoint(test.index)
    pipe = make_logistic_pipeline()
    pipe.fit(x_train, y_train)
    train_mean = pipe.named_steps["prep"].named_transformers_["num"].mean_.copy()
    both_x = pd.concat([x_train, x_test])
    both_y = pd.concat([y_train, y_test])
    leaked = make_logistic_pipeline()
    leaked.fit(both_x, both_y)
    leaked_mean = leaked.named_steps["prep"].named_transformers_["num"].mean_
    assert not np.allclose(train_mean, leaked_mean)
    scores = pipe.predict_proba(x_train.iloc[:20])[:, 1]
    assert ((scores >= 0) & (scores <= 1)).all()
    validation_x, validation_y = features_and_target(validation)
    auc = roc_auc_score(validation_y, pipe.predict_proba(validation_x)[:, 1])
    assert 0.70 < auc < 0.95


def test_small_search_returns_a_fitted_pipeline():
    frame = pd.read_csv(data_csv())
    train, _validation, _test = split_customers(frame)
    x_train, y_train = features_and_target(train)
    search = tune_forest(x_train.iloc[:600], y_train.iloc[:600], n_iter=2)
    assert np.isfinite(search.best_score_)
    proba = search.predict_proba(x_train.iloc[:5])
    assert proba.shape == (5, 2)
    assert set(search.best_params_).issubset(
        {"model__n_estimators", "model__max_depth", "model__min_samples_leaf"}
    )


def test_bundle_round_trip_and_column_guard(tmp_path):
    frame = pd.read_csv(data_csv())
    train, _validation, _test = split_customers(frame)
    x_train, y_train = features_and_target(train)
    pipe = make_logistic_pipeline()
    pipe.fit(x_train, y_train)
    path = tmp_path / "bundle.joblib"
    save_bundle(path, pipe, 0.35, 500, 100)
    loaded = load_bundle(path)
    assert loaded.threshold == 0.35
    assert loaded.feature_columns == FEATURE_COLUMNS
    fresh = loaded.pipeline.predict_proba(x_train.iloc[:4])
    again = pipe.predict_proba(x_train.iloc[:4])
    assert np.allclose(fresh, again)
    broken = DecisionBundle(
        pipeline=pipe,
        task=TASK_CHURN,
        target=TARGET,
        feature_columns=("tenure_days",),
        synthetic=True,
        threshold=0.35,
        cost_fn_thb=500,
        cost_fp_thb=100,
    )
    import joblib

    other = tmp_path / "other.joblib"
    joblib.dump(broken, other)
    try:
        load_bundle(other)
    except ValueError as exc:
        assert "columns" in str(exc)
    else:
        raise AssertionError("mismatched columns should fail the load")
    missing = tmp_path / "absent.joblib"
    try:
        load_bundle(missing)
    except FileNotFoundError as exc:
        assert "trust" in str(exc)
    else:
        raise AssertionError("a missing bundle should fail the load")
    assert TARGET not in x_train.columns
    assert SPEND_TARGET not in x_train.columns
    spend_features, spend_labels = features_and_target(train, SPEND_TARGET)
    assert CHURN_TARGET not in spend_features.columns
    assert SPEND_TARGET not in spend_features.columns
    spend_path = tmp_path / "spend.joblib"
    from pipelines import make_ridge_pipeline

    ridge = make_ridge_pipeline()
    ridge.fit(spend_features, spend_labels)
    save_spend_bundle(spend_path, ridge)
    try:
        load_bundle(spend_path, expected_task=TASK_CHURN)
    except ValueError as exc:
        assert "spend" in str(exc) and "churn" in str(exc)
    else:
        raise AssertionError("a spend bundle must not load as churn")
    from spend import clip_spend, spend_scores

    scores = spend_scores([0.0, 100.0], [-25.0, 80.0])
    assert clip_spend([-25.0])[0] == 0.0
    assert scores["mae_thb"] == 10.0
