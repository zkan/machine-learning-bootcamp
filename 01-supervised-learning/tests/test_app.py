import pandas as pd
from streamlit.testing.v1 import AppTest

from frame import features_and_target
from paths import data_csv, module_root
from persist import save_bundle, save_spend_bundle
from pipelines import make_logistic_pipeline, make_ridge_pipeline
from split import split_customers
from validate import input_errors


def _bundle(tmp_path):
    frame = pd.read_csv(data_csv())
    train, _validation, _test = split_customers(frame)
    features, labels = features_and_target(train)
    pipeline = make_logistic_pipeline()
    pipeline.fit(features, labels)
    path = tmp_path / "churn_bundle.joblib"
    save_bundle(path, pipeline, 0.4, 500, 100)
    return path


def _errors(at: AppTest) -> str:
    return " ".join(item.value for item in at.error)


def test_negative_tenure_is_rejected():
    record = {
        "tenure_days": -1,
        "orders_90d": 1,
        "spend_thb_90d": 100.0,
        "avg_basket_thb_90d": 100.0,
        "days_since_last_order": 3,
        "promo_order_share_90d": 0.1,
        "app_sessions_90d": 2,
        "support_tickets_90d": 0,
        "district": "Bang Kapi",
        "plan": "none",
        "primary_aisle": "fresh",
    }
    assert any("tenure_days" in message for message in input_errors(record))


def _spend_bundle(tmp_path, train):
    from schema import SPEND_TARGET

    features, labels = features_and_target(train, SPEND_TARGET)
    pipeline = make_ridge_pipeline()
    pipeline.fit(features, labels)
    path = tmp_path / "spend_bundle.joblib"
    save_spend_bundle(path, pipeline)
    return path


def test_missing_bundle_tells_you_what_to_finish(monkeypatch, tmp_path):
    monkeypatch.setenv("FRESH_AND_GOOD_CHURN_BUNDLE", str(tmp_path / "missing-churn.joblib"))
    monkeypatch.setenv("FRESH_AND_GOOD_SPEND_BUNDLE", str(tmp_path / "missing-spend.joblib"))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    text = _errors(at)
    assert "notebook 05" in text
    assert "notebook 06" in text
    assert "trust" in text


def test_zero_orders_with_spend_shows_the_input_error(monkeypatch, tmp_path):
    monkeypatch.setenv("FRESH_AND_GOOD_CHURN_BUNDLE", str(_bundle(tmp_path)))
    monkeypatch.setenv("FRESH_AND_GOOD_SPEND_BUNDLE", str(tmp_path / "missing-spend.joblib"))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    at.number_input[1].set_value(0)
    at.number_input[2].set_value(50.0)
    at.form_submit_button[0].click().run()
    assert "spend_thb_90d must be 0 when orders_90d is 0." in _errors(at)


def test_starter_scores_a_probability_and_stops_on_the_exercise(monkeypatch, tmp_path):
    monkeypatch.setenv("FRESH_AND_GOOD_CHURN_BUNDLE", str(_bundle(tmp_path)))
    monkeypatch.setenv("FRESH_AND_GOOD_SPEND_BUNDLE", str(tmp_path / "missing-spend.joblib"))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    at.form_submit_button[0].click().run()
    assert at.metric
    assert "%" in at.metric[0].value
    assert "Finish churn_decision" in _errors(at)
    assert not any(item.value.startswith("Offer a voucher:") for item in at.markdown)


def test_form_shows_churn_and_spend_and_rejects_the_wrong_bundle(monkeypatch, tmp_path):
    import pandas as pd

    frame = pd.read_csv(data_csv())
    train, _validation, _test = split_customers(frame)
    churn_path = _bundle(tmp_path)
    spend_path = _spend_bundle(tmp_path, train)
    monkeypatch.setenv("FRESH_AND_GOOD_CHURN_BUNDLE", str(churn_path))
    monkeypatch.setenv("FRESH_AND_GOOD_SPEND_BUNDLE", str(spend_path))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    at.form_submit_button[0].click().run()
    labels = [item.label for item in at.metric]
    assert "Churn probability" in labels
    assert "Estimated next-month spend" in labels
    assert any(item.value.endswith("THB") for item in at.metric)
    monkeypatch.setenv("FRESH_AND_GOOD_SPEND_BUNDLE", str(churn_path))
    swapped = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    assert any("not a spend bundle" in item.value for item in swapped.error)


def test_public_app_keeps_the_exercise():
    source = (module_root() / "app.py").read_text()
    assert "raise NotImplementedError" in source
    assert 'return "yes"' not in source
