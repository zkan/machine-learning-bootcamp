import pandas as pd
from streamlit.testing.v1 import AppTest

from day1.frame import features_and_target
from day1.models import make_logistic_pipeline
from day1.paths import data_csv, module_root
from day1.persist import save_bundle
from day1.split import split_customers
from day1.validate import input_errors


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


def test_missing_bundle_tells_you_what_to_finish(monkeypatch, tmp_path):
    monkeypatch.setenv("KLONGCART_BUNDLE", str(tmp_path / "missing.joblib"))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    text = _errors(at)
    assert "notebook 05" in text
    assert "trust" in text


def test_zero_orders_with_spend_shows_the_input_error(monkeypatch, tmp_path):
    monkeypatch.setenv("KLONGCART_BUNDLE", str(_bundle(tmp_path)))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    at.number_input[1].set_value(0)
    at.number_input[2].set_value(50.0)
    at.form_submit_button[0].click().run()
    assert "spend_thb_90d must be 0 when orders_90d is 0." in _errors(at)


def test_starter_scores_a_probability_and_stops_on_the_exercise(monkeypatch, tmp_path):
    monkeypatch.setenv("KLONGCART_BUNDLE", str(_bundle(tmp_path)))
    at = AppTest.from_file(module_root() / "app.py", default_timeout=30).run()
    at.form_submit_button[0].click().run()
    assert at.metric
    assert "%" in at.metric[0].value
    assert "Finish churn_decision" in _errors(at)
    assert not any(item.value.startswith("Offer a voucher:") for item in at.markdown)


def test_public_app_keeps_the_exercise():
    source = (module_root() / "app.py").read_text()
    assert "raise NotImplementedError" in source
    assert 'return "yes"' not in source
