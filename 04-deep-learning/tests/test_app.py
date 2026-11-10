import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from customers import compare_models, load_customers, save_bundle, split_frame

APP = Path(__file__).resolve().parents[1] / "app.py"


def _text(at: AppTest) -> str:
    parts = []
    for group in (at.markdown, at.caption, at.error, at.metric, at.info):
        for item in group:
            parts.append(str(item.value))
    return "\n".join(parts)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("models")
    train, validation, _test = split_frame(load_customers())
    result = compare_models(train, validation, (32,))
    save_bundle(folder / "classic.joblib", result["classic"], "classic", "RandomForestClassifier")
    save_bundle(folder / "mlp.joblib", result["mlp"], "mlp", "MLPClassifier")
    (folder / "chosen.json").write_text(
        json.dumps({"model": "mlp", "version": "1", "registered_name": "fresh-and-good-churn"})
    )
    return folder


def test_missing_files_name_the_notebook(monkeypatch, tmp_path):
    monkeypatch.setenv("DAY3_CLASSIC", str(tmp_path / "classic.joblib"))
    monkeypatch.setenv("DAY3_MLP", str(tmp_path / "mlp.joblib"))
    monkeypatch.setenv("DAY3_CHOSEN", str(tmp_path / "chosen.json"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    text = _text(at)
    assert "classic.joblib" in text
    assert "03_swap.ipynb" in text


def test_switch_shows_the_model_and_scores(monkeypatch, model_dir):
    monkeypatch.setenv("DAY3_CLASSIC", str(model_dir / "classic.joblib"))
    monkeypatch.setenv("DAY3_MLP", str(model_dir / "mlp.joblib"))
    monkeypatch.setenv("DAY3_CHOSEN", str(model_dir / "chosen.json"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert "RandomForestClassifier, not registered" in _text(at)
    at.form_submit_button[0].click().run()
    classic = at.metric[0].value
    assert classic.endswith("%")
    at.radio[0].set_value("mlp").run()
    assert "MLPClassifier, MLflow version 1" in _text(at)
    at.form_submit_button[0].click().run()
    assert at.metric[0].value.endswith("%")
    assert at.metric[0].value != classic
