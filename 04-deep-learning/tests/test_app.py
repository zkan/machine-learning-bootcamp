from streamlit.testing.v1 import AppTest

from customers import compare_models, load_customers, save_bundle, split_frame
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"


def _text(at: AppTest) -> str:
    parts = []
    for group in (at.markdown, at.caption, at.error, at.metric, at.info):
        for item in group:
            parts.append(str(getattr(item, "value", item)))
    return "\n".join(parts)


def test_missing_files_name_the_notebooks(monkeypatch, tmp_path):
    monkeypatch.setenv("DAY3_CLASSIC", str(tmp_path / "classic.joblib"))
    monkeypatch.setenv("DAY3_MLP", str(tmp_path / "mlp.joblib"))
    monkeypatch.setenv("DAY3_CNN", str(tmp_path / "cnn.pt"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    text = _text(at)
    assert "classic.joblib" in text
    assert "00_mlp_churn.ipynb" in text
    assert "cnn.pt" in text
    assert "03_cnn.ipynb" in text


def test_churn_switch_scores_both_models(monkeypatch, tmp_path):
    frame = load_customers()
    train, validation, _test = split_frame(frame)
    result = compare_models(train, validation, (32,))
    classic = tmp_path / "classic.joblib"
    mlp = tmp_path / "mlp.joblib"
    save_bundle(classic, result["classic"], "classic", "RandomForestClassifier")
    save_bundle(mlp, result["mlp"], "mlp", "MLPClassifier")
    monkeypatch.setenv("DAY3_CLASSIC", str(classic))
    monkeypatch.setenv("DAY3_MLP", str(mlp))
    monkeypatch.setenv("DAY3_CNN", str(tmp_path / "missing.pt"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert "RandomForestClassifier" in _text(at)
    at.form_submit_button[0].click().run()
    classic_score = at.metric[0].value
    at.radio[0].set_value("mlp").run()
    assert "MLPClassifier" in _text(at)
    at.form_submit_button[0].click().run()
    assert at.metric[0].value.endswith("%")
    assert at.metric[0].value != classic_score
