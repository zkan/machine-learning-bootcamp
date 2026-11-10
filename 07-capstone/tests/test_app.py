from pathlib import Path

from streamlit.testing.v1 import AppTest

MODULE = Path(__file__).resolve().parents[1]


def test_capstone_form_opens():
    at = AppTest.from_file(MODULE / "app.py", default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Fresh and Good capstone"
    assert at.radio[0].value == "Score"


def test_notebook_keeps_the_exercise():
    text = (MODULE / "00_capstone.ipynb").read_text()
    assert "NotImplementedError" in text
    assert "Exercise" in text
