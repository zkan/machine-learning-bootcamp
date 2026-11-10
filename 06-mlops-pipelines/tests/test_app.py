from streamlit.testing.v1 import AppTest

from schema import MODULE


def test_form_waits_for_a_customer():
    at = AppTest.from_file(MODULE / "app.py", default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Fresh and Good pipeline demo"
    assert at.radio[0].value == "Score"
