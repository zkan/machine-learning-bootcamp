from streamlit.testing.v1 import AppTest

from schema import MODULE


def _errors(at: AppTest) -> str:
    return " ".join(item.value for item in at.error)


def test_form_waits_for_a_customer_id():
    at = AppTest.from_file(MODULE / "app.py", default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "Fresh and Good registry demo"
    assert "FG0010" in " ".join(item.value for item in at.info)


def test_bad_customer_id_shows_the_format():
    at = AppTest.from_file(MODULE / "app.py", default_timeout=30).run()
    at.text_input[0].set_value("nope")
    at.button[0].click().run()
    assert "FG0010" in _errors(at)
