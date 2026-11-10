import sys
from pathlib import Path

AIRFLOW = Path(__file__).resolve().parents[1] / "airflow"
sys.path.insert(0, str(AIRFLOW))

import runner


def test_runner_and_retries(capsys):
    lines = []
    assert runner.run(log=lines.append) == "open tickets 12"
    assert "open tickets 12" in capsys.readouterr().out

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("once")
        return "ok"

    notes = []
    assert runner.call_with_retries(flaky, 2, (), notes.append) == "ok"
    assert notes[0].startswith("flaky failed attempt 1")
    assert notes[1] == "flaky ok attempt 2"
