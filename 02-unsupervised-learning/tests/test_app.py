import os
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_explains_missing_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("DAY2_SEGMENTS", str(tmp_path / "segments.csv"))
    monkeypatch.setenv("DAY2_NEIGHBORS", str(tmp_path / "neighbors.csv"))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    text = " ".join(item.value for item in at.error)
    assert "segments.csv" in text
    assert "neighbors.csv" in text


def test_app_shows_saved_tables(monkeypatch, tmp_path):
    segments = tmp_path / "segments.csv"
    neighbors = tmp_path / "neighbors.csv"
    pd.DataFrame(
        [
            {
                "segment": 0,
                "customers": 10,
                "churn_rate": 0.2,
                "recency_days": 12,
                "orders_90d": 4,
                "spend_thb_90d": 800,
                "persona": "weekly cooks",
                "action": "keep the rice bundle",
            }
        ]
    ).to_csv(segments, index=False)
    pd.DataFrame(
        [
            {"product": "rice", "rank": 1, "neighbor": "cooking_oil", "similarity": 0.8},
            {"product": "milk", "rank": 1, "neighbor": "eggs", "similarity": 0.7},
        ]
    ).to_csv(neighbors, index=False)
    monkeypatch.setenv("DAY2_SEGMENTS", str(segments))
    monkeypatch.setenv("DAY2_NEIGHBORS", str(neighbors))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    blob = " ".join(str(item.value) for item in at.dataframe)
    assert "weekly cooks" in blob
    assert "cooking_oil" in blob or "rice" in blob
    assert os.environ["DAY2_SEGMENTS"] == str(segments)
