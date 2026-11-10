from pathlib import Path

import mlflow

from tracking import log_cnn_run


def test_cnn_run_is_logged_without_a_registry(tmp_path):
    artifact = tmp_path / "confusion.csv"
    artifact.write_text("row\n1\n")
    run_id = log_cnn_run(
        tmp_path,
        {"learning_rate": 0.001, "epochs": 2, "source": "digits"},
        {"validation_accuracy": 0.406},
        artifact,
    )
    assert (tmp_path / "tracking").is_dir()
    names = [item.path for item in mlflow.MlflowClient().list_artifacts(run_id)]
    assert "confusion.csv" in names
    assert not (tmp_path / "tracking" / "models").exists()
