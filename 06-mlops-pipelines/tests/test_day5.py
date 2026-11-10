import ast
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from churn_tasks import register, validate
from drift import population_stability
from schema import DAGS, GATE, METRICS, MODULE, N_ROWS, PSI_LIMIT


def test_validate_rejects_a_short_table(tmp_path):
    path = tmp_path / "bad.parquet"
    frame = pd.DataFrame({"orders_90d": [1], "churned": [0], "customer_id": ["FG0000"]})
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    try:
        validate(path)
    except ValueError as exc:
        assert "2400" in str(exc) or str(N_ROWS) in str(exc)
    else:
        raise AssertionError("validate accepted a short table")


def test_identical_columns_have_no_drift():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5)
    assert population_stability(series, series) < PSI_LIMIT


def test_shifted_column_is_over_the_line():
    reference = pd.Series(range(100), dtype=float)
    current = reference + 80
    assert population_stability(reference, current) >= PSI_LIMIT


def test_gate_blocks_a_weak_score():
    original = METRICS.read_text() if METRICS.is_file() else None
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps({"roc_auc": 0.1, "model_uri": "models:/missing", "run_id": "none"}))
    try:
        register()
    except RuntimeError as exc:
        assert str(GATE) in str(exc)
    else:
        raise AssertionError("register accepted a score under the gate")
    finally:
        if original is None:
            METRICS.unlink(missing_ok=True)
        else:
            METRICS.write_text(original)


def test_runner_twice_keeps_the_same_run():
    from runner import run_all

    run_all()
    first = json.loads(METRICS.read_text())
    run_all()
    second = json.loads(METRICS.read_text())
    assert first["roc_auc"] == second["roc_auc"]
    assert first["passed"] is True
    assert first["run_id"] == second["run_id"]
    assert first["roc_auc"] >= GATE


def test_dag_files_parse_and_the_bounty_file_stays_unsolved():
    working = (DAGS / "churn_train.py").read_text()
    buggy = (DAGS / "churn_train_buggy.py").read_text()
    ast.parse(working)
    ast.parse(buggy)
    assert 'schedule="@daily"' in working
    notebook = (MODULE / "01_bug_bounty.ipynb").read_text()
    assert "NotImplementedError" in notebook
    assert "label_copy" not in notebook
    assert "customers.csv" not in notebook
