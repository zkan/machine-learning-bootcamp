import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAGS = ROOT / "airflow" / "dags"
if str(DAGS) not in sys.path:
    sys.path.insert(0, str(DAGS))
if str(ROOT / "lib") not in sys.path:
    sys.path.insert(0, str(ROOT / "lib"))

from churn_tasks import evaluate, features, ingest, register, train, validate
from schema import ATTEMPTS

STAGES = (
    ("ingest", ingest),
    ("validate", validate),
    ("features", features),
    ("train", train),
    ("evaluate", evaluate),
    ("register", register),
)


def run_stage(name: str, fn, attempts: int = ATTEMPTS):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            result = fn()
            print(f"{name} ok attempt {attempt}")
            return result
        except Exception as exc:
            last = exc
            print(f"{name} failed attempt {attempt}: {exc}")
    raise last


def run_all(attempts: int = ATTEMPTS) -> None:
    for name, fn in STAGES:
        run_stage(name, fn, attempts=attempts)


if __name__ == "__main__":
    run_all()
