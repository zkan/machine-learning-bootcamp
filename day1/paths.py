"""Repo paths. Notebooks and the app both start from these."""

from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_csv() -> Path:
    return package_root() / "data" / "klongcart_churn.csv"


def default_bundle() -> Path:
    return package_root() / "models" / "churn_bundle.joblib"


def repo_root_from(start: Path) -> Path:
    """Return the repo root when start is the repo or the notebooks folder."""
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "day1" / "schema.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find the bootcamp repo. Start Jupyter from the repo folder or from notebooks."
    )
