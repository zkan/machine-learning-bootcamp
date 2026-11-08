from pathlib import Path

MODULE_DIRNAME = "01-supervised-learning"


def module_root() -> Path:
    return Path(__file__).resolve().parent


def data_csv() -> Path:
    return module_root() / "data" / "klongcart_customers.csv"


def churn_bundle_path() -> Path:
    return module_root() / "models" / "churn_bundle.joblib"


def spend_bundle_path() -> Path:
    return module_root() / "models" / "spend_bundle.joblib"


def _is_module(path: Path) -> bool:
    return (path / "schema.py").is_file() and (path / "app.py").is_file()


def module_root_from(start: Path) -> Path:
    """Return the Day 1 folder from the repo, the module, or its notebooks folder."""
    here = start.resolve()
    for candidate in [here, *here.parents]:
        if _is_module(candidate):
            return candidate
        nested = candidate / MODULE_DIRNAME
        if _is_module(nested):
            return nested
    raise FileNotFoundError(
        "Start Jupyter from the repo folder, from 01-supervised-learning, or from its notebooks folder."
    )
