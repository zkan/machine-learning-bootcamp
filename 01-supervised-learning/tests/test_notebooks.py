import json
import subprocess
import sys
from pathlib import Path

from paths import module_root

NOTEBOOKS = (
    "00_start.ipynb",
    "01_eda.ipynb",
    "02_split.ipynb",
    "03_train_cv.ipynb",
    "04_tune.ipynb",
    "05_evaluate.ipynb",
    "06_regression.ipynb",
)


def _notebook(name):
    path = module_root() / "notebooks" / name
    return json.loads(path.read_text())


def _code(notebook, index):
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return "".join(cells[index]["source"])


def test_notebook_names_and_exercises():
    folder = module_root() / "notebooks"
    assert [path.name for path in sorted(folder.glob("*.ipynb"))] == list(NOTEBOOKS)
    for name in NOTEBOOKS:
        text = (folder / name).read_text()
        assert "Exercise" in text or "exercise" in text
        assert "churn_bundle.joblib" not in text or name == "05_evaluate.ipynb"


def test_public_notebooks_keep_placeholders():
    start = (module_root() / "notebooks" / "00_start.ipynb").read_text()
    assert "NotImplementedError" in start
    assert '"plan": "none"' not in start
    evaluate = (module_root() / "notebooks" / "05_evaluate.ipynb").read_text()
    assert "test_precision = None" in evaluate
    assert "test_precision = float" not in evaluate


def test_boot_cell_finds_the_module_from_supported_folders():
    source = _code(_notebook("01_eda.ipynb"), 0)
    folders = (module_root().parent, module_root(), module_root() / "notebooks")
    for folder in folders:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=folder,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert f"module {module_root()}" in completed.stdout
