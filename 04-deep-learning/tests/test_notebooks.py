import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "00_mlp_churn.ipynb",
    "01_training.ipynb",
    "02_opencv.ipynb",
    "03_cnn.ipynb",
    "04_mlflow.ipynb",
)


def _notebook(name):
    return json.loads((ROOT / name).read_text())


def _code(notebook, index):
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    return "".join(cells[index]["source"])


def _plain(name):
    notebook = json.loads((ROOT / name).read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_notebooks_keep_the_exercises():
    assert [path.name for path in sorted(ROOT.glob("*.ipynb"))] == list(NOTEBOOKS)
    for name in NOTEBOOKS:
        assert "Exercise" in _plain(name)
    assert 'debrief = ""' in _plain("00_mlp_churn.ipynb")
    assert "gap = None" in _plain("01_training.ipynb")
    assert 'debrief = ""' in _plain("03_cnn.ipynb")
    assert "learning_rate = 0.001" in _plain("03_cnn.ipynb")
    memo = (ROOT / "why_this_model.md").read_text()
    assert "we ship this model because" in memo
    assert "0.698" not in memo
    assert "0.782" not in memo


def test_boot_cell_finds_the_module():
    source = _code(_notebook("00_mlp_churn.ipynb"), 0)
    for folder in (ROOT.parent, ROOT):
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=folder,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert f"module {ROOT}" in completed.stdout
