import json
import subprocess
import sys

from schema import MODULE

NOTEBOOKS = (
    "00_reproduce.ipynb",
    "01_registry.ipynb",
    "02_feast.ipynb",
    "03_skew_hunt.ipynb",
)


def test_notebook_boot_finds_the_module():
    notebook = json.loads((MODULE / NOTEBOOKS[0]).read_text())
    source = "".join(notebook["cells"][1]["source"])
    for folder in (MODULE, MODULE.parent):
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=folder,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert f"module {MODULE}" in completed.stdout
