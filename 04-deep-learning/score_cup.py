"""Score a cup prediction file against the instructor label file.

The label file is not in git. The teaching team keeps it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from cup import score_files


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python score_cup.py predictions.csv labels.csv", file=sys.stderr)
        return 2
    accuracy = score_files(Path(argv[0]), Path(argv[1]))
    print(f"held-out accuracy {accuracy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
