"""Day 2 table: the Day 1 customers plus two columns that use July."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
DAY1_CSV = MODULE.parent / "01-supervised-learning" / "data" / "fresh_and_good_customers.csv"
DAY2_CSV = MODULE / "data" / "fresh_and_good_day2.csv"

LABELS = ("churned", "next_month_spend_thb")
# These two are planted. Both are known only after 30 June 2026.
LEAK_COLUMNS = ("orders_next_30d", "july_spend_copy_thb")
SEED = 42


def load_day1() -> pd.DataFrame:
    if not DAY1_CSV.is_file():
        raise FileNotFoundError(DAY1_CSV)
    return pd.read_csv(DAY1_CSV)


def plant_leaks(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add the two July columns. Row order follows the Day 1 file."""
    out = frame.copy()
    churned = out["churned"].to_numpy()
    rng = np.random.default_rng(seed)
    kept = rng.integers(1, 6, size=len(out))
    out["orders_next_30d"] = np.where(churned == 1, 0, kept).astype(int)
    out["july_spend_copy_thb"] = out["next_month_spend_thb"].astype(float)
    return out


def built_table() -> pd.DataFrame:
    return plant_leaks(load_day1())


def safe_feature_columns(columns) -> list[str]:
    banned = set(LEAK_COLUMNS) | set(LABELS)
    return [name for name in columns if name not in banned]


def write_table(path: Path = DAY2_CSV) -> pd.DataFrame:
    table = built_table()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def check_table(path: Path = DAY2_CSV) -> None:
    built = built_table()
    saved = pd.read_csv(path)
    pd.testing.assert_frame_equal(
        built.reset_index(drop=True),
        saved.reset_index(drop=True),
        check_dtype=False,
    )
    leaked = plant_leaks(load_day1())
    if not leaked["orders_next_30d"].eq(0).equals(leaked["churned"].eq(1)):
        raise AssertionError("orders_next_30d is a July churn flag")
    if not np.allclose(leaked["july_spend_copy_thb"], leaked["next_month_spend_thb"]):
        raise AssertionError("july_spend_copy_thb copies July spend")


def main(argv: list[str]) -> int:
    if "--write" in argv:
        table = write_table()
        print(f"wrote {DAY2_CSV} rows {len(table)}")
        return 0
    if "--check" in argv:
        check_table()
        print(f"day2 table matches planted leaks ({DAY2_CSV.name})")
        return 0
    print("usage: python -m leaks --check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
