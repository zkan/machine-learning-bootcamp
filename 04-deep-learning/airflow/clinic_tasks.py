"""Three plain tasks. The Dag and the Python runner both call these.

The number 12 is the clinic example. It is not a live ticket queue.
"""


def count_open_tickets() -> int:
    return 12


def summarize(count: int) -> dict:
    if count < 0:
        raise ValueError("count must be zero or more")
    return {"open_tickets": int(count), "note": "clinic only, not a real queue"}


def publish(summary: dict) -> str:
    line = f"open tickets {summary['open_tickets']}"
    print(line)
    return line
