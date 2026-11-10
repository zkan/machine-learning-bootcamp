"""Run the clinic tasks without Airflow.

Attempts match the Dag: retries=1 means two tries. A failed try is printed,
then the task runs again. The tasks are the functions in clinic_tasks.py.
"""

from clinic_tasks import count_open_tickets, publish, summarize


def call_with_retries(fn, attempts: int, args=(), log=print):
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    last = None
    for attempt in range(1, attempts + 1):
        try:
            result = fn(*args)
        except Exception as exc:
            last = exc
            log(f"{fn.__name__} failed attempt {attempt} of {attempts}: {exc}")
            continue
        log(f"{fn.__name__} ok attempt {attempt}")
        return result
    raise last


def run(log=print) -> str:
    count = call_with_retries(count_open_tickets, 2, (), log)
    summary = call_with_retries(summarize, 2, (count,), log)
    return call_with_retries(publish, 2, (summary,), log)


if __name__ == "__main__":
    run()
