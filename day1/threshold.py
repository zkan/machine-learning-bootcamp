from typing import NamedTuple

import numpy as np

DEFAULT_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


class CostCount(NamedTuple):
    cost_thb: float
    false_negatives: int
    false_positives: int


def labels_at(proba, threshold: float) -> np.ndarray:
    return (np.asarray(proba) >= threshold).astype(int)


def total_cost(y_true, y_pred, cost_fn_thb: float, cost_fp_thb: float) -> CostCount:
    actual = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_pred).astype(int)
    false_negatives = int(np.sum((actual == 1) & (predicted == 0)))
    false_positives = int(np.sum((actual == 0) & (predicted == 1)))
    cost = false_negatives * float(cost_fn_thb) + false_positives * float(cost_fp_thb)
    return CostCount(cost, false_negatives, false_positives)


def choose_threshold(
    y_true,
    proba,
    cost_fn_thb: float,
    cost_fp_thb: float,
    grid: np.ndarray | None = None,
) -> float:
    if len(y_true) != len(proba):
        raise ValueError("y_true and proba must have the same length.")
    candidates = DEFAULT_GRID if grid is None else np.asarray(grid, dtype=float)
    scored = []
    for threshold in candidates:
        cost = total_cost(
            y_true, labels_at(proba, float(threshold)), cost_fn_thb, cost_fp_thb
        )
        scored.append((cost.cost_thb, float(threshold)))
    if not scored:
        raise ValueError("The threshold grid is empty.")
    _best_cost, threshold = min(scored)
    return threshold
