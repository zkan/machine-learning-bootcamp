import numpy as np
import pandas as pd

from schema import NUMERIC, PSI_LIMIT


def population_stability(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.astype(float).to_numpy()
    current = current.astype(float).to_numpy()
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    ref_counts = np.histogram(reference, bins=cuts)[0].astype(float)
    cur_counts = np.histogram(current, bins=cuts)[0].astype(float)
    ref_share = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_share = np.clip(cur_counts / max(cur_counts.sum(), 1.0), 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in NUMERIC:
        score = population_stability(reference[name], current[name])
        rows.append({"column": name, "psi": round(score, 4), "check": score >= PSI_LIMIT})
    return pd.DataFrame(rows)
