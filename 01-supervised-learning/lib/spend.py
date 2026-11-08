import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def clip_spend(prediction) -> np.ndarray:
    """Spending cannot be negative. Evaluation and the form both use this clip."""
    return np.maximum(np.asarray(prediction, dtype=float), 0.0)


def spend_scores(y_true, y_pred) -> dict[str, float]:
    clipped = clip_spend(y_pred)
    return {
        "mae_thb": float(mean_absolute_error(y_true, clipped)),
        "rmse_thb": float(root_mean_squared_error(y_true, clipped)),
        "r2": float(r2_score(y_true, clipped)),
    }
