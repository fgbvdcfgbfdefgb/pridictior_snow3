from __future__ import annotations

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    denom = np.maximum(np.abs(actual), 1e-9)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mape_pct": float(np.mean(np.abs(error) / denom) * 100),
        "within_0_5pct": float(np.mean(np.abs(error) / denom <= 0.005) * 100),
        "directional_accuracy_pct": float(np.mean(np.sign(np.diff(predicted)) == np.sign(np.diff(actual))) * 100) if len(actual) > 1 else 0.0,
    }
