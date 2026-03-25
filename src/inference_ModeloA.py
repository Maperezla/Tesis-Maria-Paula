from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


class ModelError(Exception):
    """Error asociado al modelo de ML."""


def load_model(model_path: str):
    """Carga un modelo serializado con joblib."""
    model_path = str(Path(model_path))
    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise ModelError("El modelo cargado no tiene el método predict_proba().")
    return model


def predict_probability(model, x2d: np.ndarray) -> np.ndarray:
    """Calcula probabilidad de la clase positiva."""
    if x2d.ndim != 2:
        raise ModelError(f"X debe ser 2D, recibido shape={x2d.shape}")
    if x2d.shape[0] == 0:
        return np.array([], dtype="float32")

    proba = model.predict_proba(x2d)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        idx1 = classes.index(1) if 1 in classes else -1
    else:
        idx1 = 1
    return proba[:, idx1].astype("float32")


def summarize_prediction(valid_mask: np.ndarray, features: list[str]) -> dict:
    """Resumen básico del proceso de inferencia."""
    total = int(valid_mask.size)
    valid = int(valid_mask.sum())
    return {
        "pixels_totales": total,
        "pixels_validos": valid,
        "pct_validos": (100.0 * valid / total) if total > 0 else 0.0,
        "n_features": len(features),
        "features": list(features),
    }