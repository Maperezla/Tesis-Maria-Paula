from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class FeatureError(Exception):
    """Error en construcción de features."""


def build_feature_cube(
    l8_cube: np.ndarray,
    l8_names: list[str],
    s1_cube: np.ndarray,
    s1_names: list[str],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Une cubos L8 y S1 y construye matriz 2D para inferencia.

    Returns
    -------
    X2d : np.ndarray
        (N_valid, n_features)
    valid_mask : np.ndarray
        (H, W) bool
    """
    h, w, _ = l8_cube.shape
    if s1_cube.shape[:2] != (h, w):
        raise FeatureError("S1 y L8 no coinciden en H,W.")

    d: Dict[str, np.ndarray] = {}
    for i, name in enumerate(l8_names):
        d[name] = l8_cube[:, :, i]
    for i, name in enumerate(s1_names):
        d[name] = s1_cube[:, :, i]

    missing = [name for name in features if name not in d]
    if missing:
        raise FeatureError(f"Faltan features requeridas para el modelo: {missing}")

    stack = np.stack([d[name] for name in features], axis=-1).astype("float32")
    valid_mask = np.isfinite(stack).all(axis=-1)
    x2d = stack[valid_mask, :]
    return x2d, valid_mask


def reconstruct_from_valid_mask(values: np.ndarray, valid_mask: np.ndarray, fill_value=np.nan, dtype="float32") -> np.ndarray:
    """Reconstruye un raster 2D a partir de valores válidos y su máscara."""
    out = np.full(valid_mask.shape, fill_value, dtype=dtype)
    out[valid_mask] = values.astype(dtype)
    return out


def apply_threshold(prob: np.ndarray, threshold: float, valid_mask: np.ndarray | None = None, nodata_value: int = 255) -> np.ndarray:
    """Binariza una probabilidad usando umbral."""
    out = np.zeros(prob.shape, dtype="uint8")
    finite = np.isfinite(prob)
    out[finite] = (prob[finite] >= threshold).astype("uint8")
    if valid_mask is not None:
        out[~valid_mask] = nodata_value
    return out