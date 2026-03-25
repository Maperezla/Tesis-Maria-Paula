from __future__ import annotations

from typing import Dict, Optional

import geopandas as gpd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .io_rasters_ModeloA import read_vector, sample_raster_at_points


class ValidationError(Exception):
    """Error de validación."""


def validate_presence_only(raster_prob_path: str, fire_points_path: str, fire_layer: str | None, threshold: float) -> dict:
    """Valida solo con puntos de presencia/incendio."""
    g_fire = read_vector(fire_points_path, layer=fire_layer)
    p_fire = sample_raster_at_points(raster_prob_path, g_fire)

    valid = np.isfinite(p_fire)
    if valid.sum() == 0:
        raise ValidationError("No hay muestras válidas sobre los puntos de incendio.")

    yhat_fire = (p_fire[valid] >= threshold).astype(int)
    return {
        "n_puntos_incendio": int(len(g_fire)),
        "n_muestras_validas": int(valid.sum()),
        "tp_rate_sobre_presencia": float(100.0 * yhat_fire.mean()),
        "prob_media_en_incendios": float(np.nanmean(p_fire)),
    }


def validate_presence_absence(
    raster_prob_path: str,
    fire_points_path: str,
    fire_layer: str | None,
    abs_points_path: str,
    abs_layer: str | None,
    threshold: float,
) -> dict:
    """Valida con presencia y ausencia."""
    g_fire = read_vector(fire_points_path, layer=fire_layer)
    g_abs = read_vector(abs_points_path, layer=abs_layer)

    p_fire = sample_raster_at_points(raster_prob_path, g_fire)
    p_abs = sample_raster_at_points(raster_prob_path, g_abs)

    valid_fire = np.isfinite(p_fire)
    valid_abs = np.isfinite(p_abs)
    if valid_fire.sum() == 0 or valid_abs.sum() == 0:
        raise ValidationError("No hay muestras válidas suficientes para presencia/ausencia.")

    yhat_fire = (p_fire[valid_fire] >= threshold).astype(int)
    yhat_abs = (p_abs[valid_abs] >= threshold).astype(int)

    y_true = np.concatenate([
        np.ones(int(valid_fire.sum()), dtype=int),
        np.zeros(int(valid_abs.sum()), dtype=int),
    ])
    y_pred = np.concatenate([yhat_fire, yhat_abs])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(
        y_true,
        y_pred,
        target_names=["absence(0)", "presence(1)"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "n_fire": int(len(g_fire)),
        "n_abs": int(len(g_abs)),
        "n_fire_valid": int(valid_fire.sum()),
        "n_abs_valid": int(valid_abs.sum()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }