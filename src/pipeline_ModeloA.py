from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from .features_ModeloA import apply_threshold, build_feature_cube, reconstruct_from_valid_mask
from .inference_ModeloA import load_model, predict_probability, summarize_prediction
from .io_rasters_ModeloA import (
    RasterAlignmentError,
    build_windows,
    create_output_raster,
    read_l8_stack,
    read_l8_window,
    read_s1_stack,
    read_s1_window,
    validate_raster_alignment,
    write_geotiff,
    write_window,
)
from .validation_ModeloA import validate_presence_absence, validate_presence_only


S1_FEATURE_NAMES = ["VV", "VH", "angle"]


class PipelineError(Exception):
    """Error general del pipeline."""


def run_pipeline(cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Ejecuta el pipeline completo según la configuración."""
    paths = cfg["paths"]
    outputs = cfg["outputs"]
    threshold = float(cfg["threshold"])
    features = cfg["features"]
    l8_band_map = cfg.get(
        "l8_band_map",
        {
            "SR_B4": "RED",
            "SR_B5": "NIR",
            "SR_B6": "SWIR_1",
            "SR_B7": "SWIR_2",
            "NDVI": "NDVI",
            "EVI": "EVI",
            "NBR": "NBR",
            "IMG_COUNT": "IMG_COUNT",
        },
    )
    s1_band_idx = cfg.get("s1_band_idx", {"VV": 1, "VH": 2, "angle": 3})

    logger.info("Iniciando pipeline de inferencia RF")
    logger.info("Modelo: %s", paths["model"])
    logger.info("L8: %s", paths["l8"])
    logger.info("S1: %s", paths["s1"])

    model = load_model(paths["model"])

    align = validate_raster_alignment(paths["l8"], paths["s1"])
    logger.info("Alineación raster: %s", align)
    if not all([align["same_width"], align["same_height"], align["same_crs"], align["same_transform"]]):
        raise RasterAlignmentError("L8 y S1 no están alineados. Reproyecta/ajusta antes de inferir.")

    use_windows = bool(cfg["window"].get("use_windows", False))
    if use_windows:
        result = _run_pipeline_windows(cfg, model, logger, l8_band_map, s1_band_idx, threshold, features)
    else:
        result = _run_pipeline_in_memory(cfg, model, logger, l8_band_map, s1_band_idx, threshold, features)

    validation_cfg = cfg.get("validation", {})
    if validation_cfg.get("enabled", False):
        logger.info("Validación activada")
        val_result = _run_validation(cfg, threshold, logger)
        result["validation"] = val_result
    else:
        logger.info("Validación desactivada")
        result["validation"] = None

    report_path = outputs.get("report_json")
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=_json_default)
        logger.info("Reporte JSON guardado en: %s", report_path)

    logger.info("Pipeline finalizado correctamente")
    return result


def _run_pipeline_in_memory(cfg, model, logger, l8_band_map, s1_band_idx, threshold, features):
    """Ejecuta el pipeline leyendo todo el raster en memoria."""
    paths = cfg["paths"]
    outputs = cfg["outputs"]

    try:
        l8_cube, meta_ref, l8_names = read_l8_stack(paths["l8"], l8_band_map)
    except RasterioIOError as e:
        raise PipelineError(f"No se pudo abrir L8: {e}") from e

    try:
        s1_cube = read_s1_stack(paths["s1"], s1_band_idx, ref_shape=(meta_ref["height"], meta_ref["width"]), ref_meta=meta_ref)
    except RasterioIOError as e:
        raise PipelineError(f"No se pudo abrir S1: {e}") from e

    x2d, valid_mask = build_feature_cube(l8_cube, l8_names, s1_cube, S1_FEATURE_NAMES, features)
    summary = summarize_prediction(valid_mask, features)
    logger.info("Resumen inferencia: %s", summary)

    p_valid = predict_probability(model, x2d)
    prob = reconstruct_from_valid_mask(p_valid, valid_mask, fill_value=np.nan, dtype="float32")
    burn_bin = apply_threshold(prob, threshold, valid_mask=valid_mask, nodata_value=255)

    write_geotiff(outputs["prob"], prob, meta_ref, nodata_val=np.nan, dtype="float32")
    write_geotiff(outputs["bin"], burn_bin, meta_ref, nodata_val=255, dtype="uint8")
    logger.info("Rasters exportados: prob=%s | bin=%s", outputs["prob"], outputs["bin"])

    return {
        "mode": "in_memory",
        "summary": summary,
        "outputs": {
            "prob": outputs["prob"],
            "bin": outputs["bin"],
        },
    }


def _run_pipeline_windows(cfg, model, logger, l8_band_map, s1_band_idx, threshold, features):
    """Ejecuta el pipeline usando lectura y escritura por ventanas."""
    paths = cfg["paths"]
    outputs = cfg["outputs"]
    window_size = int(cfg["window"].get("size", 512))

    with rasterio.open(paths["l8"]) as src_l8, rasterio.open(paths["s1"]) as src_s1:
        meta_ref = {
            "crs": src_l8.crs,
            "transform": src_l8.transform,
            "width": src_l8.width,
            "height": src_l8.height,
        }
        if src_s1.crs != src_l8.crs or src_s1.transform != src_l8.transform or src_s1.width != src_l8.width or src_s1.height != src_l8.height:
            raise RasterAlignmentError("L8 y S1 no están alineados para procesamiento por ventanas.")

        create_output_raster(outputs["prob"], meta_ref, dtype="float32", nodata_val=np.nan)
        create_output_raster(outputs["bin"], meta_ref, dtype="uint8", nodata_val=255)

        total_valid = 0
        total_pixels = 0
        n_windows = 0

        for window in build_windows(src_l8.width, src_l8.height, window_size):
            l8_cube, l8_names = read_l8_window(src_l8, l8_band_map, window)
            s1_cube = read_s1_window(src_s1, s1_band_idx, window, ref_window_shape=(int(window.height), int(window.width)))

            x2d, valid_mask = build_feature_cube(l8_cube, l8_names, s1_cube, S1_FEATURE_NAMES, features)
            p_valid = predict_probability(model, x2d)
            prob = reconstruct_from_valid_mask(p_valid, valid_mask, fill_value=np.nan, dtype="float32")
            burn_bin = apply_threshold(prob, threshold, valid_mask=valid_mask, nodata_value=255)

            write_window(outputs["prob"], prob.astype("float32"), window)
            write_window(outputs["bin"], burn_bin.astype("uint8"), window)

            total_valid += int(valid_mask.sum())
            total_pixels += int(valid_mask.size)
            n_windows += 1

        summary = {
            "pixels_totales": total_pixels,
            "pixels_validos": total_valid,
            "pct_validos": (100.0 * total_valid / total_pixels) if total_pixels > 0 else 0.0,
            "n_features": len(features),
            "features": list(features),
            "n_windows": n_windows,
            "window_size": window_size,
        }
        logger.info("Resumen inferencia por ventanas: %s", summary)

    return {
        "mode": "windows",
        "summary": summary,
        "outputs": {
            "prob": outputs["prob"],
            "bin": outputs["bin"],
        },
    }


def _run_validation(cfg, threshold, logger):
    """Ejecuta validación opcional con presencia o presencia/ausencia."""
    paths = cfg["paths"]
    layers = cfg.get("layers", {})
    outputs = cfg["outputs"]
    validation_cfg = cfg.get("validation", {})

    fire_path = paths.get("fire_points")
    fire_layer = layers.get("fire_layer")
    abs_path = validation_cfg.get("abs_points")
    abs_layer = validation_cfg.get("abs_layer")

    if not fire_path:
        logger.warning("Validación activada, pero no se definió paths['fire_points'].")
        return {"warning": "Validación activada sin fire_points"}

    if abs_path:
        logger.info("Validación con presencia y ausencia")
        return validate_presence_absence(
            raster_prob_path=outputs["prob"],
            fire_points_path=fire_path,
            fire_layer=fire_layer,
            abs_points_path=abs_path,
            abs_layer=abs_layer,
            threshold=threshold,
        )

    logger.info("Validación solo con puntos de presencia")
    return validate_presence_only(
        raster_prob_path=outputs["prob"],
        fire_points_path=fire_path,
        fire_layer=fire_layer,
        threshold=threshold,
    )


def _json_default(obj):
    """Conversión segura para tipos no serializables por JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)