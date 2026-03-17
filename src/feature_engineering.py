from pathlib import Path
import numpy as np
import rasterio

from .utils_raster import get_band_index_by_name, sample_points, db_to_lin, lin_to_db


def extract_base_bands_for_image(raster_path: str, xy: list[tuple[float, float]]) -> dict[str, np.ndarray]:
    with rasterio.open(raster_path) as src:
        vv_i = get_band_index_by_name(src, "VV")
        vh_i = get_band_index_by_name(src, "VH")
        ang_i = get_band_index_by_name(src, "angle")

        vv_db = sample_points(src, xy, vv_i)
        vh_db = sample_points(src, xy, vh_i)
        ang = sample_points(src, xy, ang_i)

    return {
        "VV": vv_db,
        "VH": vh_db,
        "angle": ang
    }


def compute_derived_features(
    vv_actual_db: np.ndarray,
    vh_actual_db: np.ndarray,
    vv_pre_db: np.ndarray,
    vh_pre_db: np.ndarray,
    eps: float
) -> dict[str, np.ndarray]:
    vv_actual_lin = db_to_lin(vv_actual_db)
    vh_actual_lin = db_to_lin(vh_actual_db)
    vv_pre_lin = db_to_lin(vv_pre_db)
    vh_pre_lin = db_to_lin(vh_pre_db)

    vv_vh_ratio_lin = vv_actual_lin / np.maximum(vh_actual_lin, eps)
    vv_vh_ratio_db = lin_to_db(vv_vh_ratio_lin, eps)

    vv_pre_post_ratio_lin = vv_pre_lin / np.maximum(vv_actual_lin, eps)
    vv_pre_post_ratio_db = lin_to_db(vv_pre_post_ratio_lin, eps)

    vh_pre_post_ratio_lin = vh_pre_lin / np.maximum(vh_actual_lin, eps)
    vh_pre_post_ratio_db = lin_to_db(vh_pre_post_ratio_lin, eps)

    vh_vv_pre_lin = vh_pre_lin / np.maximum(vv_pre_lin, eps)
    vh_vv_actual_lin = vh_actual_lin / np.maximum(vv_actual_lin, eps)
    vh_vv_ratio_change_lin = vh_vv_pre_lin / np.maximum(vh_vv_actual_lin, eps)
    vh_vv_ratio_change_db = lin_to_db(vh_vv_ratio_change_lin, eps)

    return {
        "VV_VH_Ratio": vv_vh_ratio_db,
        "VV_pre_post_ratio": vv_pre_post_ratio_db,
        "VH_pre_post_ratio": vh_pre_post_ratio_db,
        "VH_VV_ratio_change": vh_vv_ratio_change_db
    }


def build_long_records_for_pair(
    point_ids,
    image_name_actual: str,
    base_actual: dict[str, np.ndarray],
    derived: dict[str, np.ndarray]
) -> list[dict]:
    records = []

    for band_name, arr in {
        "VV": base_actual["VV"],
        "VH": base_actual["VH"],
        "angle": base_actual["angle"],
        **derived
    }.items():
        for pid, v in zip(point_ids, arr):
            records.append({
                "point_id": pid,
                "image_name": image_name_actual,
                "band": band_name,
                "value": v
            })

    return records