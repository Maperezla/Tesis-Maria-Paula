from pathlib import Path
import logging
import pandas as pd
import rasterio
import numpy as np

from .io_points import load_points, reproject_points_to_raster_crs
from .io_metadata import load_metadata
from .utils_raster import list_tifs
from .temporal_matching import build_temporal_matches
from .feature_engineering import (
    extract_base_bands_for_image,
    compute_derived_features,
    build_long_records_for_pair,
)
from .stats_plots import (
    compute_band_stats,
    pivot_band_time,
    plot_multiline,
    choose_sample_points,
    pivot_band_time_subset,
    plot_multiline_subset,
)


def run_pipeline(cfg, logger: logging.Logger) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_5_dir = cfg.out_dir / "plots_5points"
    plots_5_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_points(cfg.layer_path)
    meta = load_metadata(cfg.metadata_xlsx)
    tif_paths = list_tifs(cfg.raster_dir)

    logger.info(f"Puntos cargados: {len(gdf)}")
    logger.info(f"TIFF encontrados: {len(tif_paths)}")

    df_match = build_temporal_matches(tif_paths, meta)
    logger.info(f"Emparejamientos construidos: {len(df_match)}")

    with rasterio.open(df_match.iloc[0]["raster_path_actual"]) as src0:
        raster_crs = src0.crs

    gdf_r = reproject_points_to_raster_crs(gdf, raster_crs)
    xy = [(geom.x, geom.y) for geom in gdf_r.geometry]
    point_ids = gdf_r["point_id"].values

    records = []

    for _, row in df_match.iterrows():
        image_name_actual = row["image_name_actual"]
        raster_path_actual = row["raster_path_actual"]
        image_name_pre = row["image_name_pre"]

        raster_pre_candidates = [p for p in tif_paths if p.stem == image_name_pre]

        base_actual = extract_base_bands_for_image(raster_path_actual, xy)

        if len(raster_pre_candidates) == 0:
            logger.warning(f"Sin raster pre para {image_name_actual} -> {image_name_pre}")

            nan_arr = np.full_like(base_actual["VV"], np.nan, dtype="float64")
            derived = {
                "VV_VH_Ratio": compute_derived_features(
                    vv_actual_db=base_actual["VV"],
                    vh_actual_db=base_actual["VH"],
                    vv_pre_db=base_actual["VV"],   # dummy, no se usará
                    vh_pre_db=base_actual["VH"],   # dummy, no se usará
                    eps=cfg.epsilon
                )["VV_VH_Ratio"],
                "VV_pre_post_ratio": nan_arr,
                "VH_pre_post_ratio": nan_arr,
                "VH_VV_ratio_change": nan_arr
            }
        else:
            raster_path_pre = str(raster_pre_candidates[0])
            base_pre = extract_base_bands_for_image(raster_path_pre, xy)
            
            derived = compute_derived_features(
                vv_actual_db=base_actual["VV"],
                vh_actual_db=base_actual["VH"],
                vv_pre_db=base_pre["VV"],
                vh_pre_db=base_pre["VH"],
                eps=cfg.epsilon
            )

        records.extend(
            build_long_records_for_pair(
                point_ids=point_ids,
                image_name_actual=image_name_actual,
                base_actual=base_actual,
                derived=derived
            )
        )
        
    df_long = pd.DataFrame.from_records(records)

    csv_long = cfg.out_dir / "extractions_long.csv"
    df_long.to_csv(csv_long, index=False, encoding="utf-8")

    bands_time = ["VV", "VH", "angle", "VV_VH_Ratio",
                  "VV_pre_post_ratio", "VH_pre_post_ratio", "VH_VV_ratio_change"]
    order_images = df_match["image_name_actual"].tolist()

    df_stats = compute_band_stats(df_long, bands_time)
    stats_path = cfg.out_dir / "descriptive_stats_by_band.csv"
    df_stats.to_csv(stats_path, index=False, encoding="utf-8")

    for band in bands_time:
        pv = pivot_band_time(df_long, band, order_images)
        plot_multiline(pv, band, plots_dir / f"plot_{band}.png")

    selected_points = choose_sample_points(
        df_long=df_long,
        order_images=order_images,
        bands_time=bands_time,
        n_sample=cfg.n_sample,
        min_valid_frac=cfg.min_valid_frac,
        random_state=cfg.random_state
    )

    df_5 = df_long[df_long["point_id"].isin(selected_points)].copy()
    csv_5 = cfg.out_dir / "extractions_5points.csv"
    df_5.to_csv(csv_5, index=False, encoding="utf-8")

    for band in bands_time:
        pv = pivot_band_time_subset(df_long, band, selected_points, order_images)
        plot_multiline_subset(pv, band, plots_5_dir / f"plot_5pts_{band}.png")

    logger.info(f"CSV largo: {csv_long}")
    logger.info(f"Stats: {stats_path}")
    logger.info(f"Puntos seleccionados: {selected_points}")

    return {
        "df_long": df_long,
        "df_stats": df_stats,
        "df_match": df_match,
        "selected_points": selected_points,
        "paths": {
            "csv_long": str(csv_long),
            "stats": str(stats_path),
            "plots": str(plots_dir),
            "plots_5": str(plots_5_dir),
            "csv_5": str(csv_5),
        }
    }