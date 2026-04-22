import ee

from .Copy2_collection import (
    build_collection,
    build_collection_by_pass,
    build_collection_by_pass_and_relative_orbit,
)
from .temporal_mosaics import build_temporal_mosaics
from .radiometry import border_noise_mask, gamma0_db, terrain_flattening
from .speckle import refined_lee_spatial
from .Copy2_indices import add_ratio_db, build_prepost_pairs_and_indices


def _copy_all_properties(src_img: ee.Image, out_img: ee.Image) -> ee.Image:
    src_img = ee.Image(src_img)
    out_img = ee.Image(out_img)
    return out_img.copyProperties(src_img, src_img.propertyNames())


def build_pipeline_base(aoi: ee.Geometry, cfg: dict) -> ee.ImageCollection:
    year = int(cfg["year"])
    orbit = str(cfg["orbit"])
    eps = float(cfg.get("eps", 1e-10))
    kernel_size = int(cfg.get("kernel_size", 7))
    dem_id = str(cfg["dem_id"])
    window_days = int(cfg.get("window_days", 2))

    start_date = str(cfg.get("start_date", f"{year}-01-01"))
    end_date = str(cfg.get("end_date", f"{year + 1}-01-01"))
    mosaic_reducer = str(cfg.get("mosaic_reducer", "median"))

    dem = ee.Image(dem_id)
    relative_orbit = cfg.get("relative_orbit", None)

    if relative_orbit is None:
        col_raw = build_collection_by_pass(
            aoi=aoi,
            year=year,
            orbit_pass=orbit,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        col_raw = build_collection_by_pass_and_relative_orbit(
            aoi=aoi,
            year=year,
            orbit_pass=orbit,
            relative_orbit=int(relative_orbit),
            start_date=start_date,
            end_date=end_date,
        )

    col_mosaic = build_temporal_mosaics(
        col=col_raw,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        reducer=mosaic_reducer,
    )

    col_proc = col_mosaic.map(lambda img: _copy_all_properties(img, border_noise_mask(img)))
    col_proc = col_proc.map(lambda img: _copy_all_properties(img, gamma0_db(img)))
    col_proc = col_proc.map(lambda img: _copy_all_properties(img, terrain_flattening(img, dem=dem)))
    col_proc = col_proc.map(lambda img: _copy_all_properties(img, refined_lee_spatial(img, kernel_size=kernel_size)))
    col_proc = col_proc.map(lambda img: _copy_all_properties(img, add_ratio_db(img, eps=eps)))

    return col_proc.sort("system:time_start")


def build_pipeline_with_prepost(aoi: ee.Geometry, cfg: dict) -> ee.ImageCollection:
    eps = float(cfg.get("eps", 1e-10))
    pre_max_gap_days = int(cfg.get("pre_max_gap_days", 60))
    coverage_scale = int(cfg.get("coverage_scale", 30))
    min_overlap_pixels = float(cfg.get("min_overlap_pixels", 1000))
    min_overlap_fraction_of_post = float(cfg.get("min_overlap_fraction_of_post", 0.20))

    col_base = build_pipeline_base(aoi, cfg)

    col_final = build_prepost_pairs_and_indices(
        col=col_base,
        aoi=aoi,
        eps=eps,
        max_gap_days=pre_max_gap_days,
        coverage_scale=coverage_scale,
        min_overlap_pixels=min_overlap_pixels,
        min_overlap_fraction_of_post=min_overlap_fraction_of_post,
    )

    return col_final.sort("system:time_start")


def build_pipeline(aoi: ee.Geometry, cfg: dict) -> ee.ImageCollection:
    """
    Se mantiene apuntando al flujo base por diseño.
    """
    return build_pipeline_base(aoi, cfg)