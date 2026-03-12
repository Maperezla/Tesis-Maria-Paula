import logging
import pandas as pd
import geopandas as gpd

from .selection_fire_orbits import (
    infer_platform_from_sysindex,
    choose_same_date,
    choose_post_window,
    choose_pre_window,
)

def run_assignment(cfg, logger: logging.Logger):
    gdf = gpd.read_file(cfg.points_shp)
    meta = pd.read_excel(cfg.meta_xlsx)

    required_point_cols = {cfg.col_year, cfg.col_month, cfg.col_day}
    missing_points = required_point_cols - set(gdf.columns)
    if missing_points:
        raise ValueError(f"Missing columns in points shapefile: {missing_points}")

    required_meta_cols = {cfg.col_img_date, cfg.col_orbit, cfg.col_sysindex}
    missing_meta = required_meta_cols - set(meta.columns)
    if missing_meta:
        raise ValueError(f"Missing columns in metadata Excel: {missing_meta}")

    if cfg.col_slice not in meta.columns:
        meta[cfg.col_slice] = pd.NA

    gdf["fire_date"] = pd.to_datetime(
        dict(
            year=gdf[cfg.col_year].astype(int),
            month=gdf[cfg.col_month].astype(int),
            day=gdf[cfg.col_day].astype(int),
        ),
        errors="raise"
    )

    meta["img_date"] = pd.to_datetime(
        meta[cfg.col_img_date].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    meta[cfg.col_orbit] = pd.to_numeric(meta[cfg.col_orbit], errors="coerce")
    meta["platform"] = meta[cfg.col_sysindex].apply(infer_platform_from_sysindex)

    meta = meta.sort_values(["img_date", cfg.col_sysindex]).reset_index(drop=True)

    slice_rank = {s: i for i, s in enumerate(cfg.slice_preference)}

    out_cols = [
        "orb_same", "img_same", "idx_same", "slice_same",
        "orb_post30", "img_post30", "idx_post30", "slice_post30",
        "orb_pre30", "img_pre30", "idx_pre30", "slice_pre30",
        "dt_post_days", "dt_pre_days"
    ]
    for c in out_cols:
        gdf[c] = pd.NA

    for i, row in gdf.iterrows():
        fire_date = row["fire_date"]

        pick = choose_same_date(
            meta, fire_date, cfg.col_img_date, cfg.col_slice, cfg.col_sysindex,
            slice_rank, cfg.prefer_platform
        )
        if pick is not None:
            gdf.at[i, "orb_same"] = int(pick[cfg.col_orbit])
            gdf.at[i, "img_same"] = pick["img_date"]
            gdf.at[i, "idx_same"] = str(pick[cfg.col_sysindex])
            gdf.at[i, "slice_same"] = pick[cfg.col_slice]

        pick = choose_post_window(
            meta, fire_date, cfg.col_slice, cfg.col_sysindex, slice_rank,
            cfg.prefer_platform, cfg.post_min_days, cfg.post_max_days
        )
        if pick is not None:
            gdf.at[i, "orb_post30"] = int(pick[cfg.col_orbit])
            gdf.at[i, "img_post30"] = pick["img_date"]
            gdf.at[i, "idx_post30"] = str(pick[cfg.col_sysindex])
            gdf.at[i, "slice_post30"] = pick[cfg.col_slice]
            gdf.at[i, "dt_post_days"] = int((pick["img_date"] - fire_date).days)

        pick = choose_pre_window(
            meta, fire_date, cfg.col_slice, cfg.col_sysindex, slice_rank,
            cfg.prefer_platform, cfg.pre_min_days, cfg.pre_target_days
        )
        if pick is not None:
            gdf.at[i, "orb_pre30"] = int(pick[cfg.col_orbit])
            gdf.at[i, "img_pre30"] = pick["img_date"]
            gdf.at[i, "idx_pre30"] = str(pick[cfg.col_sysindex])
            gdf.at[i, "slice_pre30"] = pick[cfg.col_slice]
            gdf.at[i, "dt_pre_days"] = int((pick["img_date"] - fire_date).days)

    summary = build_summary(gdf)
    logger.info(f"Resumen asignación: {summary}")

    return {"gdf": gdf, "summary": summary}

def build_summary(gdf):
    n = len(gdf)

    def pct_notna(col):
        return round(100.0 * gdf[col].notna().sum() / n, 2) if n else 0.0

    return {
        "same_date_pct": pct_notna("orb_same"),
        "post_30_pct": pct_notna("orb_post30"),
        "pre_30_pct": pct_notna("orb_pre30"),
        "n_points": n
    }