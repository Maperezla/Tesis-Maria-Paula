import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd
from datetime import datetime


@dataclass
class MergeSummary:
    n_inputs: int
    n_features_in: int
    n_features_out: int
    n_dropped_dupes: int
    crs_out: object


def setup_logger(log_path: str, name: str = "firms_merge") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    abs_path = os.path.abspath(log_path)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == abs_path
               for h in logger.handlers):
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        fh = logging.FileHandler(abs_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

    return logger


def next_version_path(out_shp_path: str) -> str:
    if not os.path.exists(out_shp_path):
        return out_shp_path
    base, ext = os.path.splitext(out_shp_path)
    v = 2
    while True:
        cand = f"{base}_v{v:03d}{ext}"
        if not os.path.exists(cand):
            return cand
        v += 1


def parse_dd_mm_yyyy(val) -> object:
    """Normaliza a 'YYYY-MM-DD' si puede; si no, devuelve el valor original como string."""
    if pd.isna(val):
        return val
    s = str(val).strip()

    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue

    try:
        return pd.to_datetime(s, dayfirst=True, errors="raise").date().isoformat()
    except Exception:
        return s


def read_shps(paths: Sequence[str], logger: Optional[logging.Logger] = None) -> List[gpd.GeoDataFrame]:
    gdfs = []
    for i, p in enumerate(paths, start=1):
        gdf = gpd.read_file(p)
        if gdf.empty:
            raise ValueError(f"Shapefile vacío: {p}")
        gdfs.append(gdf)

        if logger:
            logger.info(f"Leído shapefile {i}: {p} | n={len(gdf)} | crs={gdf.crs}")

    return gdfs


def warn_if_not_epsg(gdfs: Sequence[gpd.GeoDataFrame], epsg: int = 4326, logger: Optional[logging.Logger] = None) -> None:
    for i, gdf in enumerate(gdfs, start=1):
        ok = (gdf.crs is not None and gdf.crs.to_epsg() == epsg)
        if not ok:
            msg = f"Aviso: shapefile {i} no está en EPSG:{epsg} (CRS={gdf.crs}). Se continuará igual (no reproyecta)."
            if logger:
                logger.warning(msg)
            else:
                print(f"⚠️ {msg}")


def validate_required_fields(gdf: gpd.GeoDataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise ValueError(f"Faltan campos requeridos: {missing}")


def normalize_acq_date(gdf: gpd.GeoDataFrame, src_col: str = "ACQ_DATE", out_col: str = "ACQ_DATE_NORM") -> gpd.GeoDataFrame:
    if src_col not in gdf.columns:
        raise ValueError(f"Falta el campo {src_col}.")
    gdf[out_col] = gdf[src_col].apply(parse_dd_mm_yyyy)
    return gdf


def coerce_latlon_numeric(gdf: gpd.GeoDataFrame, lat_col: str = "LATITUDE", lon_col: str = "LONGITUDE",
                          logger: Optional[logging.Logger] = None) -> gpd.GeoDataFrame:
    # Convertir a numérico de forma segura: lo no convertible queda como NaN
    gdf[lat_col] = pd.to_numeric(gdf[lat_col], errors="coerce")
    gdf[lon_col] = pd.to_numeric(gdf[lon_col], errors="coerce")

    n_nan = int(gdf[[lat_col, lon_col]].isna().any(axis=1).sum())
    if n_nan > 0:
        msg = f"Hay {n_nan} filas con LAT/LON no numéricos (NaN tras coerción)."
        if logger:
            logger.warning(msg)
        else:
            print(f"⚠️ {msg}")

    return gdf


def concat_geodataframes(gdfs: Sequence[gpd.GeoDataFrame], crs_fallback=None) -> gpd.GeoDataFrame:
    df = pd.concat(gdfs, ignore_index=True)
    # CRS: usa el primero no-nulo; si todos son None, usa fallback
    crs = next((g.crs for g in gdfs if g.crs is not None), crs_fallback)
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs)


def deduplicate_prefer_instrument(
    gdf: gpd.GeoDataFrame,
    keys: Sequence[str],
    instrument_col: str = "INSTRUMENT",
    prefer_value: str = "MODIS"
) -> Tuple[gpd.GeoDataFrame, int]:
    """
    Deja 1 registro por keys. Si hay duplicados, conserva prefer_value (ej. MODIS).
    """
    # priority=0 para preferidos; 1 para los demás
    pref = gdf[instrument_col].astype(str).str.upper() == str(prefer_value).upper()
    gdf["_priority"] = (~pref).astype(int)

    before = len(gdf)
    gdf2 = (
        gdf.sort_values(by=["_priority"])
           .drop_duplicates(subset=list(keys), keep="first")
           .drop(columns=["_priority"])
    )
    dropped = before - len(gdf2)
    return gdf2, dropped


def write_shp(gdf: gpd.GeoDataFrame, out_path: str, version: bool = True,
              logger: Optional[logging.Logger] = None) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final_path = next_version_path(out_path) if version else out_path
    gdf.to_file(final_path, driver="ESRI Shapefile", encoding="utf-8")
    if logger:
        logger.info(f"Shapefile guardado: {final_path} | n={len(gdf)} | crs={gdf.crs}")
    return final_path


def merge_firms_shps(
    paths: Sequence[str],
    out_path: str,
    epsg_expected: int = 4326,
    required_cols: Sequence[str] = ("ACQ_DATE", "LATITUDE", "LONGITUDE", "INSTRUMENT"),
    date_src: str = "ACQ_DATE",
    date_out: str = "ACQ_DATE_NORM",
    dedup_keys: Sequence[str] = ("ACQ_DATE_NORM", "LATITUDE", "LONGITUDE"),
    prefer_instrument: str = "MODIS",
    version_out: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, MergeSummary]:
    gdfs = read_shps(paths, logger=logger)
    warn_if_not_epsg(gdfs, epsg=epsg_expected, logger=logger)

    g = concat_geodataframes(gdfs)

    validate_required_fields(g, required_cols)
    g = normalize_acq_date(g, src_col=date_src, out_col=date_out)
    g = coerce_latlon_numeric(g, logger=logger)

    before = len(g)
    g, dropped = deduplicate_prefer_instrument(
        g, keys=dedup_keys, instrument_col="INSTRUMENT", prefer_value=prefer_instrument
    )

    final_path = write_shp(g, out_path, version=version_out, logger=logger)

    summary = MergeSummary(
        n_inputs=len(paths),
        n_features_in=before,
        n_features_out=len(g),
        n_dropped_dupes=dropped,
        crs_out=g.crs
    )
    return final_path, summary