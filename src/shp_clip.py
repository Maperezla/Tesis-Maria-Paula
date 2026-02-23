import os
import logging
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple

import geopandas as gpd
from tqdm import tqdm


@dataclass
class ClipSummary:
    total: int = 0
    clipped_ok: int = 0
    no_intersection_or_empty: int = 0
    err: int = 0


def setup_logger(log_path: str, name: str = "shp_clip") -> logging.Logger:
    """
    Logger idempotente para Jupyter (evita handlers duplicados al re-ejecutar).
    """
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
    """
    Si existe out_shp_path, versiona: _v002, _v003, ...
    """
    if not os.path.exists(out_shp_path):
        return out_shp_path

    base, ext = os.path.splitext(out_shp_path)
    v = 2
    while True:
        candidate = f"{base}_v{v:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        v += 1


def load_aoi(aoi_path: str) -> Tuple[gpd.GeoDataFrame, object, object]:
    """
    Carga AOI, valida, devuelve:
    - aoi_gdf
    - aoi_union_geom (para intersects)
    - aoi_crs
    """
    aoi = gpd.read_file(aoi_path)
    if aoi.empty or len(aoi) < 1:
        raise ValueError("El shapefile AOI está vacío o no contiene geometrías.")
    if aoi.crs is None:
        raise ValueError("El AOI no tiene CRS definido.")

    # Union robusta (GeoPandas >= 0.13): union_all()
    aoi_union = aoi.geometry.union_all()
    return aoi, aoi_union, aoi.crs


def iter_shps(root: str, endswith: str, exclude_contains: str) -> Iterator[str]:
    """
    Itera paths .shp en subcarpetas con filtros simples.
    """
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(endswith.lower()) and exclude_contains not in f:
                yield os.path.join(dirpath, f)


def clip_one_shp(
    in_shp_path: str,
    aoi_gdf: gpd.GeoDataFrame,
    aoi_union,
    aoi_crs,
    out_suffix: str = "_recorte",
    version: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, Optional[str]]:
    """
    Recorta un shapefile al AOI.
    Retorna (status, out_path):
      - status: "OK", "NO_INTERSECTION", "CLIP_EMPTY", "ERROR"
    """
    try:
        gdf = gpd.read_file(in_shp_path)
        if gdf.empty:
            return "CLIP_EMPTY", None  # vacío desde entrada

        # Armonizar CRS
        if gdf.crs != aoi_crs:
            if logger:
                logger.warning(f"Reproyectando a CRS del AOI: {in_shp_path}")
            gdf = gdf.to_crs(aoi_crs)

        # Intersección rápida (si no hay intersección, evitar clip)
        if not gdf.intersects(aoi_union).any():
            if logger:
                logger.warning(f"No hay intersección: {in_shp_path}")
            return "NO_INTERSECTION", None

        gdf_clip = gpd.clip(gdf, aoi_gdf)

        if gdf_clip.empty:
            if logger:
                logger.warning(f"Recorte vacío: {in_shp_path}")
            return "CLIP_EMPTY", None

        folder = os.path.dirname(in_shp_path)
        fname = os.path.splitext(os.path.basename(in_shp_path))[0]
        out_path = os.path.join(folder, f"{fname}{out_suffix}.shp")
        if version:
            out_path = next_version_path(out_path)

        gdf_clip.to_file(out_path)

        if logger:
            logger.info(f"Recorte exitoso: {out_path}")

        return "OK", out_path

    except Exception as e:
        if logger:
            logger.error(f"No se pudo recortar: {in_shp_path} - Error: {e}")
        return "ERROR", None


def batch_clip(
    root_folder: str,
    aoi_path: str,
    input_endswith: str = "_wgs84.shp",
    exclude_contains: str = "_recorte",
    out_suffix: str = "_recorte",
    version: bool = True,
    logger: Optional[logging.Logger] = None
) -> ClipSummary:
    """
    Ejecuta clip en lote y devuelve resumen.
    """
    aoi_gdf, aoi_union, aoi_crs = load_aoi(aoi_path)

    summary = ClipSummary()
    shp_paths = list(iter_shps(root_folder, endswith=input_endswith, exclude_contains=exclude_contains))

    for shp in tqdm(shp_paths, desc="Recortando shapefiles"):
        summary.total += 1
        status, out_path = clip_one_shp(
            in_shp_path=shp,
            aoi_gdf=aoi_gdf,
            aoi_union=aoi_union,
            aoi_crs=aoi_crs,
            out_suffix=out_suffix,
            version=version,
            logger=logger
        )

        if status == "OK":
            summary.clipped_ok += 1
            print(f"✅ {out_path}")
        elif status in ("NO_INTERSECTION", "CLIP_EMPTY"):
            summary.no_intersection_or_empty += 1
            print(f"⚠️ {status}: {shp}")
        else:
            summary.err += 1
            print(f"❌ ERROR: {shp}")

    return summary


def format_report(summary: ClipSummary) -> str:
    return (
        f"\n--- REPORTE FINAL ---\n"
        f"📊 Total shapefiles procesados: {summary.total}\n"
        f"✅ Recortados exitosamente: {summary.clipped_ok}\n"
        f"⚠️ Sin intersección o vacíos: {summary.no_intersection_or_empty}\n"
        f"❌ Con error: {summary.err}\n"
    )