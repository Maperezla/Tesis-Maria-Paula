import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union


@dataclass
class SamplingSummary:
    n_requested: int
    n_generated: int
    buffer_m: float
    seed: int
    crs_target: str
    geom_type_allowed: str
    out_path: str


def setup_logger(log_path: str, name: str = "random_absence_points") -> logging.Logger:
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


def next_version_path_suffix_1(out_path: str) -> str:
    """Versionado simple: _1, _2, ..."""
    if not os.path.exists(out_path):
        return out_path
    base, ext = os.path.splitext(out_path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


def load_and_project(aoi_path: str, points_path: str, crs_target: str,
                     logger: Optional[logging.Logger] = None) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f"No existe AOI: {aoi_path}")
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"No existe puntos: {points_path}")

    aoi = gpd.read_file(aoi_path)
    pts = gpd.read_file(points_path)

    if aoi.empty:
        raise ValueError("AOI vacío.")
    if pts.empty:
        raise ValueError("Puntos FIRMS vacío; no se puede construir buffer.")

    if aoi.crs is None:
        raise ValueError("AOI sin CRS.")
    if pts.crs is None:
        raise ValueError("Puntos sin CRS.")

    if str(aoi.crs) != crs_target:
        if logger: logger.info(f"Reproyectando AOI {aoi.crs} -> {crs_target}")
        aoi = aoi.to_crs(crs_target)

    if str(pts.crs) != crs_target:
        if logger: logger.info(f"Reproyectando puntos {pts.crs} -> {crs_target}")
        pts = pts.to_crs(crs_target)

    return aoi, pts


def build_allowed_zone(aoi: gpd.GeoDataFrame, pts: gpd.GeoDataFrame, buffer_m: float):
    aoi_geom = unary_union(aoi.geometry)
    if aoi_geom.is_empty:
        raise ValueError("AOI unificado quedó vacío.")

    buffer_union = unary_union(pts.geometry.buffer(buffer_m))
    allowed = aoi_geom.difference(buffer_union)

    if allowed.is_empty:
        raise ValueError("Zona permitida vacía (AOI - buffer). Reduce buffer o revisa AOI/pts.")

    return allowed


def generate_random_points_in_geom(
    geom,
    n_points: int,
    seed: int = 42,
    max_attempts_factor: int = 500,
) -> List[Point]:
    """
    Muestreo por rechazo dentro del bbox.
    Usa contains() (excluye borde) para evitar caer exactamente en fronteras.
    """
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = geom.bounds

    pts: List[Point] = []
    attempts = 0
    max_attempts = n_points * max_attempts_factor

    while len(pts) < n_points and attempts < max_attempts:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if geom.contains(p):
            pts.append(p)
        attempts += 1

    if len(pts) < n_points:
        raise RuntimeError(
            f"Solo se generaron {len(pts)}/{n_points} puntos tras {attempts} intentos. "
            f"Zona permitida muy pequeña o fragmentada (o max_attempts_factor bajo)."
        )

    return pts


def write_points_shp(points: List[Point], out_path: str, crs_target: str,
                     version: bool = True, logger: Optional[logging.Logger] = None) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final = next_version_path_suffix_1(out_path) if version else out_path

    gdf = gpd.GeoDataFrame({"id": range(1, len(points) + 1)}, geometry=points, crs=crs_target)
    gdf.to_file(final, driver="ESRI Shapefile", encoding="utf-8")

    if logger:
        logger.info(f"Guardado: {final} | n={len(gdf)} | crs={gdf.crs}")

    return final


def run_random_absence_sampling(
    aoi_path: str,
    points_path: str,
    out_path: str,
    crs_target: str = "EPSG:9377",
    buffer_m: float = 1000.0,
    n_points: int = 961,
    seed: int = 42,
    max_attempts_factor: int = 500,
    logger: Optional[logging.Logger] = None
) -> SamplingSummary:
    aoi, pts = load_and_project(aoi_path, points_path, crs_target, logger=logger)
    allowed = build_allowed_zone(aoi, pts, buffer_m=buffer_m)

    if logger:
        logger.info(f"Allowed geom: type={allowed.geom_type} | bounds={allowed.bounds}")

    rand_points = generate_random_points_in_geom(
        allowed, n_points=n_points, seed=seed, max_attempts_factor=max_attempts_factor
    )

    final_out = write_points_shp(rand_points, out_path, crs_target, version=True, logger=logger)

    return SamplingSummary(
        n_requested=n_points,
        n_generated=len(rand_points),
        buffer_m=buffer_m,
        seed=seed,
        crs_target=crs_target,
        geom_type_allowed=allowed.geom_type,
        out_path=final_out
    )


def format_report(s: SamplingSummary) -> str:
    return (
        "\n--- REPORTE FINAL (PUNTOS NO CALIENTES) ---\n"
        f"🎯 Solicitados: {s.n_requested}\n"
        f"✅ Generados: {s.n_generated}\n"
        f"📏 Buffer (m): {s.buffer_m}\n"
        f"🌐 CRS: {s.crs_target}\n"
        f"🧩 Geom zona permitida: {s.geom_type_allowed}\n"
        f"🔁 Seed: {s.seed}\n"
        f"💾 Salida: {s.out_path}\n"
    )