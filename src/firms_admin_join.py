import os
import re
import unicodedata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd


@dataclass
class JoinSummary:
    n_points: int
    n_join_ok: int
    n_join_fail: int
    crs_out: object
    out_path: str


def setup_logger(log_path: str, name: str = "firms_admin_join") -> logging.Logger:
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


def next_version_path(out_path: str) -> str:
    """
    Versiona: archivo.shp -> archivo_v002.shp -> ...
    """
    if not os.path.exists(out_path):
        return out_path
    base, ext = os.path.splitext(out_path)
    v = 2
    while True:
        cand = f"{base}_v{v:03d}{ext}"
        if not os.path.exists(cand):
            return cand
        v += 1


def normalize_text(val) -> Optional[str]:
    """
    - MAYÚSCULAS
    - sin tildes
    - espacios normalizados
    """
    if pd.isna(val):
        return None
    s = str(val).upper().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ensure_columns(gdf: gpd.GeoDataFrame, required: Sequence[str], layer_name: str = "layer") -> None:
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Faltan columnas en {layer_name}: {missing}")


def load_layers(
    municipios_path: str,
    puntos_path: str,
    mun_cols: Sequence[str] = ("MpNombre", "Depto"),
    logger: Optional[logging.Logger] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    mun = gpd.read_file(municipios_path)
    pts = gpd.read_file(puntos_path)

    if mun.crs is None:
        raise ValueError("El shapefile de municipios no tiene CRS definido.")
    if pts.crs != mun.crs:
        if logger:
            logger.info(f"Reproyectando puntos {pts.crs} -> {mun.crs}")
        pts = pts.to_crs(mun.crs)

    ensure_columns(mun, mun_cols, layer_name="municipios")

    # Mantener solo lo necesario
    mun = mun[list(mun_cols) + ["geometry"]].copy()

    # Campos std
    mun["mpnombre_std"] = mun[mun_cols[0]].apply(normalize_text)
    mun["depto_std"] = mun[mun_cols[1]].apply(normalize_text)

    return mun, pts


def spatial_join_points_to_municipios(
    puntos: gpd.GeoDataFrame,
    municipios: gpd.GeoDataFrame,
    how: str = "left",
    predicate: str = "intersects",
) -> gpd.GeoDataFrame:
    gdf_join = gpd.sjoin(puntos, municipios, how=how, predicate=predicate)
    gdf_join["join_ok"] = gdf_join["mpnombre_std"].notna()
    return gdf_join


def write_shp_versioned(
    gdf: gpd.GeoDataFrame,
    out_path: str,
    version: bool = True,
    encoding: str = "utf-8",
    logger: Optional[logging.Logger] = None
) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final_path = next_version_path(out_path) if version else out_path
    gdf.to_file(final_path, driver="ESRI Shapefile", encoding=encoding)
    if logger:
        logger.info(f"Guardado: {final_path} | n={len(gdf)} | crs={gdf.crs}")
    return final_path


def run_admin_join(
    municipios_path: str,
    puntos_path: str,
    out_path: str,
    how: str = "left",
    predicate: str = "intersects",
    version_out: bool = True,
    logger: Optional[logging.Logger] = None
) -> JoinSummary:
    municipios, puntos = load_layers(municipios_path, puntos_path, logger=logger)
    gdf_join = spatial_join_points_to_municipios(puntos, municipios, how=how, predicate=predicate)

    final_path = write_shp_versioned(gdf_join, out_path, version=version_out, logger=logger)

    n_ok = int(gdf_join["join_ok"].sum())
    n_total = len(gdf_join)
    summary = JoinSummary(
        n_points=n_total,
        n_join_ok=n_ok,
        n_join_fail=n_total - n_ok,
        crs_out=gdf_join.crs,
        out_path=final_path
    )

    if logger:
        logger.info(f"Resumen: {summary}")

    return summary


def format_report(summary: JoinSummary) -> str:
    return (
        "\n--- REPORTE FINAL (JOIN MUNICIPIOS) ---\n"
        f"📌 Total puntos: {summary.n_points}\n"
        f"✅ Asignados (join_ok): {summary.n_join_ok}\n"
        f"❌ No asignados: {summary.n_join_fail}\n"
        f"🧭 CRS salida: {summary.crs_out}\n"
        f"💾 Archivo: {summary.out_path}\n"
    )