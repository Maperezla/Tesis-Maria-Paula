import os
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import geopandas as gpd
from tqdm import tqdm


@dataclass
class ReprojectSummary:
    total: int = 0
    ok: int = 0
    err: int = 0


def setup_logger(log_path: str, logger_name: str = "shp_reproject") -> logging.Logger:
    """
    Crea un logger idempotente (evita duplicar handlers si re-ejecutas en Jupyter).
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Evita handlers duplicados en re-ejecuciones del notebook
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
               for h in logger.handlers):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def next_version_path(out_shp_path: str) -> str:
    """
    Si out_shp_path existe, versiona: _v002, _v003, ...
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


def reproject_one_shp(
    in_shp_path: str,
    target_crs: str,
    suffix: str = "_wgs84",
    version: bool = True,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Lee un shapefile, valida CRS, reproyecta y guarda como shapefile.
    Devuelve la ruta del shapefile escrito.
    """
    gdf = gpd.read_file(in_shp_path)

    if gdf.empty:
        raise ValueError("El shapefile está vacío.")

    if gdf.crs is None:
        raise ValueError("El shapefile no tiene CRS definido.")

    gdf_out = gdf.to_crs(target_crs)

    folder = os.path.dirname(in_shp_path)
    fname = os.path.splitext(os.path.basename(in_shp_path))[0]
    out_shp_path = os.path.join(folder, f"{fname}{suffix}.shp")

    if version:
        out_shp_path = next_version_path(out_shp_path)

    gdf_out.to_file(out_shp_path)  # driver ESRI Shapefile por defecto

    if logger:
        logger.info(f"Reproyección exitosa: {out_shp_path}")

    return out_shp_path


def batch_reproject_shps(
    root_folder: str,
    target_crs: str,
    suffix: str = "_wgs84",
    exclude_if_contains: str = "_wgs84",
    version: bool = True,
    logger: Optional[logging.Logger] = None
) -> ReprojectSummary:
    """
    Recorre subcarpetas, reproyecta shapefiles y acumula métricas.
    """
    summary = ReprojectSummary()

    for dirpath, _, filenames in os.walk(root_folder):
        shp_files = [f for f in filenames if f.lower().endswith(".shp") and exclude_if_contains not in f]
        if not shp_files:
            continue

        for f in tqdm(shp_files, desc=f"Procesando en {dirpath}", leave=False):
            in_path = os.path.join(dirpath, f)
            summary.total += 1
            try:
                out_path = reproject_one_shp(
                    in_shp_path=in_path,
                    target_crs=target_crs,
                    suffix=suffix,
                    version=version,
                    logger=logger
                )
                summary.ok += 1
                print(f"✅ {out_path}")
            except Exception as e:
                summary.err += 1
                if logger:
                    logger.error(f"No se pudo reproyectar: {in_path} - Error: {e}")
                print(f"❌ Error en: {in_path}. Ver log.")

    return summary

def format_report(summary: ReprojectSummary, label: str = "") -> str:
    if label:
        header = f"\n--- REPORTE FINAL {label} ---\n"
    else:
        header = "\n--- REPORTE FINAL ---\n"

    return (
        f"{header}"
        + f"📊 Total shapefiles procesados: {summary.total}\n"
        + f"✅ Reproyectados correctamente: {summary.ok}\n"
        + f"❌ Con error: {summary.err}\n"
    )