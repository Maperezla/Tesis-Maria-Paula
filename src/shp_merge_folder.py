import os
import glob
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd


@dataclass
class MergeFolderSummary:
    folder: str
    n_found: int
    n_read_ok: int
    n_skipped_empty: int
    n_errors: int
    n_rows_out: int
    crs_base: object
    geom_types: List[str]
    columns_out: List[str]
    out_path: str


def setup_logger(log_path: str, name: str = "shp_merge_folder") -> logging.Logger:
    """
    Logger idempotente para evitar handlers duplicados en Jupyter.
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


def list_shapefiles(folder: str, pattern: str = "*.shp") -> List[str]:
    return sorted(glob.glob(os.path.join(folder, pattern)))


def normalize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Normaliza columnas a lower + snake_case y resuelve colisiones de nombres.
    """
    # 1) proponer nombres
    proposed = [c.lower().strip().replace(" ", "_") for c in gdf.columns]

    # 2) resolver duplicados (colisiones) agregando sufijo _2, _3...
    counts: Dict[str, int] = {}
    final_names = []
    for name in proposed:
        if name not in counts:
            counts[name] = 1
            final_names.append(name)
        else:
            counts[name] += 1
            final_names.append(f"{name}_{counts[name]}")

    mapping = {old: new for old, new in zip(gdf.columns, final_names)}
    return gdf.rename(columns=mapping)


def next_version_path(out_shp_path: str) -> str:
    """
    Versiona: archivo.shp -> archivo_v002.shp -> archivo_v003.shp ...
    """
    if not os.path.exists(out_shp_path):
        return out_shp_path

    base, ext = os.path.splitext(out_shp_path)
    v = 2
    while True:
        cand = f"{base}_v{v:03d}{ext}"
        if not os.path.exists(cand):
            return cand
        v += 1


def read_and_standardize(
    shp_paths: Sequence[str],
    normalize_cols: bool = True,
    reproject_to_base: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[gpd.GeoDataFrame], object, List[str], Dict[str, int]]:
    """
    Lee shapefiles, omite vacíos, establece CRS base (primer no vacío),
    reproyecta si es necesario y normaliza columnas.
    Retorna:
    - gdfs válidos
    - crs_base
    - geom_types globales
    - contadores (ok, empty, err)
    """
    gdfs: List[gpd.GeoDataFrame] = []
    crs_base = None
    geom_types_set = set()
    counters = {"ok": 0, "empty": 0, "err": 0}

    for i, path in enumerate(shp_paths, start=1):
        try:
            if logger:
                logger.info(f"Leyendo ({i}/{len(shp_paths)}): {path}")

            gdf = gpd.read_file(path)

            if gdf.empty:
                counters["empty"] += 1
                if logger:
                    logger.warning(f"Omitido (vacío): {path}")
                continue

            # CRS base: primer no vacío
            if crs_base is None:
                crs_base = gdf.crs
                if logger:
                    logger.info(f"CRS base establecido: {crs_base}")
            else:
                if reproject_to_base and (gdf.crs != crs_base) and (gdf.crs is not None) and (crs_base is not None):
                    src = gdf.crs
                    gdf = gdf.to_crs(crs_base)
                    if logger:
                        logger.info(f"Reproyectado: {src} -> {crs_base} | {path}")

            if normalize_cols:
                gdf = normalize_columns(gdf)

            # Geom types
            for t in gdf.geometry.geom_type.unique():
                geom_types_set.add(str(t))

            gdfs.append(gdf)
            counters["ok"] += 1

        except Exception as e:
            counters["err"] += 1
            if logger:
                logger.error(f"Error leyendo/procesando: {path} | {e}")

    if not gdfs:
        raise ValueError("No hay GeoDataFrames válidos para unir (todos vacíos o con error).")

    return gdfs, crs_base, sorted(list(geom_types_set)), counters


def merge_gdfs(gdfs: Sequence[gpd.GeoDataFrame], crs_base) -> gpd.GeoDataFrame:
    merged = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=crs_base)


def write_shapefile_versioned(
    gdf: gpd.GeoDataFrame,
    out_path: str,
    version: bool = True,
    encoding: str = "utf-8",
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Escribe SHP. Si version=True, no sobrescribe: crea _v002, _v003...
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final_path = next_version_path(out_path) if version else out_path

    # Nota: Shapefile limita nombres de columnas a 10 caracteres (se truncarán).
    gdf.to_file(final_path, driver="ESRI Shapefile", encoding=encoding)

    if logger:
        logger.info(f"Guardado: {final_path} | n={len(gdf)} | crs={gdf.crs}")

    return final_path


def merge_shapefiles_in_folder(
    folder: str,
    out_name: str,
    pattern: str = "*.shp",
    normalize_cols: bool = True,
    reproject_to_base: bool = True,
    version_out: bool = True,
    logger: Optional[logging.Logger] = None
) -> MergeFolderSummary:
    shp_paths = list_shapefiles(folder, pattern=pattern)
    if not shp_paths:
        raise FileNotFoundError("No se encontraron archivos .shp en la carpeta indicada.")

    gdfs, crs_base, geom_types, counters = read_and_standardize(
        shp_paths,
        normalize_cols=normalize_cols,
        reproject_to_base=reproject_to_base,
        logger=logger
    )

    gdf_out = merge_gdfs(gdfs, crs_base=crs_base)

    out_path = os.path.join(folder, out_name)
    final_out = write_shapefile_versioned(
        gdf_out,
        out_path=out_path,
        version=version_out,
        logger=logger
    )

    summary = MergeFolderSummary(
        folder=folder,
        n_found=len(shp_paths),
        n_read_ok=counters["ok"],
        n_skipped_empty=counters["empty"],
        n_errors=counters["err"],
        n_rows_out=len(gdf_out),
        crs_base=gdf_out.crs,
        geom_types=geom_types,
        columns_out=list(gdf_out.columns),
        out_path=final_out
    )

    if logger:
        logger.info(f"SUMMARY: {asdict(summary)}")

    return summary


def format_report(summary: MergeFolderSummary) -> str:
    return (
        "\n--- REPORTE FINAL ---\n"
        f"📁 Carpeta: {summary.folder}\n"
        f"🔎 Shapefiles encontrados: {summary.n_found}\n"
        f"✅ Leídos y usados: {summary.n_read_ok}\n"
        f"⚠️ Omitidos (vacíos): {summary.n_skipped_empty}\n"
        f"❌ Con error: {summary.n_errors}\n"
        f"📌 Registros salida: {summary.n_rows_out}\n"
        f"🧭 CRS final: {summary.crs_base}\n"
        f"🧩 Tipos geometría: {summary.geom_types}\n"
        f"🧾 Columnas: {summary.columns_out}\n"
        f"💾 Salida: {summary.out_path}\n"
    )