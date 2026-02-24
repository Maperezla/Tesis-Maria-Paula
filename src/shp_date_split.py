import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import geopandas as gpd
import pandas as pd


@dataclass
class DateSplitSummary:
    n_rows: int
    n_null_dates: int
    out_path: str


def setup_logger(log_path: str, name: str = "shp_date_split") -> logging.Logger:
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


def ensure_columns(gdf: gpd.GeoDataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")


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


def split_date_columns(
    gdf: gpd.GeoDataFrame,
    date_col: str = "acq_date_n",
    date_format: str = "%Y-%m-%d",
    year_col: str = "anio",
    month_col: str = "mes",
    day_col: str = "dia",
) -> Tuple[gpd.GeoDataFrame, int]:
    """
    Convierte date_col a datetime y crea anio/mes/dia.
    Robusto ante NaT (no castea a int directamente).
    """
    gdf = gdf.copy()
    gdf[date_col] = pd.to_datetime(gdf[date_col], format=date_format, errors="coerce")

    n_null = int(gdf[date_col].isna().sum())

    # Usar Int64 (nullable) para no reventar si hay NaT
    gdf[year_col] = gdf[date_col].dt.year.astype("Int64")
    gdf[month_col] = gdf[date_col].dt.month.astype("Int64")
    gdf[day_col] = gdf[date_col].dt.day.astype("Int64")

    return gdf, n_null


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


def process_split_date_shp(
    in_path: str,
    out_path: str,
    date_col: str = "acq_date_n",
    date_format: str = "%Y-%m-%d",
    version_out: bool = True,
    logger: Optional[logging.Logger] = None
) -> DateSplitSummary:
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"No existe el shapefile: {in_path}")

    gdf = gpd.read_file(in_path)

    ensure_columns(gdf, [date_col])

    gdf2, n_null = split_date_columns(gdf, date_col=date_col, date_format=date_format)

    final_out = write_shp_versioned(gdf2, out_path, version=version_out, logger=logger)

    if logger:
        logger.info(f"Entrada: {in_path} | filas={len(gdf)}")
        logger.info(f"Nulos en {date_col}: {n_null}")

    return DateSplitSummary(n_rows=len(gdf2), n_null_dates=n_null, out_path=final_out)


def format_report(summary: DateSplitSummary, date_col: str = "acq_date_n") -> str:
    return (
        "\n--- REPORTE FINAL (DESGLOSE FECHA) ---\n"
        f"📊 Filas procesadas: {summary.n_rows}\n"
        f"⚠️ Fechas nulas/No parseables en '{date_col}': {summary.n_null_dates}\n"
        f"💾 Salida: {summary.out_path}\n"
    )