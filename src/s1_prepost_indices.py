"""
Procesamiento Sentinel-1 pre/post temporal.

Este módulo:
1. Lee mosaicos Sentinel-1 en dB.
2. Extrae la fecha desde nombres tipo S1_YYYY-MM-DD_idx.tif.
3. Para cada mosaico central, busca:
   - pre  = mosaico más cercano del mes anterior
   - post = mosaico más cercano del mes siguiente
4. Alinea automáticamente pre y post al grid del raster central.
5. Convierte VV y VH de dB a escala lineal.
6. Calcula:
   - VV_Difference
   - VH_Difference
   - VHVV_Difference
7. Guarda las bandas derivadas en dB.
8. Escribe un TIFF final con 7 bandas:
   VV, VH, angle, VVVH_ratio, VV_Difference, VH_Difference, VHVV_Difference.
9. Genera un CSV de trazabilidad.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import DatasetReader
from rasterio.warp import reproject, Resampling
from rasterio.errors import RasterioIOError


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

S1_FILENAME_PATTERN = re.compile(
    r"^S1_(?P<date>\d{4}-\d{2}-\d{2})_(?P<idx>\d+)\.tif$",
    re.IGNORECASE
)

BAND_NAMES_OUT = [
    "VV",
    "VH",
    "angle",
    "VVVH_ratio",
    "VV_Difference",
    "VH_Difference",
    "VHVV_Difference",
]


# ============================================================
# FUNCIONES DE FECHA Y NOMBRE
# ============================================================

def parse_s1_filename(path: Path) -> Optional[Dict]:
    """
    Extrae fecha e índice desde un archivo tipo:

        S1_2021-12-02_0.tif

    Retorna None si el nombre no cumple el patrón.
    """
    match = S1_FILENAME_PATTERN.match(path.name)

    if match is None:
        return None

    date = pd.to_datetime(match.group("date"), format="%Y-%m-%d", errors="coerce")

    if pd.isna(date):
        return None

    idx = int(match.group("idx"))

    return {
        "filename": path.name,
        "stem": path.stem,
        "path": str(path),
        "date": pd.Timestamp(date),
        "year": int(date.year),
        "month": int(date.month),
        "day": int(date.day),
        "month_index": idx,
    }


def index_s1_rasters(input_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Indexa todos los TIFF Sentinel-1 disponibles en una carpeta.

    Espera nombres tipo:
        S1_YYYY-MM-DD_idx.tif
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    tif_paths = sorted(input_dir.glob("S1_*.tif"))

    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No se encontraron archivos S1_*.tif en: {input_dir}")

    rows = []
    skipped = []

    for path in tif_paths:
        parsed = parse_s1_filename(path)

        if parsed is None:
            skipped.append(path.name)
            continue

        rows.append(parsed)

    if len(rows) == 0:
        raise ValueError(
            "No se pudo indexar ningún raster Sentinel-1. "
            "Verifica que los nombres cumplan el patrón S1_YYYY-MM-DD_idx.tif."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "month_index", "filename"]).reset_index(drop=True)

    logger.info(f"[S1] Archivos encontrados: {len(tif_paths)}")
    logger.info(f"[S1] Archivos indexados correctamente: {len(df)}")
    logger.info(f"[S1] Archivos omitidos por nombre inválido: {len(skipped)}")

    if skipped:
        logger.warning(f"[S1] Archivos omitidos: {skipped}")

    return df


def previous_month(year: int, month: int) -> Tuple[int, int]:
    """
    Retorna año y mes anterior.
    """
    if month == 1:
        return year - 1, 12

    return year, month - 1


def next_month(year: int, month: int) -> Tuple[int, int]:
    """
    Retorna año y mes siguiente.
    """
    if month == 12:
        return year + 1, 1

    return year, month + 1


def select_closest_scene_in_month(
    s1_df: pd.DataFrame,
    target_date: pd.Timestamp,
    year: int,
    month: int,
) -> Optional[pd.Series]:
    """
    Selecciona el mosaico más cercano a target_date dentro de un año-mes dado.

    Si hay varios mosaicos en el mes anterior o siguiente, se selecciona
    solo el más cercano temporalmente a la fecha central.
    """
    candidates = s1_df[
        (s1_df["year"] == int(year)) &
        (s1_df["month"] == int(month))
    ].copy()

    if candidates.empty:
        return None

    candidates["abs_delta_days"] = (
        candidates["date"] - pd.Timestamp(target_date)
    ).abs().dt.days

    candidates = candidates.sort_values(
        ["abs_delta_days", "date", "month_index", "filename"]
    ).reset_index(drop=True)

    return candidates.iloc[0]


def select_pre_post_for_central(
    s1_df: pd.DataFrame,
    central_row: pd.Series,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Para una imagen central, selecciona:

    pre  = mosaico más cercano del mes anterior
    post = mosaico más cercano del mes siguiente
    """
    cyear = int(central_row["year"])
    cmonth = int(central_row["month"])
    cdate = pd.Timestamp(central_row["date"])

    pre_y, pre_m = previous_month(cyear, cmonth)
    post_y, post_m = next_month(cyear, cmonth)

    pre = select_closest_scene_in_month(
        s1_df=s1_df,
        target_date=cdate,
        year=pre_y,
        month=pre_m,
    )

    post = select_closest_scene_in_month(
        s1_df=s1_df,
        target_date=cdate,
        year=post_y,
        month=post_m,
    )

    return pre, post


# ============================================================
# CONVERSIONES dB / LINEAL
# ============================================================

def db_to_linear(arr_db: np.ndarray) -> np.ndarray:
    """
    Convierte dB a escala lineal.

    linear = 10 ** (dB / 10)
    """
    return np.power(10.0, arr_db / 10.0)


def linear_to_db(arr_linear: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Convierte escala lineal a dB.

    db = 10 * log10(linear)
    """
    arr_safe = np.where(arr_linear > eps, arr_linear, np.nan)
    return 10.0 * np.log10(arr_safe)


# ============================================================
# LECTURA, NODATA Y ALINEACIÓN
# ============================================================

def _get_src_nodata(src: DatasetReader, fallback_nodata: float) -> float:
    """
    Obtiene nodata del raster. Si no existe, usa fallback_nodata.
    """
    if src.nodata is None:
        return fallback_nodata

    return float(src.nodata)


def _clean_array(
    arr: np.ndarray,
    src_nodata: Optional[float] = None,
    output_nodata: float = -9999.0,
) -> np.ndarray:
    """
    Convierte un array a float32 y reemplaza NoData e infinitos por NaN.
    """
    arr = arr.astype("float32", copy=False)

    if src_nodata is not None:
        arr = np.where(arr == src_nodata, np.nan, arr)

    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.where(arr == output_nodata, np.nan, arr)

    return arr


def read_central_bands(
    src: DatasetReader,
    nodata: float = -9999.0,
) -> np.ndarray:
    """
    Lee las primeras 4 bandas del raster central:

    1 = VV
    2 = VH
    3 = angle
    4 = VVVH_ratio

    Retorna array con forma:
        (4, height, width)
    """
    if src.count < 4:
        raise ValueError(
            f"El raster central tiene {src.count} bandas. "
            "Se esperaban al menos 4 bandas: VV, VH, angle, VVVH_ratio."
        )

    src_nodata = _get_src_nodata(src, nodata)

    arr = src.read([1, 2, 3, 4]).astype("float32")

    for i in range(arr.shape[0]):
        arr[i] = _clean_array(
            arr=arr[i],
            src_nodata=src_nodata,
            output_nodata=nodata,
        )

    return arr


def read_and_align_bands_to_reference(
    src_path: Path,
    ref_src: DatasetReader,
    band_indices: List[int],
    nodata: float = -9999.0,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """
    Lee bandas desde src_path y las reproyecta/alinea al grid del raster central.

    La referencia espacial es ref_src:
    - CRS
    - transform
    - width
    - height
    """
    src_path = Path(src_path)

    with rasterio.open(src_path) as src:
        if src.count < max(band_indices):
            raise ValueError(
                f"El raster {src_path.name} tiene {src.count} bandas. "
                f"Se intentó leer la banda {max(band_indices)}."
            )

        src_nodata = _get_src_nodata(src, nodata)

        out = np.full(
            shape=(len(band_indices), ref_src.height, ref_src.width),
            fill_value=np.nan,
            dtype="float32",
        )

        for j, band_idx in enumerate(band_indices):
            src_band = src.read(band_idx).astype("float32")
            src_band = _clean_array(
                arr=src_band,
                src_nodata=src_nodata,
                output_nodata=nodata,
            )

            dst_band = np.full(
                shape=(ref_src.height, ref_src.width),
                fill_value=np.nan,
                dtype="float32",
            )

            reproject(
                source=src_band,
                destination=dst_band,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=np.nan,
                dst_transform=ref_src.transform,
                dst_crs=ref_src.crs,
                dst_nodata=np.nan,
                resampling=resampling,
            )

            dst_band = np.where(np.isfinite(dst_band), dst_band, np.nan)
            out[j] = dst_band

    return out


# ============================================================
# CÁLCULO DE ÍNDICES PRE/POST
# ============================================================

def compute_prepost_indices_db(
    pre_arr_db: np.ndarray,
    post_arr_db: np.ndarray,
    eps: float = 1e-10,
) -> Dict[str, np.ndarray]:
    """
    Calcula las bandas derivadas usando pre y post.

    Entrada:
        pre_arr_db  = array (2, height, width) con VV_pre, VH_pre en dB
        post_arr_db = array (2, height, width) con VV_post, VH_post en dB

    Como las entradas están en dB:
        1. VV y VH se convierten a escala lineal.
        2. Las razones se calculan en escala lineal.
        3. Las razones se convierten nuevamente a dB.

    Retorna:
        VV_Difference en dB
        VH_Difference en dB
        VHVV_Difference en dB
    """
    vv_pre_db = pre_arr_db[0]
    vh_pre_db = pre_arr_db[1]

    vv_post_db = post_arr_db[0]
    vh_post_db = post_arr_db[1]

    valid = (
        np.isfinite(vv_pre_db) &
        np.isfinite(vh_pre_db) &
        np.isfinite(vv_post_db) &
        np.isfinite(vh_post_db)
    )

    vv_pre_lin = db_to_linear(vv_pre_db)
    vh_pre_lin = db_to_linear(vh_pre_db)
    vv_post_lin = db_to_linear(vv_post_db)
    vh_post_lin = db_to_linear(vh_post_db)

    vv_pre_lin = np.where(vv_pre_lin > eps, vv_pre_lin, np.nan)
    vh_pre_lin = np.where(vh_pre_lin > eps, vh_pre_lin, np.nan)
    vv_post_lin = np.where(vv_post_lin > eps, vv_post_lin, np.nan)
    vh_post_lin = np.where(vh_post_lin > eps, vh_post_lin, np.nan)

    vv_ratio_lin = vv_pre_lin / (vv_post_lin + eps)
    vh_ratio_lin = vh_pre_lin / (vh_post_lin + eps)

    vhvv_pre_lin = vh_pre_lin / (vv_pre_lin + eps)
    vhvv_post_lin = vh_post_lin / (vv_post_lin + eps)
    vhvv_ratio_lin = vhvv_pre_lin / (vhvv_post_lin + eps)

    vv_difference_db = linear_to_db(vv_ratio_lin, eps=eps)
    vh_difference_db = linear_to_db(vh_ratio_lin, eps=eps)
    vhvv_difference_db = linear_to_db(vhvv_ratio_lin, eps=eps)

    vv_difference_db = np.where(valid & np.isfinite(vv_difference_db), vv_difference_db, np.nan)
    vh_difference_db = np.where(valid & np.isfinite(vh_difference_db), vh_difference_db, np.nan)
    vhvv_difference_db = np.where(valid & np.isfinite(vhvv_difference_db), vhvv_difference_db, np.nan)

    return {
        "VV_Difference": vv_difference_db.astype("float32"),
        "VH_Difference": vh_difference_db.astype("float32"),
        "VHVV_Difference": vhvv_difference_db.astype("float32"),
    }


# ============================================================
# ESCRITURA DE RASTER
# ============================================================

def write_output_raster(
    output_path: Path,
    ref_src: DatasetReader,
    central_arr: np.ndarray,
    derived: Dict[str, np.ndarray],
    nodata: float = -9999.0,
) -> None:
    """
    Escribe TIFF final con 7 bandas:

    1 = VV
    2 = VH
    3 = angle
    4 = VVVH_ratio
    5 = VV_Difference
    6 = VH_Difference
    7 = VHVV_Difference
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_arr = np.stack(
        [
            central_arr[0],
            central_arr[1],
            central_arr[2],
            central_arr[3],
            derived["VV_Difference"],
            derived["VH_Difference"],
            derived["VHVV_Difference"],
        ],
        axis=0,
    ).astype("float32")

    out_arr = np.where(np.isfinite(out_arr), out_arr, nodata).astype("float32")

    profile = ref_src.profile.copy()
    profile.update(
        driver="GTiff",
        count=7,
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_arr)

        for idx, band_name in enumerate(BAND_NAMES_OUT, start=1):
            dst.set_band_description(idx, band_name)


# ============================================================
# PROCESAMIENTO DE UNA IMAGEN CENTRAL
# ============================================================

def process_one_central_scene(
    central_row: pd.Series,
    pre_row: pd.Series,
    post_row: pd.Series,
    output_dir: Path,
    eps: float,
    nodata: float,
    logger: logging.Logger,
) -> Dict:
    """
    Procesa una imagen central:
    - abre el raster central;
    - lee bandas originales;
    - alinea pre y post;
    - calcula índices;
    - escribe TIFF final;
    - retorna registro de trazabilidad.
    """
    central_path = Path(central_row["path"])
    pre_path = Path(pre_row["path"])
    post_path = Path(post_row["path"])

    output_name = f"{central_path.stem}_prepost_indices.tif"
    output_path = Path(output_dir) / output_name

    with rasterio.open(central_path) as central_src:
        if central_src.count < 4:
            raise ValueError(
                f"{central_path.name} tiene {central_src.count} bandas. "
                "Se esperaban al menos 4."
            )

        central_arr = read_central_bands(
            src=central_src,
            nodata=nodata,
        )

        pre_arr = read_and_align_bands_to_reference(
            src_path=pre_path,
            ref_src=central_src,
            band_indices=[1, 2],
            nodata=nodata,
            resampling=Resampling.bilinear,
        )

        post_arr = read_and_align_bands_to_reference(
            src_path=post_path,
            ref_src=central_src,
            band_indices=[1, 2],
            nodata=nodata,
            resampling=Resampling.bilinear,
        )

        derived = compute_prepost_indices_db(
            pre_arr_db=pre_arr,
            post_arr_db=post_arr,
            eps=eps,
        )

        write_output_raster(
            output_path=output_path,
            ref_src=central_src,
            central_arr=central_arr,
            derived=derived,
            nodata=nodata,
        )

    logger.info(
        f"[OK] {central_path.name} | "
        f"pre={pre_path.name} | post={post_path.name} | out={output_path.name}"
    )

    return {
        "central_file": central_path.name,
        "central_date": pd.Timestamp(central_row["date"]).date().isoformat(),
        "central_year": int(central_row["year"]),
        "central_month": int(central_row["month"]),
        "pre_file": pre_path.name,
        "pre_date": pd.Timestamp(pre_row["date"]).date().isoformat(),
        "pre_delta_days": int(
            abs((pd.Timestamp(central_row["date"]) - pd.Timestamp(pre_row["date"])).days)
        ),
        "post_file": post_path.name,
        "post_date": pd.Timestamp(post_row["date"]).date().isoformat(),
        "post_delta_days": int(
            abs((pd.Timestamp(post_row["date"]) - pd.Timestamp(central_row["date"])).days)
        ),
        "output_file": output_path.name,
        "status": "OK",
        "message": "Raster generado correctamente.",
    }


# ============================================================
# PROCESAMIENTO GENERAL
# ============================================================

def process_all_s1_prepost(
    input_dir: Path,
    output_dir: Path,
    eps: float = 1e-10,
    nodata: float = -9999.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Procesa todos los mosaicos Sentinel-1 disponibles.

    Para cada imagen central:
    - busca imagen más cercana del mes anterior;
    - busca imagen más cercana del mes siguiente;
    - si ambas existen, genera un TIFF con 7 bandas;
    - si falta alguna, registra el caso en la trazabilidad.

    Retorna:
        DataFrame de trazabilidad.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if logger is None:
        logger = logging.getLogger("S1_PREPOST")

    s1_df = index_s1_rasters(input_dir=input_dir, logger=logger)

    trace_rows = []

    for _, central_row in s1_df.iterrows():
        central_file = central_row["filename"]
        central_date = pd.Timestamp(central_row["date"])

        try:
            pre_row, post_row = select_pre_post_for_central(
                s1_df=s1_df,
                central_row=central_row,
            )

            if pre_row is None and post_row is None:
                status = "SKIPPED_NO_PRE_POST"
                message = "No existe imagen del mes anterior ni del mes siguiente."
            elif pre_row is None:
                status = "SKIPPED_NO_PRE"
                message = "No existe imagen del mes anterior."
            elif post_row is None:
                status = "SKIPPED_NO_POST"
                message = "No existe imagen del mes siguiente."
            else:
                result = process_one_central_scene(
                    central_row=central_row,
                    pre_row=pre_row,
                    post_row=post_row,
                    output_dir=output_dir,
                    eps=eps,
                    nodata=nodata,
                    logger=logger,
                )
                trace_rows.append(result)
                continue

            logger.warning(f"[{status}] {central_file} | {message}")

            trace_rows.append(
                {
                    "central_file": central_file,
                    "central_date": central_date.date().isoformat(),
                    "central_year": int(central_row["year"]),
                    "central_month": int(central_row["month"]),
                    "pre_file": None if pre_row is None else pre_row["filename"],
                    "pre_date": None if pre_row is None else pd.Timestamp(pre_row["date"]).date().isoformat(),
                    "pre_delta_days": None if pre_row is None else int(
                        abs((central_date - pd.Timestamp(pre_row["date"])).days)
                    ),
                    "post_file": None if post_row is None else post_row["filename"],
                    "post_date": None if post_row is None else pd.Timestamp(post_row["date"]).date().isoformat(),
                    "post_delta_days": None if post_row is None else int(
                        abs((pd.Timestamp(post_row["date"]) - central_date).days)
                    ),
                    "output_file": None,
                    "status": status,
                    "message": message,
                }
            )

        except Exception as exc:
            logger.exception(f"[ERROR] {central_file}: {exc}")

            trace_rows.append(
                {
                    "central_file": central_file,
                    "central_date": central_date.date().isoformat(),
                    "central_year": int(central_row["year"]),
                    "central_month": int(central_row["month"]),
                    "pre_file": None,
                    "pre_date": None,
                    "pre_delta_days": None,
                    "post_file": None,
                    "post_date": None,
                    "post_delta_days": None,
                    "output_file": None,
                    "status": "ERROR",
                    "message": str(exc),
                }
            )

    trace = pd.DataFrame(trace_rows)

    trace_path = output_dir / "s1_prepost_trace.csv"
    trace.to_csv(trace_path, index=False, encoding="utf-8-sig")

    logger.info(f"Trazabilidad guardada en: {trace_path}")
    logger.info("Resumen de procesamiento:")
    logger.info("\n" + trace["status"].value_counts(dropna=False).to_string())

    return trace