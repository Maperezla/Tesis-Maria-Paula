import os
import re
import glob
import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from .points_loader import load_points, assign_dates_to_absences


# =========================================================
# CONFIG FIJA DE FEATURES
# =========================================================

# Landsat 8 normalizado:
# Orden fijo de bandas dentro de cada TIFF normalizado.
L8_FEATURES = [
    "SR_B5_norm",
    "SR_B6_norm",
    "SR_B7_norm",
    "NDVI_norm",
    "EVI_norm",
    "NBR_norm",
    "NDWI_norm",
]

# Sentinel-1 normalizado:
# Orden fijo de bandas dentro de cada TIFF normalizado.
S1_FEATURES = [
    "VV",
    "VH",
    "angle",
    "VVVH_ratio",
    "VV_Difference",
    "VH_Difference",
    "VHVV_Difference",
]

# =========================================================
# CONFIG MODELO C
# =========================================================

S1_GENERAL_FEATURES_MODEL_C = [
    "VV",
    "VH",
    "angle",
    "VVVH_ratio",
]

S1_DIFFERENCE_FEATURES_MODEL_C = [
    "VV_Difference_norm",
    "VH_Difference_norm",
    "VHVV_Difference_norm",
]

FEATURES_C = L8_FEATURES + S1_GENERAL_FEATURES_MODEL_C + S1_DIFFERENCE_FEATURES_MODEL_C

# Orden real de bandas dentro del TIFF Sentinel-1 normalizado.
S1_TIF_BAND_ORDER = [
    "VV",
    "VH",
    "angle",
    "VVVH_ratio",
    "VV_Difference",
    "VH_Difference",
    "VHVV_Difference",
]

S1_GENERAL_BAND_INDICES_MODEL_C = [1, 2, 3, 4]

S1_DIFFERENCE_BAND_INDICES_MODEL_C = [5, 6, 7]

FEATURES = L8_FEATURES + S1_FEATURES


# =========================================================
# HELPERS FECHAS Y BIMESTRES
# =========================================================

BIMONTH_LABEL_TO_START_MONTH = {
    "ene_feb": 1,
    "mar_abr": 3,
    "may_jun": 5,
    "jul_ago": 7,
    "sep_oct": 9,
    "nov_dic": 11,
}

START_MONTH_TO_BIMONTH_LABEL = {
    1: "ene_feb",
    3: "mar_abr",
    5: "may_jun",
    7: "jul_ago",
    9: "sep_oct",
    11: "nov_dic",
}


def bimonth_start_month(month: int) -> int:
    """
    Devuelve el mes inicial del bimestre al que pertenece un mes.
    Ejemplo:
    enero/febrero -> 1
    marzo/abril   -> 3
    """
    if month in [1, 2]:
        return 1
    if month in [3, 4]:
        return 3
    if month in [5, 6]:
        return 5
    if month in [7, 8]:
        return 7
    if month in [9, 10]:
        return 9
    if month in [11, 12]:
        return 11
    raise ValueError(f"Mes inválido: {month}")


def next_bimonth(year: int, start_month: int) -> Tuple[int, int]:
    """
    Devuelve el siguiente bimestre.
    Si el bimestre es nov-dic, retorna ene-feb del año siguiente.
    """
    if start_month == 11:
        return year + 1, 1
    return year, start_month + 2


def build_bimonth_start_date(year: int, start_month: int) -> pd.Timestamp:
    return pd.Timestamp(year=int(year), month=int(start_month), day=1)


def get_l8_target_bimonth(
    t0: pd.Timestamp,
    switch_day: int = 14
) -> Tuple[int, int, str]:
    """
    Determina qué bimestre Landsat 8 debe usarse para una fecha de incendio.

    Regla:
    - Primero se identifica el bimestre actual del incendio.
    - Si el incendio ocurrió antes del día umbral del bimestre, se usa el bimestre actual.
    - Si ocurrió desde el día umbral en adelante, se usa el bimestre siguiente.

    Con switch_day = 14:
    - 2017-03-13 -> mar_abr
    - 2017-03-14 -> may_jun

    La comparación se hace respecto al inicio real del bimestre.
    """
    if pd.isna(t0):
        return None, None, None

    t0 = pd.to_datetime(t0, errors="coerce")
    if pd.isna(t0):
        return None, None, None

    year = int(t0.year)
    start_month = bimonth_start_month(int(t0.month))
    start_date = build_bimonth_start_date(year, start_month)

    # Día 14 del bimestre equivale a start_date + 13 días.
    switch_date = start_date + pd.Timedelta(days=switch_day - 1)

    if t0 >= switch_date:
        target_year, target_start_month = next_bimonth(year, start_month)
        rule = "next_bimonth"
    else:
        target_year, target_start_month = year, start_month
        rule = "current_bimonth"

    return target_year, target_start_month, rule


def ym_neighbors(y: int, m: int) -> List[Tuple[int, int]]:
    """
    Se mantiene para el balanceo temporal de ausencias.
    Usa el bimestre Landsat asignado como referencia.
    """
    prev_y, prev_m = (y, m - 2) if m > 1 else (y - 1, 11)
    next_y, next_m = (y, m + 2) if m < 11 else (y + 1, 1)
    return [(y, m), (prev_y, prev_m), (next_y, next_m)]


# =========================================================
# INDEXACIÓN RASTERS LANDSAT 8
# =========================================================

def index_l8_rasters(l8_dir: str, logger: logging.Logger) -> Dict[Tuple[int, int], str]:
    """
    Indexa TIFF Landsat 8 normalizados con patrón:
    L8_2017_ene_feb_normalizado.tif

    Retorna:
    {
        (2017, 1): ruta_ene_feb,
        (2017, 3): ruta_mar_abr,
        ...
    }
    """
    pattern = os.path.join(l8_dir, "L8_*_normalizado.tif")
    paths = glob.glob(pattern)

    index = {}
    bad_files = []

    regex = re.compile(
        r"^L8_(?P<year>\d{4})_(?P<label>ene_feb|mar_abr|may_jun|jul_ago|sep_oct|nov_dic)_normalizado$",
        re.IGNORECASE
    )

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = regex.match(stem)

        if not m:
            bad_files.append(os.path.basename(p))
            continue

        year = int(m.group("year"))
        label = m.group("label").lower()
        start_month = BIMONTH_LABEL_TO_START_MONTH[label]

        key = (year, start_month)

        if key in index:
            logger.warning(
                f"[L8] Índice duplicado para {key}. "
                f"Se reemplaza {os.path.basename(index[key])} por {os.path.basename(p)}"
            )

        index[key] = p

    logger.info(f"[L8] Carpeta: {l8_dir}")
    logger.info(f"[L8] TIFF encontrados con patrón general: {len(paths)}")
    logger.info(f"[L8] TIFF indexados correctamente: {len(index)}")

    if bad_files:
        logger.warning(f"[L8] Archivos no parseados por nombre: {bad_files[:20]}")

    return dict(sorted(index.items()))


def find_l8_path_for_date(
    t0: pd.Timestamp,
    l8_index: Dict[Tuple[int, int], str],
    switch_day: int = 14,
    fallback_next_available: bool = True
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str], Optional[str]]:
    """
    Selecciona el TIFF Landsat 8 para un punto.

    Retorna:
    path, year, bimonth_start_month, bimonth_label, selection_rule
    """
    y, bm, rule = get_l8_target_bimonth(t0, switch_day=switch_day)

    if y is None or bm is None:
        return None, None, None, None, "invalid_date"

    key = (int(y), int(bm))

    if key in l8_index:
        label = START_MONTH_TO_BIMONTH_LABEL[int(bm)]
        return l8_index[key], int(y), int(bm), label, rule

    if not fallback_next_available:
        label = START_MONTH_TO_BIMONTH_LABEL[int(bm)]
        return None, int(y), int(bm), label, "target_missing_no_fallback"

    # Buscar el siguiente bimestre disponible más cercano.
    available_keys = sorted(l8_index.keys())
    future_keys = [k for k in available_keys if k >= key]

    if not future_keys:
        label = START_MONTH_TO_BIMONTH_LABEL[int(bm)]
        return None, int(y), int(bm), label, "target_missing_no_future_available"

    fallback_key = future_keys[0]
    fallback_y, fallback_bm = fallback_key
    fallback_label = START_MONTH_TO_BIMONTH_LABEL[int(fallback_bm)]

    return (
        l8_index[fallback_key],
        int(fallback_y),
        int(fallback_bm),
        fallback_label,
        f"{rule}_fallback_next_available"
    )


# =========================================================
# INDEXACIÓN RASTERS SENTINEL-1
# =========================================================

def index_s1_rasters_from_filenames(
    s1_dir: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Indexa TIFF Sentinel-1 normalizados extrayendo la fecha y el índice desde el nombre.

    Patrón esperado:
    S1_2017-02-26_2_prepost_indices_normalizado.tif

    Retorna DataFrame con:
    filename_prefix, path, date, year, month, day, scene_index
    """
    pattern = os.path.join(s1_dir, "S1_*_normalizado.tif")
    paths = glob.glob(pattern)

    regex = re.compile(
        r"^S1_(?P<date>\d{4}-\d{2}-\d{2})_(?P<idx>\d+)_.*normalizado$",
        re.IGNORECASE
    )

    rows = []
    bad_files = []

    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = regex.match(stem)

        if not m:
            bad_files.append(os.path.basename(p))
            continue

        date = pd.to_datetime(m.group("date"), format="%Y-%m-%d", errors="coerce")

        if pd.isna(date):
            bad_files.append(os.path.basename(p))
            continue

        scene_index = int(m.group("idx"))

        rows.append({
            "filename_prefix": stem,
            "path": p,
            "date": date,
            "year": int(date.year),
            "month": int(date.month),
            "day": int(date.day),
            "scene_index": scene_index,
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(
            ["date", "scene_index"],
            ascending=[True, False]
        ).reset_index(drop=True)

    logger.info(f"[S1] Carpeta: {s1_dir}")
    logger.info(f"[S1] TIFF encontrados con patrón general: {len(paths)}")
    logger.info(f"[S1] TIFF indexados correctamente desde nombre: {len(df)}")

    if bad_files:
        logger.warning(f"[S1] Archivos no parseados por nombre: {bad_files[:20]}")

    return df


def select_s1_scene_for_point(
    t0,
    s1_df: pd.DataFrame,
    start_offset_days: int,
    window_days: int
) -> Optional[pd.Series]:
    """
    Selecciona la primera imagen Sentinel-1 disponible después del incendio
    dentro de la ventana temporal.

    Reglas:
    - Fecha mínima = acq_date + start_offset_days.
    - Fecha máxima = acq_date + window_days.
    - Si hay varias imágenes el mismo día, selecciona la de mayor scene_index.
    """
    if t0 is None or pd.isna(t0):
        return None

    t0 = pd.to_datetime(t0, errors="coerce")
    if pd.isna(t0):
        return None

    if getattr(t0, "tzinfo", None) is not None:
        t0 = t0.tz_convert(None)

    if s1_df.empty:
        return None

    if "date" not in s1_df.columns:
        raise ValueError("s1_df no tiene columna 'date'.")

    dates = pd.to_datetime(s1_df["date"], errors="coerce")

    try:
        if hasattr(dates.dt, "tz") and dates.dt.tz is not None:
            dates = dates.dt.tz_convert(None)
    except Exception:
        pass

    start = t0 + pd.Timedelta(days=start_offset_days)
    end = t0 + pd.Timedelta(days=window_days)

    mask = (dates >= start) & (dates <= end)

    cand = s1_df.loc[mask].copy()

    if cand.empty:
        return None

    cand = cand.sort_values(
        ["date", "scene_index"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return cand.iloc[0]
    

def select_s1_scene_same_month_for_difference_bands(
    t0,
    s1_df: pd.DataFrame,
    selection_strategy: str = "closest_to_fire_date"
) -> Optional[pd.Series]:
    """
    Selecciona una imagen Sentinel-1 del mismo año y mes de la fecha del punto.

    Se usa exclusivamente para las bandas:
    - VV_Difference_norm
    - VH_Difference_norm
    - VHVV_Difference_norm

    Regla principal:
    - Debe pertenecer al mismo año y mes de acq_date.
    - Si hay varias imágenes, se selecciona la más cercana a la fecha del incendio.
    - Si hay empate en distancia temporal, se selecciona la de mayor scene_index.
    - Si no hay imagen en ese mes, retorna None.
    """
    if t0 is None or pd.isna(t0):
        return None

    t0 = pd.to_datetime(t0, errors="coerce")

    if pd.isna(t0):
        return None

    if getattr(t0, "tzinfo", None) is not None:
        t0 = t0.tz_convert(None)

    if s1_df.empty:
        return None

    required_cols = {"date", "year", "month", "scene_index"}

    if not required_cols.issubset(set(s1_df.columns)):
        raise ValueError(
            f"s1_df no tiene las columnas requeridas {required_cols}. "
            f"Columnas disponibles: {list(s1_df.columns)}"
        )

    cand = s1_df[
        (s1_df["year"].astype(int) == int(t0.year)) &
        (s1_df["month"].astype(int) == int(t0.month))
    ].copy()

    if cand.empty:
        return None

    cand["date"] = pd.to_datetime(cand["date"], errors="coerce")
    cand = cand.dropna(subset=["date"]).copy()

    if cand.empty:
        return None

    if selection_strategy == "closest_to_fire_date":
        cand["abs_delta_days"] = (cand["date"] - t0).abs().dt.days

        cand = cand.sort_values(
            ["abs_delta_days", "scene_index"],
            ascending=[True, False]
        ).reset_index(drop=True)

        return cand.iloc[0]

    elif selection_strategy == "first_available_in_month":
        cand = cand.sort_values(
            ["date", "scene_index"],
            ascending=[True, False]
        ).reset_index(drop=True)

        return cand.iloc[0]

    elif selection_strategy == "last_available_in_month":
        cand = cand.sort_values(
            ["date", "scene_index"],
            ascending=[False, False]
        ).reset_index(drop=True)

        return cand.iloc[0]

    elif selection_strategy == "highest_index_same_month":
        cand = cand.sort_values(
            ["scene_index", "date"],
            ascending=[False, True]
        ).reset_index(drop=True)

        return cand.iloc[0]

    else:
        raise ValueError(
            f"Estrategia no reconocida: {selection_strategy}. "
            "Opciones válidas: "
            "'closest_to_fire_date', "
            "'first_available_in_month', "
            "'last_available_in_month', "
            "'highest_index_same_month'."
        )


# =========================================================
# EXTRACCIÓN FEATURES
# =========================================================

def sample_raster_at_points(
    raster_path: str,
    pts: gpd.GeoDataFrame,
    band_indices_1based: List[int]
) -> np.ndarray:
    """
    Extrae valores raster en puntos.
    Reproyecta los puntos al CRS del raster antes de muestrear.
    """
    with rasterio.open(raster_path) as src:
        pts_proj = pts.to_crs(src.crs)

        coords = [(geom.x, geom.y) for geom in pts_proj.geometry]

        samples = list(src.sample(coords, indexes=band_indices_1based))
        arr = np.array(samples, dtype="float64")

        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan

        arr[~np.isfinite(arr)] = np.nan

        return arr


def validate_raster_band_count(
    raster_path: str,
    expected_n_bands: int,
    sensor_name: str,
    logger: logging.Logger
) -> bool:
    """
    Valida que el TIFF tenga al menos el número de bandas esperado.
    """
    try:
        with rasterio.open(raster_path) as src:
            n_bands = src.count

        if n_bands < expected_n_bands:
            logger.warning(
                f"[{sensor_name}] Raster con bandas insuficientes: "
                f"{os.path.basename(raster_path)} | "
                f"bandas={n_bands}, esperadas={expected_n_bands}"
            )
            return False

        return True

    except RasterioIOError:
        logger.warning(f"[{sensor_name}] No se pudo abrir raster: {raster_path}")
        return False


def build_master_features(
    g: gpd.GeoDataFrame,
    l8_index: Dict[Tuple[int, int], str],
    s1_df: pd.DataFrame,
    date_col: str,
    l8_switch_day: int,
    l8_fallback_next_available: bool,
    s1_start_offset_days: int,
    s1_window_days: int,
    logger: logging.Logger
) -> gpd.GeoDataFrame:
    """
    Construye g_master con variables Landsat 8 y Sentinel-1 extraídas en puntos.

    Modelo A/B:
    - Landsat 8: regla bimestral con umbral de día 14.
    - Sentinel-1: primera imagen posterior al incendio dentro de la ventana temporal.
    """
    g = g.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    s1_df = s1_df.copy()

    if not s1_df.empty:
        s1_df["date"] = pd.to_datetime(s1_df["date"], errors="coerce")
        s1_df = s1_df.dropna(subset=["date"]).copy()
        s1_df = s1_df.sort_values(
            ["date", "scene_index"],
            ascending=[True, False]
        ).reset_index(drop=True)

    # =====================================================
    # Inicializar columnas de features
    # =====================================================
    for col in FEATURES:
        g[col] = np.nan

    # =====================================================
    # Metadatos Landsat 8
    # =====================================================
    g["l8_found"] = False
    g["l8_file"] = None
    g["l8_year"] = pd.NA
    g["l8_bimonth_start_month"] = pd.NA
    g["l8_bimonth_label"] = None
    g["l8_selection_rule"] = None

    # =====================================================
    # Metadatos Sentinel-1
    # =====================================================
    g["s1_found"] = False
    g["s1_file"] = None
    g["s1_date"] = pd.NaT
    g["s1_scene_index"] = pd.NA

    # =====================================================
    # LANDSAT 8
    # =====================================================
    l8_targets = []

    for t0 in g[date_col]:
        path, y, bm, label, rule = find_l8_path_for_date(
            t0=t0,
            l8_index=l8_index,
            switch_day=l8_switch_day,
            fallback_next_available=l8_fallback_next_available
        )

        l8_targets.append((path, y, bm, label, rule))

    g["_l8_path"] = [x[0] for x in l8_targets]
    g["l8_year"] = [x[1] for x in l8_targets]
    g["l8_bimonth_start_month"] = [x[2] for x in l8_targets]
    g["l8_bimonth_label"] = [x[3] for x in l8_targets]
    g["l8_selection_rule"] = [x[4] for x in l8_targets]

    g["l8_year"] = g["l8_year"].astype("Int64")
    g["l8_bimonth_start_month"] = g["l8_bimonth_start_month"].astype("Int64")

    valid_l8_paths = g["_l8_path"].dropna().unique().tolist()

    logger.info(f"[L8] Rasters Landsat 8 usados por puntos: {len(valid_l8_paths)}")

    for path, grp_idx in g.dropna(subset=["_l8_path"]).groupby("_l8_path").groups.items():
        path = str(path)

        if not validate_raster_band_count(
            raster_path=path,
            expected_n_bands=len(L8_FEATURES),
            sensor_name="L8",
            logger=logger
        ):
            continue

        try:
            vals = sample_raster_at_points(
                raster_path=path,
                pts=g.loc[grp_idx],
                band_indices_1based=list(range(1, len(L8_FEATURES) + 1))
            )
        except RasterioIOError:
            logger.warning(f"[L8] No se pudo muestrear raster: {path}")
            continue

        for j, feature_name in enumerate(L8_FEATURES):
            g.loc[grp_idx, feature_name] = vals[:, j]

        g.loc[grp_idx, "l8_found"] = True
        g.loc[grp_idx, "l8_file"] = os.path.basename(path)

    n_l8_found = int(g["l8_found"].sum())
    n_l8_missing = int((~g["l8_found"]).sum())

    logger.info(f"[L8] Puntos con imagen Landsat 8: {n_l8_found}")
    logger.info(f"[L8] Puntos sin imagen Landsat 8: {n_l8_missing}")

    # =====================================================
    # SENTINEL-1
    # =====================================================
    for i, row in g.iterrows():
        t0 = row[date_col]

        sel = select_s1_scene_for_point(
            t0=t0,
            s1_df=s1_df,
            start_offset_days=s1_start_offset_days,
            window_days=s1_window_days
        )

        if sel is None:
            continue

        path = sel["path"]

        if not validate_raster_band_count(
            raster_path=path,
            expected_n_bands=len(S1_FEATURES),
            sensor_name="S1",
            logger=logger
        ):
            continue

        try:
            vals = sample_raster_at_points(
                raster_path=path,
                pts=g.loc[[i]],
                band_indices_1based=list(range(1, len(S1_FEATURES) + 1))
            )
        except RasterioIOError:
            logger.warning(f"[S1] No se pudo muestrear raster: {path}")
            continue

        for j, feature_name in enumerate(S1_FEATURES):
            g.at[i, feature_name] = vals[0, j]

        g.at[i, "s1_found"] = True
        g.at[i, "s1_file"] = os.path.basename(path)
        g.at[i, "s1_date"] = pd.Timestamp(sel["date"])
        g.at[i, "s1_scene_index"] = int(sel["scene_index"])

    n_s1_found = int(g["s1_found"].sum())
    n_s1_missing = int((~g["s1_found"]).sum())

    logger.info(f"[S1] Puntos con imagen Sentinel-1: {n_s1_found}")
    logger.info(f"[S1] Puntos sin imagen Sentinel-1: {n_s1_missing}")

    g.drop(columns=["_l8_path"], inplace=True)

    return g

def build_master_features_model_c(
    g: gpd.GeoDataFrame,
    l8_index: Dict[Tuple[int, int], str],
    s1_df: pd.DataFrame,
    date_col: str,
    l8_switch_day: int,
    l8_fallback_next_available: bool,
    s1_start_offset_days: int,
    s1_window_days: int,
    s1_difference_selection_strategy: str,
    logger: logging.Logger
) -> gpd.GeoDataFrame:
    """
    Construye g_master_c para Modelo C.

    Modelo C:
    - Landsat 8: misma regla bimestral del Modelo A.
    - Sentinel-1 general: VV, VH, angle, VVVH_ratio desde la primera imagen posterior.
    - Sentinel-1 diferencias: VV_Difference_norm, VH_Difference_norm,
      VHVV_Difference_norm desde imagen del mismo mes del incendio.
    """
    g = g.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    s1_df = s1_df.copy()

    if not s1_df.empty:
        s1_df["date"] = pd.to_datetime(s1_df["date"], errors="coerce")
        s1_df = s1_df.dropna(subset=["date"]).copy()
        s1_df = s1_df.sort_values(
            ["date", "scene_index"],
            ascending=[True, False]
        ).reset_index(drop=True)

    # Inicializar features Modelo C.
    for col in FEATURES_C:
        g[col] = np.nan

    # =====================================================
    # METADATOS LANDSAT 8
    # =====================================================
    g["l8_found"] = False
    g["l8_file"] = None
    g["l8_year"] = pd.NA
    g["l8_bimonth_start_month"] = pd.NA
    g["l8_bimonth_label"] = None
    g["l8_selection_rule"] = None

    # =====================================================
    # METADATOS SENTINEL-1 GENERAL
    # =====================================================
    g["s1_general_found"] = False
    g["s1_general_file"] = None
    g["s1_general_date"] = pd.NaT
    g["s1_general_scene_index"] = pd.NA

    # =====================================================
    # METADATOS SENTINEL-1 DIFERENCIAS
    # =====================================================
    g["s1_difference_found"] = False
    g["s1_difference_file"] = None
    g["s1_difference_date"] = pd.NaT
    g["s1_difference_scene_index"] = pd.NA
    g["s1_difference_selection_rule"] = None

    # =====================================================
    # LANDSAT 8
    # =====================================================
    l8_targets = []

    for t0 in g[date_col]:
        path, y, bm, label, rule = find_l8_path_for_date(
            t0=t0,
            l8_index=l8_index,
            switch_day=l8_switch_day,
            fallback_next_available=l8_fallback_next_available
        )

        l8_targets.append((path, y, bm, label, rule))

    g["_l8_path"] = [x[0] for x in l8_targets]
    g["l8_year"] = [x[1] for x in l8_targets]
    g["l8_bimonth_start_month"] = [x[2] for x in l8_targets]
    g["l8_bimonth_label"] = [x[3] for x in l8_targets]
    g["l8_selection_rule"] = [x[4] for x in l8_targets]

    g["l8_year"] = g["l8_year"].astype("Int64")
    g["l8_bimonth_start_month"] = g["l8_bimonth_start_month"].astype("Int64")

    valid_l8_paths = g["_l8_path"].dropna().unique().tolist()
    logger.info(f"[MODELO C][L8] Rasters Landsat 8 usados por puntos: {len(valid_l8_paths)}")

    for path, grp_idx in g.dropna(subset=["_l8_path"]).groupby("_l8_path").groups.items():
        path = str(path)

        if not validate_raster_band_count(
            raster_path=path,
            expected_n_bands=len(L8_FEATURES),
            sensor_name="MODELO C - L8",
            logger=logger
        ):
            continue

        try:
            vals = sample_raster_at_points(
                raster_path=path,
                pts=g.loc[grp_idx],
                band_indices_1based=list(range(1, len(L8_FEATURES) + 1))
            )
        except RasterioIOError:
            logger.warning(f"[MODELO C][L8] No se pudo muestrear raster: {path}")
            continue

        for j, feature_name in enumerate(L8_FEATURES):
            g.loc[grp_idx, feature_name] = vals[:, j]

        g.loc[grp_idx, "l8_found"] = True
        g.loc[grp_idx, "l8_file"] = os.path.basename(path)

    logger.info(f"[MODELO C][L8] Puntos con imagen Landsat 8: {int(g['l8_found'].sum())}")
    logger.info(f"[MODELO C][L8] Puntos sin imagen Landsat 8: {int((~g['l8_found']).sum())}")

    # =====================================================
    # SENTINEL-1 GENERAL
    # =====================================================
    for i, row in g.iterrows():
        t0 = row[date_col]

        sel_general = select_s1_scene_for_point(
            t0=t0,
            s1_df=s1_df,
            start_offset_days=s1_start_offset_days,
            window_days=s1_window_days
        )

        if sel_general is None:
            continue

        path_general = sel_general["path"]

        if not validate_raster_band_count(
            raster_path=path_general,
            expected_n_bands=len(S1_TIF_BAND_ORDER),
            sensor_name="MODELO C - S1 GENERAL",
            logger=logger
        ):
            continue

        try:
            vals_general = sample_raster_at_points(
                raster_path=path_general,
                pts=g.loc[[i]],
                band_indices_1based=S1_GENERAL_BAND_INDICES_MODEL_C
            )
        except RasterioIOError:
            logger.warning(f"[MODELO C][S1 GENERAL] No se pudo muestrear raster: {path_general}")
            continue

        for j, feature_name in enumerate(S1_GENERAL_FEATURES_MODEL_C):
            g.at[i, feature_name] = vals_general[0, j]

        g.at[i, "s1_general_found"] = True
        g.at[i, "s1_general_file"] = os.path.basename(path_general)
        g.at[i, "s1_general_date"] = pd.Timestamp(sel_general["date"])
        g.at[i, "s1_general_scene_index"] = int(sel_general["scene_index"])

    logger.info(
        f"[MODELO C][S1 GENERAL] Puntos con Sentinel-1 general: "
        f"{int(g['s1_general_found'].sum())}"
    )
    logger.info(
        f"[MODELO C][S1 GENERAL] Puntos sin Sentinel-1 general: "
        f"{int((~g['s1_general_found']).sum())}"
    )

    # =====================================================
    # SENTINEL-1 DIFERENCIAS MISMO MES
    # =====================================================
    for i, row in g.iterrows():
        t0 = row[date_col]

        sel_diff = select_s1_scene_same_month_for_difference_bands(
            t0=t0,
            s1_df=s1_df,
            selection_strategy=s1_difference_selection_strategy
        )

        if sel_diff is None:
            g.at[i, "s1_difference_selection_rule"] = "same_month_missing"
            continue

        path_diff = sel_diff["path"]

        if not validate_raster_band_count(
            raster_path=path_diff,
            expected_n_bands=len(S1_TIF_BAND_ORDER),
            sensor_name="MODELO C - S1 DIFFERENCE",
            logger=logger
        ):
            continue

        try:
            vals_diff = sample_raster_at_points(
                raster_path=path_diff,
                pts=g.loc[[i]],
                band_indices_1based=S1_DIFFERENCE_BAND_INDICES_MODEL_C
            )
        except RasterioIOError:
            logger.warning(f"[MODELO C][S1 DIFFERENCE] No se pudo muestrear raster: {path_diff}")
            continue

        for j, feature_name in enumerate(S1_DIFFERENCE_FEATURES_MODEL_C):
            g.at[i, feature_name] = vals_diff[0, j]

        g.at[i, "s1_difference_found"] = True
        g.at[i, "s1_difference_file"] = os.path.basename(path_diff)
        g.at[i, "s1_difference_date"] = pd.Timestamp(sel_diff["date"])
        g.at[i, "s1_difference_scene_index"] = int(sel_diff["scene_index"])
        g.at[i, "s1_difference_selection_rule"] = s1_difference_selection_strategy

    logger.info(
        f"[MODELO C][S1 DIFFERENCE] Puntos con Sentinel-1 diferencias mismo mes: "
        f"{int(g['s1_difference_found'].sum())}"
    )
    logger.info(
        f"[MODELO C][S1 DIFFERENCE] Puntos sin Sentinel-1 diferencias mismo mes: "
        f"{int((~g['s1_difference_found']).sum())}"
    )

    g.drop(columns=["_l8_path"], inplace=True)

    return g


def coverage_report(g: gpd.GeoDataFrame, features: List[str]) -> pd.DataFrame:
    rep = []

    for col in features:
        rep.append({
            "feature": col,
            "missing_n": int(g[col].isna().sum()),
            "missing_pct": float(g[col].isna().mean() * 100.0),
        })

    return pd.DataFrame(rep).sort_values("missing_pct", ascending=False)


# =========================================================
# DATASETS A / B
# =========================================================

def build_dataset_A_balanced_exact(
    g_master: gpd.GeoDataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    y_col: str = "l8_year",
    m_col: str = "l8_bimonth_start_month",
    seed: int = 42,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    elig = g_master.dropna(subset=feature_cols + [y_col, m_col]).copy()
    elig[y_col] = elig[y_col].astype(int)
    elig[m_col] = elig[m_col].astype(int)

    pos = elig[elig[label_col] == 1].copy()
    neg = elig[elig[label_col] == 0].copy()

    selected_pos = []
    selected_neg = []
    used_neg = set()

    for (y, m), pos_g in pos.groupby([y_col, m_col]):
        n_need = len(pos_g)

        if n_need == 0:
            continue

        selected_pos.append(pos_g)

        ym_list = ym_neighbors(int(y), int(m))
        cand = neg[neg[[y_col, m_col]].apply(tuple, axis=1).isin(ym_list)].copy()

        if len(cand) == 0:
            if allow_replacement:
                cand = neg.copy()
            else:
                selected_pos[-1] = pos_g.iloc[0:0]
                continue

        cand_free = cand.loc[~cand.index.isin(used_neg)]

        if len(cand_free) >= n_need:
            chosen_idx = rng.choice(cand_free.index.to_numpy(), size=n_need, replace=False)
        else:
            if not allow_replacement:
                take = len(cand_free)
                selected_pos[-1] = pos_g.sample(n=take, random_state=seed) if take > 0 else pos_g.iloc[0:0]
                chosen_idx = cand_free.index.to_numpy()
            else:
                idx_list = cand_free.index.to_list()
                remaining = n_need - len(idx_list)
                extra_idx = rng.choice(cand.index.to_numpy(), size=remaining, replace=True)
                chosen_idx = np.array(idx_list + extra_idx.tolist())

        used_neg.update([i for i in chosen_idx if i in cand.index])
        selected_neg.append(neg.loc[chosen_idx].copy())

    df_pos = pd.concat(selected_pos, ignore_index=True) if selected_pos else pos.iloc[0:0].copy()
    df_neg = pd.concat(selected_neg, ignore_index=True) if selected_neg else neg.iloc[0:0].copy()

    out = pd.concat([df_pos, df_neg], ignore_index=True)

    c = out[label_col].value_counts().to_dict()

    assert c.get(1, 0) == c.get(0, 0), "No quedó balance 1:1 en Dataset A."

    return out.drop(columns=["geometry"], errors="ignore")
    

def build_dataset_C_balanced_exact(
    g_master_c: gpd.GeoDataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    y_col: str = "l8_year",
    m_col: str = "l8_bimonth_start_month",
    seed: int = 42,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    """
    Dataset C:
    - Caso completo.
    - Sin imputación.
    - Balance 1:1 presencia/ausencia.
    - Misma lógica de balanceo temporal usada en Dataset A.
    """
    return build_dataset_A_balanced_exact(
        g_master=g_master_c,
        feature_cols=feature_cols,
        label_col=label_col,
        y_col=y_col,
        m_col=m_col,
        seed=seed,
        allow_replacement=allow_replacement
    )


def impute_by_class_median(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    fallback: str = "global"
) -> pd.DataFrame:
    df = df.copy()

    for c in feature_cols:
        med_pos = df.loc[df[label_col] == 1, c].median(skipna=True)
        med_neg = df.loc[df[label_col] == 0, c].median(skipna=True)

        if fallback == "global":
            med_global = df[c].median(skipna=True)

            if pd.isna(med_pos):
                med_pos = med_global

            if pd.isna(med_neg):
                med_neg = med_global

        mask_pos = (df[label_col] == 1) & (df[c].isna())
        mask_neg = (df[label_col] == 0) & (df[c].isna())

        df.loc[mask_pos, c] = med_pos
        df.loc[mask_neg, c] = med_neg

    return df


def build_dataset_B_balanced_exact(
    g_master: gpd.GeoDataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    y_col: str = "l8_year",
    m_col: str = "l8_bimonth_start_month",
    seed: int = 42,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    df = g_master.dropna(subset=[y_col, m_col]).drop(columns=["geometry"], errors="ignore").copy()

    df = impute_by_class_median(
        df,
        feature_cols,
        label_col=label_col,
        fallback="global"
    )

    df[y_col] = df[y_col].astype(int)
    df[m_col] = df[m_col].astype(int)

    pos = df[df[label_col] == 1].copy()
    neg = df[df[label_col] == 0].copy()

    rng = np.random.default_rng(seed)
    used_neg = set()

    selected_pos = []
    selected_neg = []

    for (y, m), pos_g in pos.groupby([y_col, m_col]):
        n_need = len(pos_g)

        if n_need == 0:
            continue

        selected_pos.append(pos_g)

        ym_list = ym_neighbors(int(y), int(m))
        cand = neg[neg[[y_col, m_col]].apply(tuple, axis=1).isin(ym_list)].copy()

        if len(cand) == 0:
            cand = neg.copy()

        cand_free = cand.loc[~cand.index.isin(used_neg)]

        if len(cand_free) >= n_need:
            chosen_idx = rng.choice(cand_free.index.to_numpy(), size=n_need, replace=False)
        else:
            if not allow_replacement:
                take = len(cand_free)
                selected_pos[-1] = pos_g.sample(n=take, random_state=seed) if take > 0 else pos_g.iloc[0:0]
                chosen_idx = cand_free.index.to_numpy()
            else:
                idx_list = cand_free.index.to_list()
                remaining = n_need - len(idx_list)
                extra_idx = rng.choice(cand.index.to_numpy(), size=remaining, replace=True)
                chosen_idx = np.array(idx_list + extra_idx.tolist())

        used_neg.update([i for i in chosen_idx if i in cand.index])
        selected_neg.append(neg.loc[chosen_idx].copy())

    df_pos = pd.concat(selected_pos, ignore_index=True) if selected_pos else pos.iloc[0:0].copy()
    df_neg = pd.concat(selected_neg, ignore_index=True) if selected_neg else neg.iloc[0:0].copy()

    out = pd.concat([df_pos, df_neg], ignore_index=True)

    counts = out[label_col].value_counts().to_dict()

    assert counts.get(1, 0) == counts.get(0, 0), "Dataset B no quedó 1:1."

    return out


# =========================================================
# CV RF
# =========================================================

def run_grouped_cv_rf(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    group_col: str = "point_id",
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, pd.DataFrame, str]:
    if group_col not in df.columns:
        raise ValueError(f"Falta columna '{group_col}' en df.")

    X = df[feature_cols].to_numpy()
    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col].astype(int).to_numpy()

    n_unique_groups = len(np.unique(groups))

    if n_unique_groups < n_splits:
        raise ValueError(
            f"No hay suficientes grupos únicos para {n_splits}-fold CV. "
            f"Grupos únicos disponibles: {n_unique_groups}"
        )

    if HAS_SGK:
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
        split_iter = cv.split(X, y, groups=groups)
        cv_name = "StratifiedGroupKFold"
    else:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, y, groups=groups)
        cv_name = "GroupKFold"

    fold_rows = []
    cms = []
    reports = []
    importances = []

    for fold, (tr, te) in enumerate(split_iter, start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )

        model.fit(Xtr, ytr)

        yp = model.predict(Xte)

        cm = confusion_matrix(yte, yp, labels=[0, 1])
        cms.append(cm)

        fold_rows.append({
            "fold": fold,
            "cv": cv_name,
            "n_train": len(tr),
            "n_test": len(te),
            "balanced_accuracy": balanced_accuracy_score(yte, yp),
            "precision_pos": precision_score(yte, yp, pos_label=1, zero_division=0),
            "recall_pos": recall_score(yte, yp, pos_label=1, zero_division=0),
            "f1_pos": f1_score(yte, yp, pos_label=1, zero_division=0),
            "precision_neg": precision_score(yte, yp, pos_label=0, zero_division=0),
            "recall_neg": recall_score(yte, yp, pos_label=0, zero_division=0),
            "f1_neg": f1_score(yte, yp, pos_label=0, zero_division=0),
            "tn": cm[0, 0],
            "fp": cm[0, 1],
            "fn": cm[1, 0],
            "tp": cm[1, 1],
        })

        reports.append(
            f"--- Fold {fold} ({cv_name}) ---\n"
            + classification_report(
                yte,
                yp,
                target_names=["absence(0)", "presence(1)"],
                zero_division=0
            )
        )

        importances.append(model.feature_importances_)

    df_folds = pd.DataFrame(fold_rows)

    cm_sum = np.sum(np.stack(cms, axis=0), axis=0)

    metric_cols = [
        "balanced_accuracy",
        "precision_pos",
        "recall_pos",
        "f1_pos",
        "precision_neg",
        "recall_neg",
        "f1_neg"
    ]

    summary = df_folds[metric_cols].agg(["mean", "std"]).T

    imp_arr = np.vstack(importances)

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": imp_arr.mean(axis=0),
        "importance_std": imp_arr.std(axis=0)
    }).sort_values("importance_mean", ascending=False)

    final_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    final_model.fit(X, y)

    txt = []
    txt.append(f"========== CV REPORT ({cv_name}) fold-by-fold ==========\n")
    txt.extend([r + "\n" for r in reports])
    txt.append("\n========== METRICS PER FOLD ==========\n")
    txt.append(df_folds.to_string(index=False))
    txt.append("\n\n========== SUMMARY (mean ± std) ==========\n")
    txt.append(summary.to_string())
    txt.append("\n\n========== AGGREGATED CONFUSION MATRIX (sum over folds) ==========\n")
    txt.append(str(cm_sum))
    txt.append("\n\n========== FEATURE IMPORTANCE (mean ± std over folds) ==========\n")
    txt.append(imp_df.to_string(index=False))

    return final_model, df_folds, "\n".join(txt)


# =========================================================
# EXPORT
# =========================================================

def _dedupe_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    seen = {}
    new_cols = []

    for c in cols:
        key = c.lower()

        if key not in seen:
            seen[key] = 1
            new_cols.append(c)
        else:
            seen[key] += 1
            new_cols.append(f"{c}_{seen[key]}")

    df = df.copy()
    df.columns = new_cols

    return df


def export_outputs(
    g_master: gpd.GeoDataFrame,
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    modelA,
    modelB,
    reportA: str,
    reportB: str,
    out_gpkg: str,
    out_layer: str,
    out_csv_a: str,
    out_csv_b: str,
    out_model_a: str,
    out_model_b: str,
    out_report: str,
):
    g_master_out = g_master.copy()

    if "acq_date" in g_master_out.columns:
        g_master_out["acq_date"] = pd.to_datetime(
            g_master_out["acq_date"],
            errors="coerce"
        )

        try:
            g_master_out["acq_date"] = g_master_out["acq_date"].dt.tz_localize(None)
        except Exception:
            pass

    if "s1_date" in g_master_out.columns:
        g_master_out["s1_date"] = pd.to_datetime(
            g_master_out["s1_date"],
            errors="coerce"
        )

        try:
            g_master_out["s1_date"] = g_master_out["s1_date"].dt.tz_localize(None)
        except Exception:
            pass

    for c in g_master_out.columns:
        if c != "geometry" and g_master_out[c].dtype == "object":
            g_master_out[c] = g_master_out[c].astype(str)

    for cand in ["Label", "LABEL"]:
        if cand in g_master_out.columns:
            g_master_out = g_master_out.rename(columns={cand: f"{cand}_src"})

    g_master_out = _dedupe_columns_case_insensitive(g_master_out)

    g_master_out.to_file(out_gpkg, layer=out_layer, driver="GPKG")

    dfA.to_csv(out_csv_a, index=False)
    dfB.to_csv(out_csv_b, index=False)

    joblib.dump(modelA, out_model_a)
    joblib.dump(modelB, out_model_b)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("########################\n# DATASET A\n########################\n\n")
        f.write(reportA)
        f.write("\n\n\n########################\n# DATASET B\n########################\n\n")
        f.write(reportB)


def export_outputs_model_c(
    g_master_c: gpd.GeoDataFrame,
    dfC: pd.DataFrame,
    modelC,
    reportC: str,
    out_gpkg_c: str,
    out_layer_c: str,
    out_csv_c: str,
    out_model_c: str,
    out_report_c: str,
):
    g_out = g_master_c.copy()

    date_cols = [
        "acq_date",
        "s1_general_date",
        "s1_difference_date"
    ]

    for dc in date_cols:
        if dc in g_out.columns:
            g_out[dc] = pd.to_datetime(g_out[dc], errors="coerce")

            try:
                g_out[dc] = g_out[dc].dt.tz_localize(None)
            except Exception:
                pass

    for c in g_out.columns:
        if c != "geometry" and g_out[c].dtype == "object":
            g_out[c] = g_out[c].astype(str)

    for cand in ["Label", "LABEL"]:
        if cand in g_out.columns:
            g_out = g_out.rename(columns={cand: f"{cand}_src"})

    g_out = _dedupe_columns_case_insensitive(g_out)

    g_out.to_file(out_gpkg_c, layer=out_layer_c, driver="GPKG")

    dfC.to_csv(out_csv_c, index=False)

    joblib.dump(modelC, out_model_c)

    with open(out_report_c, "w", encoding="utf-8") as f:
        f.write("########################\n")
        f.write("# DATASET C - MODELO C\n")
        f.write("########################\n\n")
        f.write(reportC)


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def run_pipeline(cfg, logger: logging.Logger):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    l8_dir = str(cfg.l8_dir)
    s1_dir = str(cfg.s1_dir)

    logger.info("=================================================")
    logger.info("INICIO PIPELINE RANDOM FOREST")
    logger.info("=================================================")

    logger.info(f"Ruta incendios: {cfg.path_fire_shp}")
    logger.info(f"Ruta ausencias: {cfg.path_abs_shp}")
    logger.info(f"Ruta L8 normalizado: {l8_dir}")
    logger.info(f"Ruta S1 normalizado: {s1_dir}")
    logger.info(f"OUT_DIR: {cfg.out_dir}")

    g_points = load_points(
        path_fire_shp=str(cfg.path_fire_shp),
        path_abs_shp=str(cfg.path_abs_shp),
        date_col=cfg.date_col,
        epsg_work=cfg.epsg_work,
    )

    logger.info(f"CRS puntos combinados: {g_points.crs}")
    logger.info(f"Incendios: {(g_points['label'] == 1).sum()}")
    logger.info(f"Ausencias: {(g_points['label'] == 0).sum()}")
    logger.info(f"Total puntos: {len(g_points)}")

    g_points = assign_dates_to_absences(
        g_points,
        date_col=cfg.date_col,
        seed=cfg.seed_abs_dates
    )

    l8_index = index_l8_rasters(l8_dir, logger=logger)

    s1_index = index_s1_rasters_from_filenames(
        s1_dir=s1_dir,
        logger=logger
    )

    logger.info(f"[L8] Bimestres indexados: {len(l8_index)}")
    logger.info(f"[S1] Escenas indexadas: {len(s1_index)}")

    if len(l8_index) == 0:
        raise ValueError(
            "No se indexó ningún TIFF Landsat 8. "
            "Revisa la ruta l8_dir y el patrón L8_YYYY_ene_feb_normalizado.tif."
        )

    if len(s1_index) == 0:
        raise ValueError(
            "No se indexó ningún TIFF Sentinel-1. "
            "Revisa la ruta s1_dir y el patrón S1_YYYY-MM-DD_idx_prepost_indices_normalizado.tif."
        )

    g_master = build_master_features(
        g=g_points,
        l8_index=l8_index,
        s1_df=s1_index,
        date_col=cfg.date_col,
        l8_switch_day=cfg.l8_switch_day,
        l8_fallback_next_available=cfg.l8_fallback_next_available,
        s1_start_offset_days=cfg.s1_start_offset_days,
        s1_window_days=cfg.s1_window_days,
        logger=logger
    )

    cov = coverage_report(g_master, FEATURES)

    logger.info("Coverage report:\n" + cov.to_string(index=False))

    dfA = build_dataset_A_balanced_exact(
        g_master,
        feature_cols=FEATURES,
        seed=cfg.seed_dataset_a,
        allow_replacement=True
    )

    dfB = build_dataset_B_balanced_exact(
        g_master,
        feature_cols=FEATURES,
        seed=cfg.seed_dataset_b,
        allow_replacement=True
    )

    logger.info(f"Dataset A shape: {dfA.shape}")
    logger.info(f"Dataset B shape: {dfB.shape}")

    print("=== DEBUG PRE-CV ===")
    print("g_points:", g_points.shape)
    print("g_master:", g_master.shape)
    print("FEATURES:", FEATURES)

    print("\nConteo label en g_master:")
    print(g_master["label"].value_counts(dropna=False))

    print("\nL8 found:")
    print(g_master["l8_found"].value_counts(dropna=False))

    print("\nS1 found:")
    print(g_master["s1_found"].value_counts(dropna=False))

    print("\nNaN por feature en g_master:")
    print(g_master[FEATURES].isna().sum())

    print("\nComplete cases en FEATURES:")
    tmp_complete = g_master.dropna(subset=FEATURES)
    print(tmp_complete.shape)

    if len(tmp_complete) > 0:
        print(tmp_complete["label"].value_counts(dropna=False))

    print("\ndfA shape:", dfA.shape)

    if len(dfA) > 0:
        print(dfA["label"].value_counts(dropna=False))
        print("point_id únicos en dfA:", dfA["point_id"].nunique())

    print("\ndfB shape:", dfB.shape)

    if len(dfB) > 0:
        print(dfB["label"].value_counts(dropna=False))
        print("point_id únicos en dfB:", dfB["point_id"].nunique())

    # =========================
    # VALIDACIONES PREVIAS A CV
    # =========================

    if dfA.empty:
        raise ValueError(
            "dfA quedó vacío antes de CV. "
            "Revisa cobertura completa de FEATURES, disponibilidad L8/S1 y balanceo temporal."
        )

    if dfB.empty:
        raise ValueError(
            "dfB quedó vacío antes de CV. "
            "Revisa imputación, columnas temporales y balanceo."
        )

    if dfA["label"].nunique() < 2:
        raise ValueError(
            f"dfA no tiene ambas clases. "
            f"Conteo actual: {dfA['label'].value_counts(dropna=False).to_dict()}"
        )

    if dfB["label"].nunique() < 2:
        raise ValueError(
            f"dfB no tiene ambas clases. "
            f"Conteo actual: {dfB['label'].value_counts(dropna=False).to_dict()}"
        )

    if dfA["point_id"].nunique() < 10:
        raise ValueError(
            f"dfA tiene menos de 10 grupos únicos para CV 10-fold. "
            f"point_id únicos: {dfA['point_id'].nunique()}"
        )

    if dfB["point_id"].nunique() < 10:
        raise ValueError(
            f"dfB tiene menos de 10 grupos únicos para CV 10-fold. "
            f"point_id únicos: {dfB['point_id'].nunique()}"
        )

    # =========================
    # ENTRENAMIENTO / CV
    # =========================

    modelA, foldsA, reportA = run_grouped_cv_rf(
        dfA,
        FEATURES,
        label_col="label",
        group_col="point_id",
        n_splits=10,
        random_state=cfg.seed_cv
    )

    modelB, foldsB, reportB = run_grouped_cv_rf(
        dfB,
        FEATURES,
        label_col="label",
        group_col="point_id",
        n_splits=10,
        random_state=cfg.seed_cv
    )

    # =========================
    # EXPORTACIÓN
    # =========================

    export_outputs(
        g_master=g_master,
        dfA=dfA,
        dfB=dfB,
        modelA=modelA,
        modelB=modelB,
        reportA=reportA,
        reportB=reportB,
        out_gpkg=str(cfg.out_dir / cfg.out_gpkg),
        out_layer=cfg.out_layer,
        out_csv_a=str(cfg.out_dir / cfg.out_csv_a),
        out_csv_b=str(cfg.out_dir / cfg.out_csv_b),
        out_model_a=str(cfg.out_dir / cfg.out_model_a),
        out_model_b=str(cfg.out_dir / cfg.out_model_b),
        out_report=str(cfg.out_dir / cfg.out_report),
    )

    logger.info(f"Outputs generados en: {cfg.out_dir}")

    logger.info("=================================================")
    logger.info("FIN PIPELINE RANDOM FOREST")
    logger.info("=================================================")

    return {
        "g_points": g_points,
        "g_master": g_master,
        "dfA": dfA,
        "dfB": dfB,
        "foldsA": foldsA,
        "foldsB": foldsB,
        "features": FEATURES,
        "coverage": cov,
        "out_dir": str(cfg.out_dir),
    }

def run_pipeline_model_c(cfg, logger: logging.Logger):
    """
    Ejecuta Modelo C.

    Modelo C:
    - Derivado del Modelo A.
    - Caso completo.
    - Sin imputación.
    - Balance 1:1.
    - Diferencia clave:
      las bandas VV_Difference_norm, VH_Difference_norm y VHVV_Difference_norm
      se extraen desde una imagen Sentinel-1 del mismo mes del incendio.
    """
    if not cfg.model_c_enabled:
        raise ValueError("El Modelo C está desactivado en la configuración.")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    l8_dir = str(cfg.l8_dir)
    s1_dir = str(cfg.s1_dir)

    logger.info("=================================================")
    logger.info("INICIO PIPELINE MODELO C")
    logger.info("=================================================")

    logger.info(f"[MODELO C] Ruta incendios: {cfg.path_fire_shp}")
    logger.info(f"[MODELO C] Ruta ausencias: {cfg.path_abs_shp}")
    logger.info(f"[MODELO C] Ruta L8 normalizado: {l8_dir}")
    logger.info(f"[MODELO C] Ruta S1 normalizado: {s1_dir}")
    logger.info(f"[MODELO C] OUT_DIR: {cfg.out_dir}")
    logger.info(f"[MODELO C] Estrategia diferencias S1: {cfg.s1_difference_selection_strategy}")

    # =====================================================
    # CARGAR PUNTOS
    # =====================================================
    g_points = load_points(
        path_fire_shp=str(cfg.path_fire_shp),
        path_abs_shp=str(cfg.path_abs_shp),
        date_col=cfg.date_col,
        epsg_work=cfg.epsg_work,
    )

    logger.info(f"[MODELO C] CRS puntos combinados: {g_points.crs}")
    logger.info(f"[MODELO C] Incendios: {(g_points['label'] == 1).sum()}")
    logger.info(f"[MODELO C] Ausencias: {(g_points['label'] == 0).sum()}")
    logger.info(f"[MODELO C] Total puntos: {len(g_points)}")

    # Las ausencias usan fecha asignada igual que A/B.
    g_points = assign_dates_to_absences(
        g_points,
        date_col=cfg.date_col,
        seed=cfg.seed_abs_dates
    )

    # =====================================================
    # INDEXAR RASTERS
    # =====================================================
    l8_index = index_l8_rasters(l8_dir, logger=logger)

    s1_index = index_s1_rasters_from_filenames(
        s1_dir=s1_dir,
        logger=logger
    )

    logger.info(f"[MODELO C][L8] Bimestres indexados: {len(l8_index)}")
    logger.info(f"[MODELO C][S1] Escenas indexadas: {len(s1_index)}")

    if len(l8_index) == 0:
        raise ValueError(
            "[MODELO C] No se indexó ningún TIFF Landsat 8. "
            "Revisa l8_dir y el patrón L8_YYYY_ene_feb_normalizado.tif."
        )

    if len(s1_index) == 0:
        raise ValueError(
            "[MODELO C] No se indexó ningún TIFF Sentinel-1. "
            "Revisa s1_dir y el patrón S1_YYYY-MM-DD_idx_prepost_indices_normalizado.tif."
        )

    # =====================================================
    # CONSTRUIR FEATURES MODELO C
    # =====================================================
    g_master_c = build_master_features_model_c(
        g=g_points,
        l8_index=l8_index,
        s1_df=s1_index,
        date_col=cfg.date_col,
        l8_switch_day=cfg.l8_switch_day,
        l8_fallback_next_available=cfg.l8_fallback_next_available,
        s1_start_offset_days=cfg.s1_start_offset_days,
        s1_window_days=cfg.s1_window_days,
        s1_difference_selection_strategy=cfg.s1_difference_selection_strategy,
        logger=logger
    )

    coverageC = coverage_report(g_master_c, FEATURES_C)

    logger.info("[MODELO C] Coverage report:\n" + coverageC.to_string(index=False))

    # =====================================================
    # DATASET C
    # =====================================================
    dfC = build_dataset_C_balanced_exact(
        g_master_c=g_master_c,
        feature_cols=FEATURES_C,
        seed=cfg.seed_dataset_c,
        allow_replacement=True
    )

    logger.info(f"[MODELO C] Dataset C shape: {dfC.shape}")

    print("=== DEBUG PRE-CV MODELO C ===")
    print("g_points:", g_points.shape)
    print("g_master_c:", g_master_c.shape)
    print("FEATURES_C:", FEATURES_C)

    print("\nConteo label en g_master_c:")
    print(g_master_c["label"].value_counts(dropna=False))

    print("\nL8 found:")
    print(g_master_c["l8_found"].value_counts(dropna=False))

    print("\nS1 general found:")
    print(g_master_c["s1_general_found"].value_counts(dropna=False))

    print("\nS1 difference found:")
    print(g_master_c["s1_difference_found"].value_counts(dropna=False))

    print("\nNaN por feature en g_master_c:")
    print(g_master_c[FEATURES_C].isna().sum())

    print("\nComplete cases en FEATURES_C:")
    tmp_complete = g_master_c.dropna(subset=FEATURES_C)
    print(tmp_complete.shape)

    if len(tmp_complete) > 0:
        print(tmp_complete["label"].value_counts(dropna=False))

    print("\ndfC shape:", dfC.shape)

    if len(dfC) > 0:
        print(dfC["label"].value_counts(dropna=False))
        print("point_id únicos en dfC:", dfC["point_id"].nunique())

    # =====================================================
    # VALIDACIONES PREVIAS A CV
    # =====================================================
    if dfC.empty:
        raise ValueError(
            "[MODELO C] dfC quedó vacío antes de CV. "
            "Revisa cobertura de FEATURES_C y disponibilidad de S1 mismo mes."
        )

    if dfC["label"].nunique() < 2:
        raise ValueError(
            f"[MODELO C] dfC no tiene ambas clases. "
            f"Conteo actual: {dfC['label'].value_counts(dropna=False).to_dict()}"
        )

    if dfC["point_id"].nunique() < 10:
        raise ValueError(
            f"[MODELO C] dfC tiene menos de 10 grupos únicos para CV 10-fold. "
            f"point_id únicos: {dfC['point_id'].nunique()}"
        )

    # =====================================================
    # ENTRENAMIENTO / CV MODELO C
    # =====================================================
    modelC, foldsC, reportC = run_grouped_cv_rf(
        dfC,
        FEATURES_C,
        label_col="label",
        group_col="point_id",
        n_splits=10,
        random_state=cfg.seed_cv
    )

    # =====================================================
    # EXPORTACIÓN MODELO C
    # =====================================================
    export_outputs_model_c(
        g_master_c=g_master_c,
        dfC=dfC,
        modelC=modelC,
        reportC=reportC,
        out_gpkg_c=str(cfg.out_dir / cfg.out_gpkg_c),
        out_layer_c=cfg.out_layer_c,
        out_csv_c=str(cfg.out_dir / cfg.out_csv_c),
        out_model_c=str(cfg.out_dir / cfg.out_model_c),
        out_report_c=str(cfg.out_dir / cfg.out_report_c),
    )

    logger.info(f"[MODELO C] Outputs generados en: {cfg.out_dir}")

    logger.info("=================================================")
    logger.info("FIN PIPELINE MODELO C")
    logger.info("=================================================")

    return {
        "g_points": g_points,
        "g_master_c": g_master_c,
        "dfC": dfC,
        "foldsC": foldsC,
        "featuresC": FEATURES_C,
        "coverageC": coverageC,
        "out_dir": str(cfg.out_dir),
    }