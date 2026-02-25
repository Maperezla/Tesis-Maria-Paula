import os
import re
import unicodedata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import pandas as pd
import geopandas as gpd


# -------------------------
# Summary
# -------------------------
@dataclass
class MatchSummary:
    n_points: int
    n_ungrd_rows: int
    n_candidates_after_admin_merge: int
    n_points_with_any_ungrd: int
    n_exact: int
    n_window: int
    n_none: int
    out_path: str


# -------------------------
# Logging + versioning
# -------------------------
def setup_logger(log_path: str, name: str = "firms_ungrd_match") -> logging.Logger:
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
    """
    Mantiene tu estilo: _1, _2, ...
    """
    if not os.path.exists(out_path):
        return out_path
    base, ext = os.path.splitext(out_path)
    k = 1
    while True:
        cand = f"{base}_{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1


# -------------------------
# Utilities
# -------------------------
def ensure_columns(df, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas en {name}: {missing}")


def normalize_text(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).upper().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_series(s: pd.Series) -> pd.Series:
    return s.apply(normalize_text)


# -------------------------
# Loaders
# -------------------------
def load_ungrd_excel(
    excel_path: str,
    col_depto: str = "DEPARTAMENTO",
    col_muni: str = "MUNICIPIO",
    col_fecha: str = "FECHA",
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"No existe Excel UNGRD: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = df.columns.astype(str).str.strip()

    ensure_columns(df, [col_depto, col_muni, col_fecha], "UNGRD Excel")

    # Normalizar admin
    df["DEPTO_std"] = normalize_series(df[col_depto])
    df["MUNICIPIO_std"] = normalize_series(df[col_muni])

    # Asegurar fecha datetime (robusto)
    df[col_fecha] = pd.to_datetime(df[col_fecha], dayfirst=True, errors="coerce")

    if logger:
        logger.info(f"UNGRD leído: {excel_path} | n={len(df)} | NaT_FECHA={int(df[col_fecha].isna().sum())}")

    return df


def load_firms_shp(
    shp_path: str,
    col_depto_std: str = "depto_std",
    col_muni_raw: str = "mpnombre_s",   # ojo: nombres truncados en SHP
    col_year: str = "anio",
    col_month: str = "mes",
    col_day: str = "dia",
    logger: Optional[logging.Logger] = None
) -> gpd.GeoDataFrame:
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"No existe SHP FIRMS: {shp_path}")

    gdf = gpd.read_file(shp_path)

    ensure_columns(gdf, [col_depto_std, col_muni_raw, col_year, col_month, col_day], "FIRMS SHP")

    # Tipos seguros (nullable)
    gdf[col_year] = pd.to_numeric(gdf[col_year], errors="coerce").astype("Int64")
    gdf[col_month] = pd.to_numeric(gdf[col_month], errors="coerce").astype("Int64")
    gdf[col_day] = pd.to_numeric(gdf[col_day], errors="coerce").astype("Int64")

    gdf["fecha_FIRMS"] = pd.to_datetime(
        dict(year=gdf[col_year], month=gdf[col_month], day=gdf[col_day]),
        errors="coerce"
    )

    gdf["depto_std"] = normalize_series(gdf[col_depto_std])
    gdf["mpnombre_s_std"] = normalize_series(gdf[col_muni_raw])

    if logger:
        logger.info(f"FIRMS leído: {shp_path} | n={len(gdf)} | NaT_fecha_FIRMS={int(gdf['fecha_FIRMS'].isna().sum())}")

    return gdf


# -------------------------
# Matching core
# -------------------------
def build_candidates(
    gdf_firms: gpd.GeoDataFrame,
    df_ungrd: pd.DataFrame,
    ungrd_fecha_col: str = "FECHA"
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Crea candidatos por (depto, muni). Devuelve:
    - merged candidates (DataFrame)
    - gdf_firms_reset (GeoDataFrame con idx_firms)
    """
    gdf_firms_reset = gdf_firms.reset_index(drop=True).reset_index().rename(columns={"index": "idx_firms"})

    firms_subset = gdf_firms_reset[["idx_firms", "depto_std", "mpnombre_s_std", "fecha_FIRMS"]].copy()
    ungrd_subset = df_ungrd[["DEPTO_std", "MUNICIPIO_std", ungrd_fecha_col]].copy()

    merged = firms_subset.merge(
        ungrd_subset,
        left_on=["depto_std", "mpnombre_s_std"],
        right_on=["DEPTO_std", "MUNICIPIO_std"],
        how="left",
        suffixes=("_FIRMS", "_UNGRD")
    )

    return merged, gdf_firms_reset


def best_match_per_point(
    merged: pd.DataFrame,
    ungrd_fecha_col: str = "FECHA",
    window_days: int = 5
) -> pd.DataFrame:
    """
    Selecciona 1 match por idx_firms: el de menor |dif_dias|.
    Clasifica: exacta / ventana / sin_coincidencia.
    """
    merged = merged.copy()
    merged["dif_dias"] = (merged["fecha_FIRMS"] - merged[ungrd_fecha_col]).dt.days
    merged["dif_dias_abs"] = merged["dif_dias"].abs()

    # Solo filas con fecha UNGRD válida
    valid = merged[merged[ungrd_fecha_col].notna()].copy()
    if len(valid) == 0:
        # Sin UNGRD válido para nadie
        out = merged[["idx_firms"]].drop_duplicates().copy()
        out["fecha_UNGRD_match"] = pd.NaT
        out["dif_dias_min"] = pd.NA
        out["dif_dias_abs"] = pd.NA
        out["tipo_coincidencia"] = "sin_coincidencia"
        return out

    # Ordenar por punto y abs diff
    valid = valid.sort_values(["idx_firms", "dif_dias_abs"])

    # Elegir el primero por idx_firms
    best = valid.groupby("idx_firms").first().reset_index()

    best["coincidencia_exacta"] = best["dif_dias"] == 0
    best["coincidencia_ventana"] = (~best["coincidencia_exacta"]) & (best["dif_dias_abs"] <= window_days)

    def tipo(row):
        if bool(row["coincidencia_exacta"]):
            return "exacta"
        if bool(row["coincidencia_ventana"]):
            return f"ventana_{window_days}dias"
        return "sin_coincidencia"

    best["tipo_coincidencia"] = best.apply(tipo, axis=1)

    out = best[["idx_firms", ungrd_fecha_col, "dif_dias", "dif_dias_abs", "tipo_coincidencia"]].copy()
    out = out.rename(columns={ungrd_fecha_col: "fecha_UNGRD_match", "dif_dias": "dif_dias_min"})
    return out


def attach_matches(
    gdf_firms_reset: gpd.GeoDataFrame,
    matches: pd.DataFrame
) -> gpd.GeoDataFrame:
    # Asegurar que todos los puntos queden
    base = pd.DataFrame({"idx_firms": gdf_firms_reset["idx_firms"]})
    m = base.merge(matches, on="idx_firms", how="left")

    # Relleno seguro
    m["tipo_coincidencia"] = m["tipo_coincidencia"].fillna("sin_coincidencia")

    gdf_final = gdf_firms_reset.merge(m, on="idx_firms", how="left")
    return gpd.GeoDataFrame(gdf_final, geometry="geometry", crs=gdf_firms_reset.crs)


def write_shp(gdf: gpd.GeoDataFrame, out_path: str, version: bool = True,
              logger: Optional[logging.Logger] = None) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final = next_version_path_suffix_1(out_path) if version else out_path
    gdf.to_file(final, driver="ESRI Shapefile", encoding="utf-8")
    if logger:
        logger.info(f"Salida SHP: {final} | n={len(gdf)}")
    return final


def run_firms_vs_ungrd(
    ungrd_excel: str,
    firms_shp: str,
    out_shp: str,
    window_days: int = 5,
    logger: Optional[logging.Logger] = None
) -> Tuple[gpd.GeoDataFrame, MatchSummary]:
    df_ungrd = load_ungrd_excel(ungrd_excel, logger=logger)
    gdf_firms = load_firms_shp(firms_shp, logger=logger)

    merged, gdf_firms_reset = build_candidates(gdf_firms, df_ungrd, ungrd_fecha_col="FECHA")

    matches = best_match_per_point(merged, ungrd_fecha_col="FECHA", window_days=window_days)
    gdf_out = attach_matches(gdf_firms_reset, matches)

    final_path = write_shp(gdf_out, out_shp, version=True, logger=logger)

    n_points = len(gdf_out)
    n_exact = int((gdf_out["tipo_coincidencia"] == "exacta").sum())
    n_window = int((gdf_out["tipo_coincidencia"] == f"ventana_{window_days}dias").sum())
    n_none = int((gdf_out["tipo_coincidencia"] == "sin_coincidencia").sum())

    # puntos con algún ungrd (match con fecha no nula)
    n_with_any = int(gdf_out["fecha_UNGRD_match"].notna().sum())

    summary = MatchSummary(
        n_points=n_points,
        n_ungrd_rows=len(df_ungrd),
        n_candidates_after_admin_merge=len(merged),
        n_points_with_any_ungrd=n_with_any,
        n_exact=n_exact,
        n_window=n_window,
        n_none=n_none,
        out_path=final_path
    )

    if logger:
        logger.info(f"SUMMARY: {summary}")

    return gdf_out, summary


def format_report(summary: MatchSummary, window_days: int = 5) -> str:
    return (
        "\n--- REPORTE FINAL (FIRMS vs UNGRD) ---\n"
        f"📍 Puntos FIRMS: {summary.n_points}\n"
        f"📄 Filas UNGRD: {summary.n_ungrd_rows}\n"
        f"🔗 Candidatos tras merge admin: {summary.n_candidates_after_admin_merge}\n"
        f"🧩 Puntos con algún match UNGRD: {summary.n_points_with_any_ungrd}\n"
        f"✅ Exacta: {summary.n_exact}\n"
        f"🟨 Ventana ±{window_days}d: {summary.n_window}\n"
        f"❌ Sin coincidencia: {summary.n_none}\n"
        f"💾 Salida: {summary.out_path}\n"
    )