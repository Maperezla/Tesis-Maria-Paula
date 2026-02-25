import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict

import pandas as pd


@dataclass
class UNGRDSummary:
    n_rows_in: int
    n_rows_filtered: int
    n_nat_dates: int
    out_path: str


def setup_logger(log_path: str, name: str = "ungrd_filter") -> logging.Logger:
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
    Versiona: archivo.xlsx -> archivo_v002.xlsx -> ...
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


def read_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return pd.read_excel(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    return df


def add_date_parts(
    df: pd.DataFrame,
    date_col: str = "FECHA",
    dayfirst: bool = True,
    year_col: str = "año_UNGRD",
    month_col: str = "mes_UNGRD",
    day_col: str = "dia_UNGRD",
) -> Tuple[pd.DataFrame, int]:
    if date_col not in df.columns:
        raise KeyError(f"Falta la columna requerida: {date_col}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=dayfirst, errors="coerce")

    n_nat = int(df[date_col].isna().sum())

    # Nullable Int64 para no romper si hay NaT
    df[year_col] = df[date_col].dt.year.astype("Int64")
    df[month_col] = df[date_col].dt.month.astype("Int64")
    df[day_col] = df[date_col].dt.day.astype("Int64")

    return df, n_nat


def normalize_text_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        raise KeyError(f"Falta la columna requerida: {col}")
    df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def normalize_categories(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        df2 = normalize_text_col(df2, c)
    return df2


def year_counts(df: pd.DataFrame, year_col: str = "año_UNGRD") -> pd.Series:
    if year_col not in df.columns:
        raise KeyError(f"Falta la columna: {year_col}")
    return df.groupby(year_col).size().rename("registros_totales")


def top_values(df: pd.DataFrame, col: str, k: int = 20) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Falta la columna: {col}")
    return df[col].value_counts().head(k)


def filter_events_deptos(
    df: pd.DataFrame,
    eventos: Sequence[str],
    deptos: Sequence[str],
    col_evento: str = "EVENTO",
    col_depto: str = "DEPARTAMENTO",
) -> pd.DataFrame:
    # Asegurar normalización previa (mayúsculas/strip)
    missing = [c for c in [col_evento, col_depto] if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para filtrar: {missing}")

    ev = [str(x).strip().upper() for x in eventos]
    dp = [str(x).strip().upper() for x in deptos]

    return df[(df[col_evento].isin(ev)) & (df[col_depto].isin(dp))].copy()


def build_year_summary(
    total_counts: pd.Series,
    filtered_counts: pd.Series,
) -> pd.DataFrame:
    tabla = pd.concat([total_counts, filtered_counts.rename("registros_filtrados")], axis=1).fillna(0)
    tabla["registros_totales"] = tabla["registros_totales"].astype(int)
    tabla["registros_filtrados"] = tabla["registros_filtrados"].astype(int)
    return tabla.sort_index()


def write_excel(df: pd.DataFrame, out_path: str, version: bool = True) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final_path = next_version_path(out_path) if version else out_path
    df.to_excel(final_path, index=False)
    return final_path


def run_ungrd_pipeline(
    in_path: str,
    out_path: str,
    eventos_interes: Sequence[str],
    deptos_interes: Sequence[str],
    dayfirst: bool = True,
    version_out: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, UNGRDSummary]:
    df = read_excel(in_path)
    df = clean_columns(df)

    df, n_nat = add_date_parts(df, date_col="FECHA", dayfirst=dayfirst)
    df = normalize_categories(df, cols=["EVENTO", "DEPARTAMENTO"])

    total_counts = year_counts(df, year_col="año_UNGRD")

    df_filtered = filter_events_deptos(
        df, eventos=eventos_interes, deptos=deptos_interes,
        col_evento="EVENTO", col_depto="DEPARTAMENTO"
    )
    filtered_counts = year_counts(df_filtered, year_col="año_UNGRD")

    year_table = build_year_summary(total_counts, filtered_counts)

    final_out = write_excel(df_filtered, out_path, version=version_out)

    summary = UNGRDSummary(
        n_rows_in=len(df),
        n_rows_filtered=len(df_filtered),
        n_nat_dates=n_nat,
        out_path=final_out
    )

    if logger:
        logger.info(f"Entrada: {in_path} | filas={summary.n_rows_in}")
        logger.info(f"NaT en FECHA: {summary.n_nat_dates}")
        logger.info(f"Filtradas: {summary.n_rows_filtered}")
        logger.info(f"Salida: {summary.out_path}")

    return df_filtered, year_table, summary


def format_report(summary: UNGRDSummary) -> str:
    return (
        "\n--- REPORTE FINAL (UNGRD FILTRO) ---\n"
        f"📊 Filas entrada: {summary.n_rows_in}\n"
        f"✅ Filas filtradas: {summary.n_rows_filtered}\n"
        f"⚠️ FECHA no interpretable (NaT): {summary.n_nat_dates}\n"
        f"💾 Salida: {summary.out_path}\n"
    )