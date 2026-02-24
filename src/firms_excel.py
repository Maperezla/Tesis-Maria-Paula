import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


@dataclass
class ExcelSummary:
    n_rows_in: int
    n_rows_out: int
    n_null_dates: int
    out_path: str


def setup_logger(log_path: str, name: str = "firms_excel") -> logging.Logger:
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


def read_firms_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return pd.read_excel(path)


def add_year_month(
    df: pd.DataFrame,
    date_col: str = "ACQ_DATE",
    date_format: Optional[str] = "%Y/%m/%d",
    year_col: str = "ano_FIRMS",
    month_col: str = "mes_FIRMS",
) -> Tuple[pd.DataFrame, int]:
    if date_col not in df.columns:
        raise ValueError(f"Falta la columna requerida: {date_col}")

    # Parse robusto: si date_format falla, deja NaT
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="coerce")
    n_null = int(df[date_col].isna().sum())

    df[year_col] = df[date_col].dt.year
    df[month_col] = df[date_col].dt.month

    return df, n_null


def write_excel(df: pd.DataFrame, out_path: str, version: bool = True) -> str:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    final_path = next_version_path(out_path) if version else out_path
    df.to_excel(final_path, index=False)
    return final_path


def process_firms_excel(
    in_path: str,
    out_path: str,
    date_col: str = "ACQ_DATE",
    date_format: Optional[str] = "%Y/%m/%d",
    version_out: bool = True,
    logger: Optional[logging.Logger] = None,
) -> ExcelSummary:
    df = read_firms_excel(in_path)
    n_in = len(df)

    df2, n_null_dates = add_year_month(df, date_col=date_col, date_format=date_format)

    final_path = write_excel(df2, out_path, version=version_out)

    if logger:
        logger.info(f"Entrada: {in_path} | filas={n_in}")
        logger.info(f"Salida: {final_path} | filas={len(df2)} | fechas NaT={n_null_dates}")

    return ExcelSummary(
        n_rows_in=n_in,
        n_rows_out=len(df2),
        n_null_dates=n_null_dates,
        out_path=final_path,
    )