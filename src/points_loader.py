import numpy as np
import pandas as pd
import geopandas as gpd


def load_points(
    path_fire_shp: str,
    path_abs_shp: str,
    date_col: str = "acq_date",
    epsg_work: int = 9377,
) -> gpd.GeoDataFrame:
    # -------------------------
    # Incendios (SHP)
    # -------------------------
    g_fire = gpd.read_file(path_fire_shp).copy()

    if g_fire.empty:
        raise ValueError("El shapefile de incendios está vacío.")
    if g_fire.crs is None:
        raise ValueError("El shapefile de incendios no tiene CRS definido.")
    if date_col not in g_fire.columns:
        raise ValueError(
            f"No se encontró '{date_col}' en incendios. Columnas disponibles: {list(g_fire.columns)}"
        )

    g_fire[date_col] = pd.to_datetime(g_fire[date_col], format="%d/%m/%Y", errors="coerce")
    if g_fire[date_col].isna().any():
        n_bad = int(g_fire[date_col].isna().sum())
        raise ValueError(
            f"Hay {n_bad} fechas inválidas en incendios en '{date_col}'. "
            "Revisa formato dd/mm/yyyy."
        )

    g_fire["label"] = 1
    g_fire = g_fire.to_crs(epsg=epsg_work)

    # -------------------------
    # Ausencias (SHP)
    # -------------------------
    g_abs = gpd.read_file(path_abs_shp).copy()

    if g_abs.empty:
        raise ValueError("El shapefile de ausencias está vacío.")
    if g_abs.crs is None:
        raise ValueError("El shapefile de ausencias no tiene CRS definido.")

    g_abs["label"] = 0
    g_abs[date_col] = pd.NaT
    g_abs = g_abs.to_crs(epsg=epsg_work)

    # -------------------------
    # Unir
    # -------------------------
    g = pd.concat([g_fire, g_abs], ignore_index=True)
    g = gpd.GeoDataFrame(g, geometry="geometry", crs=f"EPSG:{epsg_work}")

    g = g.reset_index(drop=True)
    g["point_id"] = np.arange(1, len(g) + 1, dtype=int)

    return g


def assign_dates_to_absences(
    g: gpd.GeoDataFrame,
    date_col: str = "acq_date",
    seed: int = 42
) -> gpd.GeoDataFrame:
    g = g.copy()

    fire_dates = g.loc[g["label"] == 1, date_col].dropna().values
    if len(fire_dates) == 0:
        raise ValueError("No hay fechas de incendios para muestrear.")

    rng = np.random.default_rng(seed)
    idx_abs = g.index[g["label"] == 0].to_numpy()
    sampled = rng.choice(fire_dates, size=len(idx_abs), replace=True)

    g.loc[idx_abs, date_col] = pd.to_datetime(sampled)
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    return g