from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window


class RasterAlignmentError(Exception):
    """Error de alineación entre rasters."""


def read_l8_stack(l8_path: str, band_map: Dict[str, str]) -> tuple[np.ndarray, dict, list[str]]:
    """Lee Landsat 8 por descriptions y retorna stack completo en memoria.

    Parameters
    ----------
    l8_path : str
        Ruta al GeoTIFF multibanda Landsat 8.
    band_map : dict
        Mapeo {descripcion_en_raster: nombre_feature_modelo}.

    Returns
    -------
    arr : np.ndarray
        Array (H, W, C) float32.
    meta_ref : dict
        Metadatos de referencia.
    out_names : list[str]
        Nombres finales de bandas/features en el mismo orden del stack.
    """
    with rasterio.open(l8_path) as src:
        desc = list(src.descriptions)
        if any(d is None for d in desc):
            raise ValueError(
                "L8: hay bandas sin 'descriptions'. El pipeline requiere descriptions para mapear nombres."
            )

        missing = [k for k in band_map.keys() if k not in desc]
        if missing:
            raise ValueError(f"L8: faltan bandas requeridas en descriptions: {missing}. Disponibles: {desc}")

        bands_out: List[np.ndarray] = []
        out_names: List[str] = []
        for band_in, band_out in band_map.items():
            idx = desc.index(band_in) + 1
            band = src.read(idx).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            band[~np.isfinite(band)] = np.nan
            bands_out.append(band)
            out_names.append(band_out)

        arr = np.stack(bands_out, axis=-1)
        meta_ref = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": "float32",
            "nodata": src.nodata,
        }
    return arr, meta_ref, out_names


def read_s1_stack(s1_path: str, band_idx: Dict[str, int], ref_shape: tuple[int, int], ref_meta: dict | None = None) -> np.ndarray:
    """Lee Sentinel-1 por índices 1-based y valida alineación básica."""
    with rasterio.open(s1_path) as src:
        h_ref, w_ref = ref_shape
        if src.height != h_ref or src.width != w_ref:
            raise RasterAlignmentError(
                f"S1: shape distinto a referencia. S1={src.height}x{src.width} vs ref={h_ref}x{w_ref}."
            )

        if ref_meta is not None:
            if src.crs != ref_meta.get("crs"):
                raise RasterAlignmentError(f"S1: CRS distinto al raster de referencia. S1={src.crs}, ref={ref_meta.get('crs')}")
            if src.transform != ref_meta.get("transform"):
                raise RasterAlignmentError("S1: transform distinto al raster de referencia.")

        bands: List[np.ndarray] = []
        for name in ["VV", "VH", "angle"]:
            idx = band_idx[name]
            band = src.read(idx).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            band[~np.isfinite(band)] = np.nan
            bands.append(band)

        arr = np.stack(bands, axis=-1)
    return arr


def build_windows(width: int, height: int, size: int) -> Iterator[Window]:
    """Genera ventanas cuadradas para recorrer un raster por bloques."""
    if size <= 0:
        raise ValueError("El tamaño de ventana debe ser > 0")

    for row_off in range(0, height, size):
        win_h = min(size, height - row_off)
        for col_off in range(0, width, size):
            win_w = min(size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def read_l8_window(src: rasterio.io.DatasetReader, band_map: Dict[str, str], window: Window) -> tuple[np.ndarray, list[str]]:
    """Lee una ventana de L8 usando descriptions."""
    desc = list(src.descriptions)
    if any(d is None for d in desc):
        raise ValueError("L8: hay bandas sin descriptions.")

    missing = [k for k in band_map.keys() if k not in desc]
    if missing:
        raise ValueError(f"L8: faltan bandas requeridas: {missing}")

    bands_out: List[np.ndarray] = []
    out_names: List[str] = []
    for band_in, band_out in band_map.items():
        idx = desc.index(band_in) + 1
        band = src.read(idx, window=window).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            band[band == nodata] = np.nan
        band[~np.isfinite(band)] = np.nan
        bands_out.append(band)
        out_names.append(band_out)

    arr = np.stack(bands_out, axis=-1)
    return arr, out_names


def read_s1_window(src: rasterio.io.DatasetReader, band_idx: Dict[str, int], window: Window, ref_window_shape: tuple[int, int]) -> np.ndarray:
    """Lee una ventana de S1 usando índices fijos."""
    h_ref, w_ref = ref_window_shape
    bands: List[np.ndarray] = []
    for name in ["VV", "VH", "angle"]:
        idx = band_idx[name]
        band = src.read(idx, window=window).astype("float32")
        if band.shape != (h_ref, w_ref):
            raise RasterAlignmentError(
                f"S1 ventana con shape distinto. Esperado {(h_ref, w_ref)}, obtenido {band.shape}"
            )
        nodata = src.nodata
        if nodata is not None:
            band[band == nodata] = np.nan
        band[~np.isfinite(band)] = np.nan
        bands.append(band)

    arr = np.stack(bands, axis=-1)
    return arr


def get_raster_meta(raster_path: str) -> dict:
    """Devuelve metadatos básicos de un raster."""
    with rasterio.open(raster_path) as src:
        return {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "count": src.count,
            "nodata": src.nodata,
        }


def write_geotiff(out_path: str, arr2d: np.ndarray, meta_ref: dict, nodata_val, dtype: str) -> None:
    """Escribe un GeoTIFF de una banda usando metadatos de referencia."""
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = arr2d.shape
    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "crs": meta_ref["crs"],
        "transform": meta_ref["transform"],
        "dtype": dtype,
        "nodata": nodata_val,
        "compress": "deflate",
        "predictor": 2 if dtype in ("float32", "float64") else 1,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr2d.astype(dtype), 1)


def create_output_raster(out_path: str, meta_ref: dict, dtype: str, nodata_val) -> None:
    """Crea un GeoTIFF vacío listo para escritura por ventanas."""
    out_path = str(Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "GTiff",
        "height": meta_ref["height"],
        "width": meta_ref["width"],
        "count": 1,
        "crs": meta_ref["crs"],
        "transform": meta_ref["transform"],
        "dtype": dtype,
        "nodata": nodata_val,
        "compress": "deflate",
        "predictor": 2 if dtype in ("float32", "float64") else 1,
    }
    with rasterio.open(out_path, "w", **meta):
        pass


def write_window(out_path: str, arr2d: np.ndarray, window: Window) -> None:
    """Escribe una ventana en un raster ya creado."""
    with rasterio.open(out_path, "r+") as dst:
        dst.write(arr2d, 1, window=window)


def read_vector(path: str, layer: str | None = None, engine: str | None = None) -> gpd.GeoDataFrame:
    """Lee un archivo vectorial (shp, gpkg, etc.)."""
    kwargs = {}
    if layer is not None:
        kwargs["layer"] = layer
    if engine is not None:
        kwargs["engine"] = engine
    return gpd.read_file(path, **kwargs)


def sample_raster_at_points(raster_path: str, gdf_pts: gpd.GeoDataFrame) -> np.ndarray:
    """Muestrea un raster en puntos y devuelve array de tamaño N."""
    with rasterio.open(raster_path) as src:
        pts = gdf_pts.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts.geometry]
        vals = np.array([v[0] for v in src.sample(coords, indexes=1)], dtype="float32")
        nodata = src.nodata
        if nodata is not None:
            vals[vals == nodata] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        return vals


def validate_raster_alignment(l8_path: str, s1_path: str) -> dict:
    """Verifica alineación entre dos rasters y retorna resumen."""
    with rasterio.open(l8_path) as l8, rasterio.open(s1_path) as s1:
        result = {
            "same_width": l8.width == s1.width,
            "same_height": l8.height == s1.height,
            "same_crs": l8.crs == s1.crs,
            "same_transform": l8.transform == s1.transform,
            "l8_shape": (l8.height, l8.width),
            "s1_shape": (s1.height, s1.width),
            "l8_crs": str(l8.crs),
            "s1_crs": str(s1.crs),
        }
    return result