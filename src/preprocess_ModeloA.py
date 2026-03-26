from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}


class RasterPreprocessError(Exception):
    """Errores de preprocesamiento raster."""


def _normalize_nodata_array(arr: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    out = arr.astype("float32", copy=False)
    if nodata is not None and not np.isnan(nodata):
        out = np.where(out == nodata, np.nan, out)
    out[~np.isfinite(out)] = np.nan
    return out


def align_raster_to_reference(
    reference_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    band_map: Dict[str, int],
    band_order: Optional[Iterable[str]] = None,
    resampling_method: str = "bilinear",
    dst_nodata: float = np.nan,
    compress: str = "deflate",
) -> str:
    """
    Reproyecta y remuestrea un raster fuente a la grilla exacta de un raster de referencia.

    Parámetros
    ----------
    reference_path : raster de referencia, por ejemplo Landsat 8.
    source_path : raster fuente a alinear, por ejemplo Sentinel-1.
    output_path : ruta de salida del raster alineado.
    band_map : dict nombre_banda -> índice 1-based en el raster fuente.
    band_order : orden de escritura de bandas. Si es None, usa band_map.keys().
    resampling_method : nearest, bilinear, cubic o average.
    dst_nodata : valor nodata de salida.
    compress : compresión GeoTIFF.
    """
    reference_path = Path(reference_path)
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if resampling_method not in RESAMPLING_MAP:
        raise RasterPreprocessError(
            f"Método de remuestreo no soportado: {resampling_method}. "
            f"Use uno de {list(RESAMPLING_MAP.keys())}."
        )

    ordered_names: List[str] = list(band_order) if band_order is not None else list(band_map.keys())
    if not ordered_names:
        raise RasterPreprocessError("No se definió ningún nombre de banda para alinear.")

    missing_names = [name for name in ordered_names if name not in band_map]
    if missing_names:
        raise RasterPreprocessError(
            f"Las siguientes bandas no existen en band_map: {missing_names}. band_map={band_map}"
        )

    resampling = RESAMPLING_MAP[resampling_method]

    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile.copy()
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(source_path) as src:
        out_profile = ref_profile.copy()
        out_profile.update(
            driver="GTiff",
            count=len(ordered_names),
            dtype="float32",
            nodata=dst_nodata,
            compress=compress,
            predictor=2,
        )

        with rasterio.open(output_path, "w", **out_profile) as dst:
            for out_idx, band_name in enumerate(ordered_names, start=1):
                src_idx = band_map[band_name]
                destination = np.full((ref_height, ref_width), dst_nodata, dtype="float32")

                reproject(
                    source=rasterio.band(src, src_idx),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    dst_nodata=dst_nodata,
                    resampling=resampling,
                )

                destination = _normalize_nodata_array(destination, dst_nodata)
                dst.write(destination, out_idx)
                dst.set_band_description(out_idx, band_name)

    return str(output_path)


def compare_raster_grids(reference_path: str | Path, other_path: str | Path) -> dict:
    """Compara grilla, CRS y transform entre dos raster."""
    with rasterio.open(reference_path) as ref, rasterio.open(other_path) as other:
        return {
            "reference_path": str(reference_path),
            "other_path": str(other_path),
            "same_width": ref.width == other.width,
            "same_height": ref.height == other.height,
            "same_crs": ref.crs == other.crs,
            "same_transform": ref.transform == other.transform,
            "reference_shape": (ref.height, ref.width),
            "other_shape": (other.height, other.width),
            "reference_crs": str(ref.crs),
            "other_crs": str(other.crs),
            "reference_transform": tuple(ref.transform),
            "other_transform": tuple(other.transform),
        }