from pathlib import Path
import numpy as np
from rasterio.io import DatasetReader


def db_to_lin(x_db: np.ndarray) -> np.ndarray:
    return np.power(10.0, x_db / 10.0)


def lin_to_db(x_lin: np.ndarray, eps: float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x_lin, eps))


def get_band_index_by_name(src: DatasetReader, band_name: str) -> int:
    desc = list(src.descriptions)
    if desc is None or all(d is None for d in desc):
        raise ValueError("El raster no tiene 'band descriptions'.")
    try:
        return desc.index(band_name) + 1
    except ValueError:
        raise ValueError(f"No se encontró banda '{band_name}'. Disponibles: {desc}")


def sample_points(src: DatasetReader, xy: list[tuple[float, float]], band_index: int) -> np.ndarray:
    vals = np.fromiter(
        (v[0] for v in src.sample(xy, indexes=band_index)),
        dtype="float64",
        count=len(xy)
    )

    nod = src.nodata
    if nod is not None:
        vals = np.where(vals == nod, np.nan, vals)

    vals = np.where(np.isfinite(vals), vals, np.nan)
    return vals


def list_tifs(raster_dir: Path) -> list[Path]:
    tifs = sorted(raster_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No se encontraron .tif en {raster_dir}")
    return tifs