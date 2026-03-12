from pathlib import Path
from typing import List
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask


def list_tiffs(tiff_dir: Path, pattern: str) -> List[Path]:
    files = sorted(tiff_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron TIFF con el patrón: {tiff_dir / pattern}")
    return files


def load_and_clean_aoi(aoi_path: Path) -> gpd.GeoDataFrame:
    aoi_gdf = gpd.read_file(aoi_path)
    aoi_gdf["geometry"] = aoi_gdf["geometry"].buffer(0)
    aoi_gdf = aoi_gdf[aoi_gdf.is_valid & ~aoi_gdf.is_empty].copy()

    if aoi_gdf.empty:
        raise ValueError("El AOI está vacío o inválido después de limpiar geometrías.")
    if aoi_gdf.crs is None:
        raise ValueError("El AOI no tiene CRS definido.")

    return aoi_gdf


def read_aoi_in_raster_crs(aoi_gdf: gpd.GeoDataFrame, raster_crs):
    if str(aoi_gdf.crs) != str(raster_crs):
        return aoi_gdf.to_crs(raster_crs)
    return aoi_gdf


def mask_raster_to_aoi(
    src: rasterio.io.DatasetReader,
    aoi_gdf: gpd.GeoDataFrame,
    all_touched: bool = False
) -> np.ndarray:
    aoi_local = read_aoi_in_raster_crs(aoi_gdf, src.crs)
    geoms = [geom.__geo_interface__ for geom in aoi_local.geometry]
    arr, _ = mask(
        src,
        geoms,
        crop=True,
        filled=True,
        nodata=np.nan,
        all_touched=all_touched
    )
    return arr.astype("float32")