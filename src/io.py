from pathlib import Path
from typing import List, Tuple
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask

def load_and_clean_aoi(aoi_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(aoi_path)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty].copy()
    if gdf.empty:
        raise ValueError("El AOI está vacío o inválido después de limpiar geometrías.")
    if gdf.crs is None:
        raise ValueError("El AOI no tiene CRS definido.")
    return gdf

def list_tiffs(tiff_dir: Path, pattern: str) -> List[Path]:
    files = sorted(tiff_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron TIFF con el patrón: {tiff_dir / pattern}")
    return files

def aoi_in_raster_crs(aoi_gdf: gpd.GeoDataFrame, raster_crs) -> gpd.GeoDataFrame:
    if str(aoi_gdf.crs) != str(raster_crs):
        return aoi_gdf.to_crs(raster_crs)
    return aoi_gdf

def mask_raster_to_aoi(src: rasterio.io.DatasetReader,
                       aoi_gdf: gpd.GeoDataFrame,
                       all_touched: bool = False) -> Tuple[np.ndarray, rasterio.Affine]:
    aoi_local = aoi_in_raster_crs(aoi_gdf, src.crs)
    geoms = [geom.__geo_interface__ for geom in aoi_local.geometry]
    arr, out_transform = mask(
        src, geoms, crop=True, filled=True, nodata=np.nan, all_touched=all_touched
    )
    return arr.astype("float32"), out_transform