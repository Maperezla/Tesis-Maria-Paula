import numpy as np
import geopandas as gpd


def load_points(gpkg_path, layer_name):
    gdf = gpd.read_file(gpkg_path, layer=layer_name)

    if gdf.empty:
        raise ValueError("La capa de puntos está vacía.")

    if "point_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["point_id"] = np.arange(1, len(gdf) + 1)

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.type.isin(["Point"])].copy()

    if len(gdf) == 0:
        raise ValueError("No se encontraron geometrías tipo Point válidas.")

    if gdf.crs is None:
        raise ValueError("El GeoPackage no tiene CRS definido.")

    return gdf


def reproject_points_to_raster_crs(gdf, raster_crs):
    if raster_crs is None:
        raise ValueError("El raster no tiene CRS definido.")
    return gdf.to_crs(raster_crs)