import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
import ee


def load_aoi_to_ee(path: str):
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=4326)

    # limpiar geometrías
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    geom_union = unary_union(gdf.geometry)

    if isinstance(geom_union, Polygon):
        polys = [geom_union]
    elif isinstance(geom_union, MultiPolygon):
        polys = list(geom_union.geoms)
    else:
        raise ValueError("Geometría no soportada")

    rings = []
    for p in polys:
        coords = [[float(x), float(y)] for x, y, *rest in p.exterior.coords]
        rings.append(coords)

    if len(rings) == 1:
        return ee.Geometry.Polygon(rings[0])
    else:
        return ee.Geometry.MultiPolygon(rings)