import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon
import ee

def aoi_shp_to_ee_geometry(aoi_path: str) -> ee.Geometry:
    gdf = gpd.read_file(aoi_path).to_crs(epsg=4326)
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    geom_union = unary_union(gdf.geometry)
    polys = [geom_union] if isinstance(geom_union, Polygon) else list(geom_union.geoms)

    rings = []
    for p in polys:
        coords = [[float(x), float(y)] for x, y, *rest in p.exterior.coords]
        rings.append(coords)

    return ee.Geometry.Polygon(rings[0]) if len(rings) == 1 else ee.Geometry.MultiPolygon(rings)