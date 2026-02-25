import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon
import ee


def load_aoi_ee_geometry(shp_path: str, target_epsg: int = 4326) -> ee.Geometry:
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("AOI vacío.")

    gdf = gdf.to_crs(epsg=target_epsg)

    # reparar geometrías inválidas
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    if gdf.empty:
        raise ValueError("AOI quedó vacío tras limpieza geométrica.")

    geom = unary_union(gdf.geometry)

    if isinstance(geom, GeometryCollection):
        geoms = [g for g in geom.geoms if (not g.is_empty)]
        if not geoms:
            raise ValueError("No quedaron geometrías válidas en GeometryCollection.")
        geom = unary_union(geoms)

    # Convertir a ee.Geometry sin mapping (solo 2D lon/lat)
    polygons = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise ValueError(f"Tipo no soportado: {type(geom)}")

    rings = []
    for poly in polygons:
        ext = list(poly.exterior.coords)
        ext_2d = [[float(x), float(y)] for x, y, *rest in ext]
        rings.append(ext_2d)

    if len(rings) == 1:
        return ee.Geometry.Polygon(rings[0])
    return ee.Geometry.MultiPolygon(rings)