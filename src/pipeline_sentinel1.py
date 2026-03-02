import ee
from .collection import build_collection
from .radiometry import border_noise_mask, gamma0_db, terrain_flattening
from .speckle import refined_lee_spatial
from .indices import add_ratio_db, add_vv_difference_1y_signed_db

def build_pipeline(aoi: ee.Geometry, cfg: dict) -> ee.ImageCollection:
    year = int(cfg["year"])
    orbit = str(cfg["orbit"])
    eps = float(cfg["eps"])
    kernel_size = int(cfg["kernel_size"])
    dem_id = str(cfg["dem_id"])

    dem = ee.Image(dem_id)

    col = build_collection(aoi, year, orbit)
    col = col.map(border_noise_mask)
    col = col.map(gamma0_db)
    col = col.map(lambda img: terrain_flattening(img, dem=dem))
    col = col.map(lambda img: refined_lee_spatial(img, kernel_size=kernel_size))
    col = col.map(lambda img: add_ratio_db(img, eps=eps))
    col = add_vv_difference_1y_signed_db(col, eps=eps)

    return col