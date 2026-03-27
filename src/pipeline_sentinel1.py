import ee

from .collection import build_collection
from .temporal_mosaics import build_temporal_mosaics
from .radiometry import border_noise_mask, gamma0_db, terrain_flattening
from .speckle import refined_lee_spatial
from .indices import add_ratio_db, build_prepost_pairs_and_indices


def build_pipeline(aoi: ee.Geometry, cfg: dict) -> ee.ImageCollection:
    """
    Pipeline completo Sentinel-1 para detección puntual de incendios con mosaicos móviles.

    Flujo:
    1) construir colección base
    2) construir mosaicos móviles
    3) máscara de borde
    4) gamma0
    5) terrain flattening
    6) speckle filtering
    7) VVVH_ratio
    8) índices pre/post:
       - VV_Difference
       - VH_Difference
       - VHVV_Difference

    Config esperada en cfg
    ----------------------
    year : int
    orbit : str
    eps : float
    kernel_size : int
    dem_id : str
    window_days : int

    Opcionales:
    start_date : str ('YYYY-MM-DD')
    end_date : str ('YYYY-MM-DD')
    mosaic_reducer : str = 'median'
    pre_max_gap_days : int = 60
    """
    year = int(cfg["year"])
    orbit = str(cfg["orbit"])
    eps = float(cfg.get("eps", 1e-10))
    kernel_size = int(cfg.get("kernel_size", 7))
    dem_id = str(cfg["dem_id"])
    window_days = int(cfg.get("window_days", 2))

    start_date = str(cfg.get("start_date", f"{year}-01-01"))
    end_date = str(cfg.get("end_date", f"{year + 1}-01-01"))
    mosaic_reducer = str(cfg.get("mosaic_reducer", "median"))
    pre_max_gap_days = int(cfg.get("pre_max_gap_days", 60))

    dem = ee.Image(dem_id)

    # 1) colección base
    col_raw = build_collection(
        aoi=aoi,
        year=year,
        orbit=orbit,
        start_date=start_date,
        end_date=end_date,
    )

    # 2) mosaicos temporales
    col_mosaic = build_temporal_mosaics(
        col=col_raw,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        reducer=mosaic_reducer,
    )

    # 3) procesamiento SAR por mosaico
    col_proc = col_mosaic.map(border_noise_mask)
    col_proc = col_proc.map(gamma0_db)
    col_proc = col_proc.map(lambda img: terrain_flattening(img, dem=dem))
    col_proc = col_proc.map(lambda img: refined_lee_spatial(img, kernel_size=kernel_size))
    col_proc = col_proc.map(lambda img: add_ratio_db(img, eps=eps))

    # 4) índices de cambio pre/post
    col_final = build_prepost_pairs_and_indices(
        col=col_proc,
        eps=eps,
        max_gap_days=pre_max_gap_days,
    )

    return col_final