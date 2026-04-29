import ee

from src.temporal_mosaics import build_temporal_mosaics
from src.radiometry import border_noise_mask, gamma0_db, terrain_flattening
from src.speckle import refined_lee_spatial
from src.Copy2_indices import add_ratio_db


def build_collection_by_pass(
    aoi,
    year,
    orbit_pass,
    start_date=None,
    end_date=None,
):
    """
    Construye una colección base Sentinel-1 GRD filtrada por AOI,
    fechas, pase orbital, modo IW y polarizaciones VV/VH.

    Parámetros
    ----------
    aoi : ee.Geometry
        Área de interés.
    year : int
        Año de análisis. Se usa si start_date y end_date no son definidos.
    orbit_pass : str
        Pase orbital: 'ASCENDING' o 'DESCENDING'.
    start_date : str | None
        Fecha inicial en formato 'YYYY-MM-DD'.
    end_date : str | None
        Fecha final en formato 'YYYY-MM-DD'.

    Retorna
    -------
    ee.ImageCollection
        Colección Sentinel-1 filtrada.
    """
    year = int(year)

    if start_date is None:
        start_date = f"{year}-01-01"

    if end_date is None:
        end_date = f"{year + 1}-01-01"

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("productType", "GRD"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
        .sort("system:time_start")
    )

    return col


def build_collection_by_pass_and_relative_orbit(
    aoi,
    year,
    orbit_pass,
    relative_orbit,
    start_date=None,
    end_date=None,
):
    """
    Construye una colección base Sentinel-1 GRD filtrada por AOI,
    fechas, pase orbital y órbita relativa.

    Parámetros
    ----------
    aoi : ee.Geometry
        Área de interés.
    year : int
        Año de análisis.
    orbit_pass : str
        Pase orbital: 'ASCENDING' o 'DESCENDING'.
    relative_orbit : int
        Número de órbita relativa Sentinel-1.
    start_date : str | None
        Fecha inicial.
    end_date : str | None
        Fecha final.

    Retorna
    -------
    ee.ImageCollection
        Colección Sentinel-1 filtrada.
    """
    col = build_collection_by_pass(
        aoi=aoi,
        year=year,
        orbit_pass=orbit_pass,
        start_date=start_date,
        end_date=end_date,
    )

    col = col.filter(
        ee.Filter.eq("relativeOrbitNumber_start", int(relative_orbit))
    )

    return col.sort("system:time_start")


def build_pipeline_base(aoi, cfg):
    """
    Ejecuta el flujo base Sentinel-1 sin lógica pre/post.

    Flujo:
    1. Filtra colección Sentinel-1.
    2. Genera mosaicos temporales con window_days.
    3. Aplica máscara de ruido de borde.
    4. Aplica corrección gamma0.
    5. Aplica corrección de terreno.
    6. Aplica filtro Refined Lee.
    7. Calcula VVVH_ratio.
    8. Ordena la colección final por fecha.

    Esta función NO:
    - busca pares pre/post;
    - calcula diferencias VV/VH;
    - genera VV_Difference, VH_Difference ni VHVV_Difference.
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
    relative_orbit = cfg.get("relative_orbit", None)

    if relative_orbit is None:
        col0 = build_collection_by_pass(
            aoi=aoi,
            year=year,
            orbit_pass=orbit,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        col0 = build_collection_by_pass_and_relative_orbit(
            aoi=aoi,
            year=year,
            orbit_pass=orbit,
            relative_orbit=int(relative_orbit),
            start_date=start_date,
            end_date=end_date,
        )

    col1 = build_temporal_mosaics(
        col=col0,
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        reducer=mosaic_reducer,
    )

    col2 = col1.map(border_noise_mask)
    col3 = col2.map(gamma0_db)

    dem = ee.Image(dem_id)
    col4 = col3.map(lambda img: terrain_flattening(img, dem=dem))

    col5 = col4.map(lambda img: refined_lee_spatial(img, kernel_size=kernel_size))
    col6 = col5.map(lambda img: add_ratio_db(img, eps=eps))

    return col6.sort("system:time_start")


def build_pipeline(aoi, cfg):
    """
    Alias principal del pipeline.

    Por defecto apunta al flujo base sin pre/post.
    """
    return build_pipeline_base(aoi, cfg)


