import ee


def build_collection(
    aoi: ee.Geometry,
    year: int,
    orbit: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.ImageCollection:
    """
    Construye la colección Sentinel-1 GRD para un año / rango temporal y órbita,
    asegurando las bandas VV, VH y angle.

    Parámetros
    ----------
    aoi : ee.Geometry
        Área de interés.
    year : int
        Año de trabajo (se usa si start_date/end_date no se pasan).
    orbit : str
        Órbita, típicamente 'ASCENDING' o 'DESCENDING'.
    start_date : str | None
        Fecha inicial en formato 'YYYY-MM-DD'. Si es None, usa 1 de enero del año.
    end_date : str | None
        Fecha final en formato 'YYYY-MM-DD'. Si es None, usa 1 de enero del año+1.

    Retorna
    -------
    ee.ImageCollection
        Colección filtrada y ordenada por fecha.
    """
    if start_date is None:
        start = ee.Date.fromYMD(int(year), 1, 1)
    else:
        start = ee.Date(start_date)

    if end_date is None:
        end = ee.Date.fromYMD(int(year) + 1, 1, 1)
    else:
        end = ee.Date(end_date)

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("productType", "GRD"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
        .sort("system:time_start")
    )

    return col