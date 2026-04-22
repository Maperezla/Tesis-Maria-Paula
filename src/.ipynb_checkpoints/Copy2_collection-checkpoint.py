import ee


def build_collection_by_pass(
    aoi: ee.Geometry,
    year: int,
    orbit_pass: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.ImageCollection:
    """
    Construye la colección Sentinel-1 GRD para un solo pase:
    ASCENDING o DESCENDING.
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
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])
        .sort("system:time_start")
    )

    return col


def build_collection_ascending(
    aoi: ee.Geometry,
    year: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.ImageCollection:
    return build_collection_by_pass(
        aoi=aoi,
        year=year,
        orbit_pass="ASCENDING",
        start_date=start_date,
        end_date=end_date,
    )


def build_collection_descending(
    aoi: ee.Geometry,
    year: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.ImageCollection:
    return build_collection_by_pass(
        aoi=aoi,
        year=year,
        orbit_pass="DESCENDING",
        start_date=start_date,
        end_date=end_date,
    )


def build_collection_by_pass_and_relative_orbit(
    aoi: ee.Geometry,
    year: int,
    orbit_pass: str,
    relative_orbit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.ImageCollection:
    """
    Construye una colección Sentinel-1 de un solo pase y,
    opcionalmente, de una sola órbita relativa.
    """
    col = build_collection_by_pass(
        aoi=aoi,
        year=year,
        orbit_pass=orbit_pass,
        start_date=start_date,
        end_date=end_date,
    )

    if relative_orbit is not None:
        col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", int(relative_orbit)))

    return col.sort("system:time_start")


# Compatibilidad con tu código actual
def build_collection(
    aoi: ee.Geometry,
    year: int,
    orbit: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> ee.ImageCollection:
    """
    Wrapper de compatibilidad.
    'orbit' aquí se interpreta como ASCENDING o DESCENDING.
    """
    return build_collection_by_pass(
        aoi=aoi,
        year=year,
        orbit_pass=orbit,
        start_date=start_date,
        end_date=end_date,
    )