
import ee

def build_collection(aoi: ee.Geometry, year: int, orbit: str) -> ee.ImageCollection:
    """
    Construye la colección Sentinel-1 GRD para un año y una órbita, asegurando VV, VH y angle.
    VV/VH en la colección COPERNICUS/S1_GRD vienen típicamente en dB.
    """
    start = ee.Date.fromYMD(int(year), 1, 1)
    end = ee.Date.fromYMD(int(year) + 1, 1, 1)

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