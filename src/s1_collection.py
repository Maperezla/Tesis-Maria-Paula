import ee

def build_s1_collection(aoi, year, orbit):
    start = ee.Date.fromYMD(year, 1, 1)
    end   = ee.Date.fromYMD(year + 1, 1, 1)

    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("productType", "GRD"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH", "angle"])  # no perder angle
        .sort("system:time_start")
    )