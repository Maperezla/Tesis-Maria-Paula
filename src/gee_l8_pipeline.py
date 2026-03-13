import ee


def maskL8sr(image):
    qa_pixel = image.select('QA_PIXEL')
    qa_mask = qa_pixel.bitwiseAnd(31).eq(0)
    saturation_mask = image.select('QA_RADSAT').eq(0)

    optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2).float()

    solar_elev = ee.Number(image.get('SUN_ELEVATION'))
    solar_zenith = ee.Number(90).subtract(solar_elev)

    dem = ee.Image('USGS/SRTMGL1_003')
    hill_shadow = ee.Terrain.hillShadow(
        dem, azimuth=ee.Number(180), zenith=solar_zenith, hysteresis=True
    )

    topo_corrected = optical.updateMask(hill_shadow).float()

    return (image.addBands(topo_corrected, None, True)
            .updateMask(qa_mask)
            .updateMask(saturation_mask))


def addNDVI(image):
    return image.addBands(image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))


def addEVI(image):
    L, C1, C2, C3 = 1.0, 2.5, 6.0, 7.5
    evi = image.expression(
        '((C1 * (NIR - RED)) / (NIR + (C2 * RED) - (C3 * BLUE) + L))',
        {'NIR': image.select('SR_B5'),
         'RED': image.select('SR_B4'),
         'BLUE': image.select('SR_B2'),
         'L': L, 'C1': C1, 'C2': C2, 'C3': C3}
    ).rename('EVI')
    return image.addBands(evi)


def addNBR(image):
    nbr = image.expression(
        '((NIR - SWIR1) / (NIR + SWIR1))',
        {'NIR': image.select('SR_B5'),
         'SWIR1': image.select('SR_B6')}
    ).rename('NBR')
    return image.addBands(nbr)


def addNDWI(image):
    ndwi = image.expression(
        '((GREEN - NIR) / (GREEN + NIR))',
        {'GREEN': image.select('SR_B3'),
         'NIR': image.select('SR_B5')}
    ).rename('NDWI')
    return image.addBands(ndwi)


def build_monthly_collection(aoi, start_date, end_date):
    col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
           .filterDate(start_date, end_date)
           .filterBounds(aoi)
           .map(maskL8sr)
           .map(addNDVI)
           .map(addEVI)
           .map(addNBR)
           .map(addNDWI))
    return col