import ee

def gamma0_normalization(img):
    theta = img.select("angle").multiply(3.141592653589793 / 180)
    cos_t = theta.cos()
    vv = img.select("VV").divide(cos_t).rename("VV")
    vh = img.select("VH").divide(cos_t).rename("VH")
    return img.addBands(vv, overwrite=True).addBands(vh, overwrite=True)


def add_vvvh_ratio(img):
    ratio = img.select("VV").divide(img.select("VH")).rename("VVVH_ratio")
    return img.addBands(ratio)


def speckle_filter(col, n_prev=5, boxcar=7):
    kernel = ee.Kernel.square((boxcar - 1) // 2, units='pixels', normalize=True)

    def _filter(img):
        t = ee.Date(img.get("system:time_start"))

        prev = (
            col.filter(ee.Filter.lt("system:time_start", t.millis()))
              .sort("system:time_start", False)
              .limit(n_prev)
        )

        prev = ee.ImageCollection(
            ee.Algorithms.If(prev.size().gt(0), prev, ee.ImageCollection([img]))
        )

        med = prev.select(["VV", "VH", "angle"]).median()

        vv = med.select("VV").convolve(kernel).rename("VV")
        vh = med.select("VH").convolve(kernel).rename("VH")

        return ee.Image.cat([vv, vh]).copyProperties(img, img.propertyNames())

    return col.map(_filter)