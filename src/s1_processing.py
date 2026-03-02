import ee

def gamma0_normalization(img):
    theta = img.select("angle").multiply(3.141592653589793 / 180.0)
    cos_theta = theta.cos()
    correction = cos_theta.log10().multiply(10)

    vv = img.select("VV").subtract(correction).rename("VV")
    vh = img.select("VH").subtract(correction).rename("VH")

    return img.addBands(vv, overwrite=True).addBands(vh, overwrite=True)


def speckle_filter(col, n_prev=5, boxcar=7):
    kernel = ee.Kernel.square((boxcar - 1) // 2, units="pixels", normalize=True)

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
        angle = med.select("angle").rename("angle")

        return ee.Image.cat([vv, vh, angle]).copyProperties(img, img.propertyNames())

    return col.map(_filter)