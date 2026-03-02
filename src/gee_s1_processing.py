import ee

def gamma0_normalization(img):
    theta = img.select("angle").multiply(3.141592653589793 / 180)
    cos_t = theta.cos()
    vv = img.select("VV").divide(cos_t).rename("VV")
    vh = img.select("VH").divide(cos_t).rename("VH")
    return img.addBands([vv, vh, angle.rename("angle")], overwrite=True)


def db_to_linear(db_img):
    # linear = 10^(dB/10)
    return ee.Image(10).pow(ee.Image(db_img).divide(10.0))

def linear_to_db(lin_img):
    # dB = 10*log10(linear)
    # proteger contra ceros
    eps = ee.Image.constant(1e-10)
    lin_safe = ee.Image(lin_img).max(eps)
    return lin_safe.log10().multiply(10.0)


def add_vvvh_ratio(img):
    vv_db = img.select("VV")
    vh_db = img.select("VH")

    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)

    # Proteger VH cercano a 0 en lineal
    eps = ee.Image.constant(1e-10)
    ratio_lin = vv_lin.divide(vh_lin.max(eps))

    ratio_db = linear_to_db(ratio_lin).rename("VVVH_ratio")

    return img.addBands(ratio_db)


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