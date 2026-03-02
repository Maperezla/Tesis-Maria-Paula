import ee

def add_vvvh_ratio_after_speckle(img, eps=1e-10):
    vv_db = img.select("VV")
    vh_db = img.select("VH")

    vv_lin = ee.Image(10).pow(vv_db.divide(10))
    vh_lin = ee.Image(10).pow(vh_db.divide(10))

    ratio_lin = vv_lin.divide(vh_lin.max(eps))
    ratio_db  = ratio_lin.log10().multiply(10).rename("VVVH_ratio")

    return img.addBands(ratio_db)