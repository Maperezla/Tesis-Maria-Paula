import ee

def add_vv_difference_1y_signed_db(col, orbit, eps=1e-10):
    def _add(img):
        t = ee.Date(img.get("system:time_start"))
        ref_date = t.advance(-365, "day")

        ref_col = (
            col.filter(ee.Filter.eq("orbitProperties_pass", orbit))
              .map(lambda i: i.set(
                  "time_diff",
                  ee.Number(i.get("system:time_start")).subtract(ref_date.millis()).abs()
              ))
              .sort("time_diff")
        )

        ref_img = ee.Image(ref_col.first())

        vv_lin     = ee.Image(10).pow(img.select("VV").divide(10))
        vv_ref_lin = ee.Image(10).pow(ref_img.select("VV").divide(10))

        diff_lin = vv_lin.subtract(vv_ref_lin)

        sign = diff_lin.gt(0).multiply(2).subtract(1)
        diff_abs = diff_lin.abs().max(eps)

        diff_db = diff_abs.log10().multiply(10)
        diff_signed = diff_db.multiply(sign).rename("VV_Difference")

        return img.addBands(diff_signed)

    return col.map(_add)