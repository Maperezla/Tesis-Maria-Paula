
import ee

def _db_to_lin(db_img: ee.Image) -> ee.Image:
    return ee.Image(10).pow(db_img.divide(10))

def _lin_to_db(lin_img: ee.Image) -> ee.Image:
    return lin_img.log10().multiply(10)

def add_ratio_db(img: ee.Image, eps: float = 1e-10) -> ee.Image:
    """
    Calcula ratio VV/VH *después* del speckle:
    - Convierte VV y VH (dB) a lineal
    - ratio_lin = VV_lin / max(VH_lin, eps)
    - ratio_db = 10*log10(ratio_lin)

    Importante sobre "signo":
    En física SAR, VV_lin y VH_lin son potencias (>=0). Por tanto ratio_lin no es negativo.
    Si en tu flujo aparece "negativo", típicamente es por operar en dB (restas) o por artefactos.
    Aquí lo hacemos físicamente correcto (ratio_lin >= 0).
    """
    vv_db = img.select("VV")
    vh_db = img.select("VH")

    vv_lin = _db_to_lin(vv_db)
    vh_lin = _db_to_lin(vh_db)

    ratio_lin = vv_lin.divide(vh_lin.max(eps))
    ratio_db = _lin_to_db(ratio_lin).rename("VVVH_ratio")

    return img.addBands(ratio_db)

def add_vv_difference_1y_signed_db(col: ee.ImageCollection, eps: float = 1e-10) -> ee.ImageCollection:
    """
    Para cada imagen en fecha t:
    - ref_date = t - 365 días
    - referencia = imagen en la MISMA COLECCIÓN cuya fecha sea la más cercana a ref_date
      (si tu colección ya está filtrada por órbita, esto mantiene consistencia).

    Diferencia:
    - en lineal: diff_lin = VV_lin(t) - VV_lin(ref)
    - luego transformación firmada a "dB con signo":
        diff_db_signed = sign(diff_lin) * 10*log10(|diff_lin| + eps)

    Nota: esto NO es un dB estándar de razón; es una transformación para visualizar
    magnitud + dirección del cambio.
    """
    eps = float(eps)

    def _add(img: ee.Image) -> ee.Image:
        t = ee.Date(img.get("system:time_start"))
        ref_date = t.advance(-365, "day")

        # Buscar referencia: imagen con fecha más cercana a ref_date
        ref_col = col.map(
            lambda i: i.set(
                "tdiff",
                ee.Number(i.get("system:time_start")).subtract(ref_date.millis()).abs()
            )
        ).sort("tdiff")

        ref = ee.Image(ref_col.first())

        vv_lin = _db_to_lin(img.select("VV"))
        ref_lin = _db_to_lin(ref.select("VV"))

        diff_lin = vv_lin.subtract(ref_lin)

        # sign: +1 si diff>0, -1 si diff<=0 (incluye 0 como -1; si quieres 0->0, se ajusta)
        sign = diff_lin.gt(0).multiply(2).subtract(1)

        diff_abs = diff_lin.abs().add(eps)
        diff_db = _lin_to_db(diff_abs)

        diff_signed = diff_db.multiply(sign).rename("VV_Difference")

        return img.addBands(diff_signed)

    return col.map(_add)