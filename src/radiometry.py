import ee
import math

_PI = math.pi


def border_noise_mask(img: ee.Image) -> ee.Image:
    """
    Máscara simple basada en 'angle' para reducir bordes ruidosos.
    Heurística actual:
      30 < angle < 46
    """
    angle = img.select("angle")
    mask = angle.gt(30).And(angle.lt(46))
    return img.updateMask(mask)


def gamma0_db(img: ee.Image) -> ee.Image:
    """
    Normalización gamma0 en dB:
      gamma0_dB = sigma0_dB - 10*log10(cos(theta))

    donde theta es el ángulo de incidencia en radianes.
    """
    theta = img.select("angle").multiply(_PI / 180.0)
    cos_theta = theta.cos()

    correction_db = cos_theta.log10().multiply(10)

    vv = img.select("VV").subtract(correction_db).rename("VV")
    vh = img.select("VH").subtract(correction_db).rename("VH")

    return (
        img.addBands(vv, overwrite=True)
           .addBands(vh, overwrite=True)
    )


def terrain_flattening(img: ee.Image, dem: ee.Image) -> ee.Image:
    """
    Terrain Flattening (RTC simplificado).
    Mantiene VV/VH en dB y aplica una corrección topográfica aproximada.
    Además, aplica una máscara geométrica simplificada.
    """
    theta = img.select("angle").multiply(_PI / 180.0)

    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope").multiply(_PI / 180.0)
    _aspect = terrain.select("aspect").multiply(_PI / 180.0)  # reservado por si luego lo usas

    nom = theta.cos()
    den = theta.subtract(slope).cos()

    vol_model = nom.divide(den.max(1e-6))
    corr_db = vol_model.log10().multiply(10)

    vv = img.select("VV").subtract(corr_db).rename("VV")
    vh = img.select("VH").subtract(corr_db).rename("VH")

    layover = slope.gt(theta)
    shadow = slope.lt(theta.multiply(-1))

    mask = layover.Not().And(shadow.Not())

    return (
        img.addBands(vv, overwrite=True)
           .addBands(vh, overwrite=True)
           .updateMask(mask)
    )