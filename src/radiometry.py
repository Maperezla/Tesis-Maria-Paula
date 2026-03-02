
import ee
import math

_PI = math.pi

def border_noise_mask(img: ee.Image) -> ee.Image:
    """
    Máscara simple basada en 'angle' (tu aproximación actual).
    Nota: esta es una heurística. Si luego quieres una máscara más “oficial”,
    se reemplaza aquí sin tocar el pipeline.
    """
    angle = img.select("angle")
    mask = angle.gt(30).And(angle.lt(46))
    return img.updateMask(mask)

def gamma0_db(img: ee.Image) -> ee.Image:
    """
    Normalización gamma0 CONSISTENTE en dB:
      gamma0_dB = sigma0_dB - 10*log10(cos(theta))
    donde theta es el ángulo de incidencia (angle) en grados.

    Importante:
    - En dB NO se divide por cos() (eso sería lineal).
    """
    theta = img.select("angle").multiply(_PI / 180.0)
    cos_theta = theta.cos()

    # 10*log10(cos(theta))
    correction_db = cos_theta.log10().multiply(10)

    vv = img.select("VV").subtract(correction_db).rename("VV")
    vh = img.select("VH").subtract(correction_db).rename("VH")

    return (
        img.addBands(vv, overwrite=True)
           .addBands(vh, overwrite=True)
    )

def terrain_flattening(img: ee.Image, dem: ee.Image) -> ee.Image:
    """
    Terrain Flattening (RTC) simplificado para mantener consistencia con tu decisión:
    - Mantener VV/VH en dB ya "flattened" (recomendado por consistencia).
    - Aplicar máscara geométrica layover/shadow (más correcto, puede quitar píxeles).

    NOTA IMPORTANTE:
    Este es un modelo simplificado (volumétrico aproximado). Para un RTC más riguroso
    (p.ej. Small 2011 / Vollrath), se puede sustituir esta función.

    Entradas:
    - img: ee.Image con bandas VV, VH, angle (VV/VH en dB)
    - dem: ee.Image DEM (SRTM recomendado)
    """
    # theta (incidence) en radianes
    theta = img.select("angle").multiply(_PI / 180.0)

    terrain = ee.Terrain.products(dem)
    slope = terrain.select("slope").multiply(_PI / 180.0)
    aspect = terrain.select("aspect").multiply(_PI / 180.0)

    # Asumimos heading ~ 0 como aproximación (tu "phi = 0")
    # En un RTC formal, phi depende del heading del satélite y aspect.
    phi = ee.Image.constant(0)

    # Modelo volumétrico simplificado:
    # vol_model = cos(theta) / cos(theta - slope)
    # (ojo: esto es una aproximación)
    nom = theta.cos()
    den = theta.subtract(slope).cos()

    # evitar división por 0
    vol_model = nom.divide(den.max(1e-6))

    # aplicar corrección en dB: subtract 10*log10(vol_model)
    corr_db = vol_model.log10().multiply(10)

    vv = img.select("VV").subtract(corr_db).rename("VV")
    vh = img.select("VH").subtract(corr_db).rename("VH")

    # Máscara layover/shadow (simplificada)
    # Layover: slope > theta
    layover = slope.gt(theta)
    # Shadow: condición simplificada (esta forma suele ser más compleja en RTC formal)
    shadow = slope.lt(theta.multiply(-1))

    mask = layover.Not().And(shadow.Not())

    return (
        img.addBands(vv, overwrite=True)
           .addBands(vh, overwrite=True)
           .updateMask(mask)
    )