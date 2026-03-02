
import ee

def _db_to_lin(db_img: ee.Image) -> ee.Image:
    return ee.Image(10).pow(db_img.divide(10))

def _lin_to_db(lin_img: ee.Image) -> ee.Image:
    # lin -> dB (10*log10)
    return lin_img.log10().multiply(10)

def refined_lee_spatial(img: ee.Image, kernel_size: int = 7) -> ee.Image:
    """
    Speckle filter espacial tipo Lee (aprox "Refined Lee" simplificado).
    - Trabaja en dominio lineal (recomendado físicamente).
    - Devuelve VV y VH en dB.
    - Preserva la banda 'angle' (no la filtra, solo la re-adjunta).

    Nota: un Refined Lee "completo" suele usar ventanas direccionales y gradientes;
    aquí mantenemos un Lee clásico/robusto (simple y reproducible) con kernel square.
    Si necesitas el "refined" direccional real, se implementa aquí.

    Entradas:
    - img: ee.Image con VV/VH en dB y angle
    - kernel_size: tamaño de ventana (7 -> 7x7)
    """
    k = int(kernel_size)
    radius = (k - 1) // 2
    kernel = ee.Kernel.square(radius, units="pixels", normalize=False)

    vv_lin = _db_to_lin(img.select("VV"))
    vh_lin = _db_to_lin(img.select("VH"))

    def lee_filter(band_lin: ee.Image) -> ee.Image:
        mean = band_lin.reduceNeighborhood(ee.Reducer.mean(), kernel)
        var = band_lin.reduceNeighborhood(ee.Reducer.variance(), kernel)

        # ruido relativo (aprox); protegemos contra mean=0
        mean2 = mean.pow(2).max(1e-12)
        noise = var.divide(mean2)

        # peso (aprox): var - noise*mean^2 / var
        # proteger contra var=0
        var_safe = var.max(1e-12)
        weight = var.subtract(noise.multiply(mean2)).divide(var_safe)

        result = mean.add(weight.multiply(band_lin.subtract(mean)))
        return result

    vv_f = lee_filter(vv_lin)
    vh_f = lee_filter(vh_lin)

    vv_db = _lin_to_db(vv_f).rename("VV")
    vh_db = _lin_to_db(vh_f).rename("VH")

    angle = img.select("angle").rename("angle")

    # re-adjuntar asegurando no perder angle
    return (
        ee.Image.cat([vv_db, vh_db, angle])
        .copyProperties(img, img.propertyNames())
    )