import ee


def _db_to_lin(img: ee.Image) -> ee.Image:
    """Convierte dB a escala lineal."""
    return ee.Image(10).pow(ee.Image(img).divide(10))


def _lin_to_db(img: ee.Image) -> ee.Image:
    """Convierte escala lineal a dB."""
    return ee.Image(img).log10().multiply(10)


def add_ratio_db(img: ee.Image, eps: float = 1e-10) -> ee.Image:
    """
    Calcula la razón VV/VH en dB y la agrega como VVVH_ratio.
    """
    img = ee.Image(img)

    vv_lin = _db_to_lin(img.select("VV"))
    vh_lin = _db_to_lin(img.select("VH"))

    ratio_lin = vv_lin.divide(vh_lin.max(eps))
    ratio_db = _lin_to_db(ratio_lin).rename("VVVH_ratio")

    return img.addBands(ratio_db)


# =============================================================================
# DESACTIVADA TEMPORALMENTE POR DEPURACIÓN DE COBERTURA ESPACIAL DEL MOSAICO EXPORTADO
# -----------------------------------------------------------------------------
# La lógica pre/post se desactiva temporalmente porque el diagnóstico mostró
# una caída fuerte de cobertura válida al calcular índices de cambio entre
# imágenes pre y post.
#
# Durante esta fase, el pipeline debe trabajar SOLO con el flujo base:
# VV, VH, angle y VVVH_ratio.
# =============================================================================
def add_prepost_change_indices(
    pre_img: ee.Image,
    post_img: ee.Image,
    eps: float = 1e-10,
) -> ee.Image:
    raise RuntimeError(
        "add_prepost_change_indices() está DESACTIVADA TEMPORALMENTE POR "
        "DEPURACIÓN DE COBERTURA ESPACIAL DEL MOSAICO EXPORTADO. "
        "Use build_pipeline_base() / build_pipeline() durante esta prueba."
    )


def _valid_pixel_sum(img: ee.Image, band: str, region: ee.Geometry, scale: int = 30) -> ee.Number:
    result = ee.Image(img).select(band).mask().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    )
    return ee.Number(result.get(band, 0))


def _overlap_valid_pixel_sum(
    pre_img: ee.Image,
    post_img: ee.Image,
    band: str,
    region: ee.Geometry,
    scale: int = 30,
) -> ee.Number:
    pre_mask = ee.Image(pre_img).select(band).mask()
    post_mask = ee.Image(post_img).select(band).mask()
    overlap = pre_mask.And(post_mask).rename("overlap")

    result = overlap.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    )
    return ee.Number(result.get("overlap", 0))


def build_prepost_pairs_and_indices(
    col: ee.ImageCollection,
    aoi: ee.Geometry,
    eps: float = 1e-10,
    max_gap_days: int = 60,
    coverage_scale: int = 30,
    min_overlap_pixels: float = 1.0,
    min_overlap_fraction_of_post: float = 0.10,
) -> ee.ImageCollection:
    """
    Función mantenida en el módulo para la variante extendida del pipeline,
    pero no debe usarse durante la prueba de depuración del flujo base.

    Si deseas reactivar la lógica pre/post más adelante, primero debes
    restaurar add_prepost_change_indices().
    """
    raise RuntimeError(
        "build_prepost_pairs_and_indices() no debe usarse durante esta prueba. "
        "Actualmente build_pipeline() apunta al flujo base SIN pre/post. "
        "Para reactivar este flujo, restaure explícitamente "
        "add_prepost_change_indices() y use build_pipeline_with_prepost()."
    )