import ee


def _db_to_lin(db_img: ee.Image) -> ee.Image:
    return ee.Image(10).pow(db_img.divide(10))


def _lin_to_db(lin_img: ee.Image) -> ee.Image:
    return lin_img.log10().multiply(10)


def add_ratio_db(img: ee.Image, eps: float = 1e-10) -> ee.Image:
    """
    Calcula VV/VH en dominio lineal y lo convierte a dB.

    ratio_lin = VV_lin / VH_lin
    ratio_db = 10 * log10(ratio_lin)
    """
    vv_db = img.select("VV")
    vh_db = img.select("VH")

    vv_lin = _db_to_lin(vv_db)
    vh_lin = _db_to_lin(vh_db)

    ratio_lin = vv_lin.divide(vh_lin.max(eps))
    ratio_db = _lin_to_db(ratio_lin).rename("VVVH_ratio")

    return img.addBands(ratio_db)


def add_prepost_change_indices(
    pre_img: ee.Image,
    post_img: ee.Image,
    eps: float = 1e-10,
) -> ee.Image:
    """
    Calcula índices de cambio pre/post y los agrega sobre la imagen post.

    Variables calculadas:
    - VV_Difference   = 10*log10(VV_pre_lin / VV_post_lin)
    - VH_Difference   = 10*log10(VH_pre_lin / VH_post_lin)
    - VHVV_Difference = 10*log10((VH_pre/VV_pre) / (VH_post/VV_post))

    Nota:
    - VV/VH_ratio ya debe haber sido calculado previamente sobre cada mosaico.
    """
    vv_pre = _db_to_lin(pre_img.select("VV"))
    vh_pre = _db_to_lin(pre_img.select("VH"))

    vv_post = _db_to_lin(post_img.select("VV"))
    vh_post = _db_to_lin(post_img.select("VH"))

    vv_diff_lin = vv_pre.divide(vv_post.max(eps))
    vh_diff_lin = vh_pre.divide(vh_post.max(eps))

    vhvv_pre = vh_pre.divide(vv_pre.max(eps))
    vhvv_post = vh_post.divide(vv_post.max(eps))
    vhvv_diff_lin = vhvv_pre.divide(vhvv_post.max(eps))

    vv_diff_db = _lin_to_db(vv_diff_lin).rename("VV_Difference")
    vh_diff_db = _lin_to_db(vh_diff_lin).rename("VH_Difference")
    vhvv_diff_db = _lin_to_db(vhvv_diff_lin).rename("VHVV_Difference")

    return post_img.addBands([vv_diff_db, vh_diff_db, vhvv_diff_db])


def build_prepost_pairs_and_indices(
    col: ee.ImageCollection,
    eps: float = 1e-10,
    max_gap_days: int = 60,
) -> ee.ImageCollection:
    """
    Para cada mosaico post en la colección:
    - busca el mosaico anterior más cercano
    - exige que la separación temporal sea <= max_gap_days
    - calcula índices pre/post
    - devuelve solo imágenes con par válido

    Parámetros
    ----------
    col : ee.ImageCollection
        Colección de mosaicos ya procesados y con VVVH_ratio.
    eps : float
        Épsilon numérico.
    max_gap_days : int
        Separación máxima permitida entre pre y post.

    Retorna
    -------
    ee.ImageCollection
        Colección con VV, VH, angle, VVVH_ratio y variables de cambio.
    """
    max_gap_ms = ee.Number(max_gap_days).multiply(24 * 60 * 60 * 1000)

    col_sorted = col.sort("system:time_start")
    n = col_sorted.size()
    img_list = col_sorted.toList(n)

    indices = ee.List.sequence(1, n.subtract(1))

    def _build_one(i):
        i = ee.Number(i)

        post_img = ee.Image(img_list.get(i))
        pre_img = ee.Image(img_list.get(i.subtract(1)))

        post_t = ee.Number(post_img.get("system:time_start"))
        pre_t = ee.Number(pre_img.get("system:time_start"))
        delta_ms = post_t.subtract(pre_t).abs()

        def _valid_pair():
            out = add_prepost_change_indices(pre_img, post_img, eps=eps)

            return (
                out.set("pre_time", pre_t)
                   .set("post_time", post_t)
                   .set("delta_days", delta_ms.divide(1000 * 60 * 60 * 24))
                   .set("has_pre_pair", 1)
            )

        return ee.Algorithms.If(delta_ms.lte(max_gap_ms), _valid_pair(), None)

    out_list = ee.List(indices.map(_build_one)).removeAll([None])

    return ee.ImageCollection.fromImages(out_list).sort("system:time_start")