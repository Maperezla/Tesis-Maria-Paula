import ee


def _db_to_lin(img: ee.Image) -> ee.Image:
    return ee.Image(10).pow(ee.Image(img).divide(10))


def _lin_to_db(img: ee.Image) -> ee.Image:
    return ee.Image(img).log10().multiply(10)


def add_ratio_db(img: ee.Image, eps: float = 1e-10) -> ee.Image:
    img = ee.Image(img)

    vv_lin = _db_to_lin(img.select("VV"))
    vh_lin = _db_to_lin(img.select("VH"))

    ratio_lin = vv_lin.divide(vh_lin.max(eps))
    ratio_db = _lin_to_db(ratio_lin).rename("VVVH_ratio")

    return img.addBands(ratio_db)


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


def _empty_change_bands(reference_img: ee.Image) -> ee.Image:
    """
    Crea bandas de cambio vacías pero con la misma proyección espacial del mosaico base.
    Quedan totalmente enmascaradas.
    """
    ref = ee.Image(reference_img).select("VV")
    empty = ref.multiply(0).selfMask()

    return ee.Image.cat([
        empty.rename("VV_Difference"),
        empty.rename("VH_Difference"),
        empty.rename("VHVV_Difference"),
    ])


def add_prepost_change_indices(
    pre_img: ee.Image,
    post_img: ee.Image,
    eps: float = 1e-10,
) -> ee.Image:
    """
    Calcula índices de cambio pre/post y los agrega sobre la base del post_img.
    """
    base = ee.Image(post_img).select(["VV", "VH", "angle", "VVVH_ratio"])

    vv_pre = _db_to_lin(ee.Image(pre_img).select("VV"))
    vh_pre = _db_to_lin(ee.Image(pre_img).select("VH"))

    vv_post = _db_to_lin(ee.Image(post_img).select("VV"))
    vh_post = _db_to_lin(ee.Image(post_img).select("VH"))

    vv_diff_lin = vv_pre.divide(vv_post.max(eps))
    vh_diff_lin = vh_pre.divide(vh_post.max(eps))

    vhvv_pre = vh_pre.divide(vv_pre.max(eps))
    vhvv_post = vh_post.divide(vv_post.max(eps))
    vhvv_diff_lin = vhvv_pre.divide(vhvv_post.max(eps))

    vv_diff_db = _lin_to_db(vv_diff_lin).rename("VV_Difference")
    vh_diff_db = _lin_to_db(vh_diff_lin).rename("VH_Difference")
    vhvv_diff_db = _lin_to_db(vhvv_diff_lin).rename("VHVV_Difference")

    return base.addBands([vv_diff_db, vh_diff_db, vhvv_diff_db])


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
    Mantiene TODAS las imágenes de la colección.
    - Si una imagen tiene pre válido, calcula bandas de cambio y marca has_pre_pair=1.
    - Si no tiene pre válido, conserva el mosaico base, agrega bandas de cambio vacías
      y marca has_pre_pair=0.
    """
    max_gap_ms = ee.Number(max_gap_days).multiply(24 * 60 * 60 * 1000)

    col_sorted = col.sort("system:time_start")
    n = col_sorted.size()
    img_list = col_sorted.toList(n)

    indices = ee.List.sequence(0, n.subtract(1))

    def _build_one(i):
        i = ee.Number(i)
        post_img = ee.Image(img_list.get(i))
        post_t = ee.Number(post_img.get("system:time_start"))

        post_valid_vv = _valid_pixel_sum(post_img, "VV", aoi, coverage_scale)
        post_valid_vh = _valid_pixel_sum(post_img, "VH", aoi, coverage_scale)

        # Si es la primera imagen, no puede tener pre
        def _no_pre_output():
            base = ee.Image(post_img).select(["VV", "VH", "angle", "VVVH_ratio"])
            out = base.addBands(_empty_change_bands(post_img))

            return (
                out.copyProperties(post_img, post_img.propertyNames())
                   .set("pre_index", None)
                   .set("pre_time", None)
                   .set("post_time", post_t)
                   .set("delta_days", None)
                   .set("has_pre_pair", 0)
                   .set("pair_selection", "no_previous_image")
                   .set("pre_valid_vv", None)
                   .set("pre_valid_vh", None)
                   .set("post_valid_vv", post_valid_vv)
                   .set("post_valid_vh", post_valid_vh)
                   .set("overlap_vv", None)
                   .set("overlap_vh", None)
                   .set("overlap_frac_vv", None)
                   .set("overlap_frac_vh", None)
                   .set("pair_score", None)
            )

        def _with_candidates():
            candidate_indices = ee.List.sequence(0, i.subtract(1))

            def _candidate_score(j):
                j = ee.Number(j)
                pre_img = ee.Image(img_list.get(j))
                pre_t = ee.Number(pre_img.get("system:time_start"))
                delta_ms = post_t.subtract(pre_t)

                is_valid_time = delta_ms.gt(0).And(delta_ms.lte(max_gap_ms))

                def _build_candidate():
                    pre_valid_vv = _valid_pixel_sum(pre_img, "VV", aoi, coverage_scale)
                    pre_valid_vh = _valid_pixel_sum(pre_img, "VH", aoi, coverage_scale)

                    overlap_vv = _overlap_valid_pixel_sum(pre_img, post_img, "VV", aoi, coverage_scale)
                    overlap_vh = _overlap_valid_pixel_sum(pre_img, post_img, "VH", aoi, coverage_scale)

                    overlap_frac_vv = overlap_vv.divide(post_valid_vv.max(1))
                    overlap_frac_vh = overlap_vh.divide(post_valid_vh.max(1))

                    delta_days = delta_ms.divide(1000 * 60 * 60 * 24)

                    score = (
                        overlap_vv.multiply(10)
                        .add(overlap_vh.multiply(10))
                        .add(pre_valid_vv)
                        .add(pre_valid_vh)
                        .subtract(delta_days.multiply(1000))
                    )

                    return ee.Feature(
                        None,
                        {
                            "pre_index": j,
                            "pre_time": pre_t,
                            "delta_ms": delta_ms,
                            "delta_days": delta_days,
                            "pre_valid_vv": pre_valid_vv,
                            "pre_valid_vh": pre_valid_vh,
                            "post_valid_vv": post_valid_vv,
                            "post_valid_vh": post_valid_vh,
                            "overlap_vv": overlap_vv,
                            "overlap_vh": overlap_vh,
                            "overlap_frac_vv": overlap_frac_vv,
                            "overlap_frac_vh": overlap_frac_vh,
                            "score": score,
                        },
                    )

                return ee.Algorithms.If(is_valid_time, _build_candidate(), None)

            candidates_fc = ee.FeatureCollection(
                ee.List(candidate_indices.map(_candidate_score)).removeAll([None])
            )

            def _no_valid_candidate_output():
                base = ee.Image(post_img).select(["VV", "VH", "angle", "VVVH_ratio"])
                out = base.addBands(_empty_change_bands(post_img))

                return (
                    out.copyProperties(post_img, post_img.propertyNames())
                       .set("pre_index", None)
                       .set("pre_time", None)
                       .set("post_time", post_t)
                       .set("delta_days", None)
                       .set("has_pre_pair", 0)
                       .set("pair_selection", "no_valid_candidate")
                       .set("pre_valid_vv", None)
                       .set("pre_valid_vh", None)
                       .set("post_valid_vv", post_valid_vv)
                       .set("post_valid_vh", post_valid_vh)
                       .set("overlap_vv", None)
                       .set("overlap_vh", None)
                       .set("overlap_frac_vv", None)
                       .set("overlap_frac_vh", None)
                       .set("pair_score", None)
                )

            def _select_best_candidate():
                best = ee.Feature(candidates_fc.sort("score", False).first())

                best_pre_index = ee.Number(best.get("pre_index"))
                best_pre_img = ee.Image(img_list.get(best_pre_index))

                best_overlap_vv = ee.Number(best.get("overlap_vv"))
                best_overlap_vh = ee.Number(best.get("overlap_vh"))
                best_overlap_frac_vv = ee.Number(best.get("overlap_frac_vv"))
                best_overlap_frac_vh = ee.Number(best.get("overlap_frac_vh"))

                overlap_ok = (
                    best_overlap_vv.gte(min_overlap_pixels)
                    .And(best_overlap_vh.gte(min_overlap_pixels))
                    .And(best_overlap_frac_vv.gte(min_overlap_fraction_of_post))
                    .And(best_overlap_frac_vh.gte(min_overlap_fraction_of_post))
                )

                def _paired_output():
                    out = add_prepost_change_indices(best_pre_img, post_img, eps=eps)

                    return (
                        out.copyProperties(post_img, post_img.propertyNames())
                           .set("pre_index", best_pre_index)
                           .set("pre_time", best.get("pre_time"))
                           .set("post_time", post_t)
                           .set("delta_days", best.get("delta_days"))
                           .set("has_pre_pair", 1)
                           .set("pair_selection", "best_previous_by_valid_coverage")
                           .set("pre_valid_vv", best.get("pre_valid_vv"))
                           .set("pre_valid_vh", best.get("pre_valid_vh"))
                           .set("post_valid_vv", best.get("post_valid_vv"))
                           .set("post_valid_vh", best.get("post_valid_vh"))
                           .set("overlap_vv", best.get("overlap_vv"))
                           .set("overlap_vh", best.get("overlap_vh"))
                           .set("overlap_frac_vv", best.get("overlap_frac_vv"))
                           .set("overlap_frac_vh", best.get("overlap_frac_vh"))
                           .set("pair_score", best.get("score"))
                    )

                return ee.Algorithms.If(overlap_ok, _paired_output(), _no_valid_candidate_output())

            return ee.Algorithms.If(candidates_fc.size().gt(0), _select_best_candidate(), _no_valid_candidate_output())

        return ee.Algorithms.If(i.eq(0), _no_pre_output(), _with_candidates())

    out_list = ee.List(indices.map(_build_one))

    return ee.ImageCollection.fromImages(out_list).sort("system:time_start")