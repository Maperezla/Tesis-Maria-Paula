import ee


def build_temporal_mosaics(
    col: ee.ImageCollection,
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    window_days: int = 2,
    reducer: str = "median",
) -> ee.ImageCollection:
    """
    Construye mosaicos temporales con ventanas móviles.

    Para cada ventana:
    - filtra imágenes en [f_start, f_end)
    - agrega con median (o mean)
    - asigna como system:time_start la fecha más reciente dentro de la ventana
    - guarda metadatos de la ventana

    Parámetros
    ----------
    col : ee.ImageCollection
        Colección base ya filtrada.
    aoi : ee.Geometry
        Área de interés.
    start_date : str
        Fecha inicial, formato YYYY-MM-DD.
    end_date : str
        Fecha final, formato YYYY-MM-DD.
    window_days : int
        Tamaño de la ventana móvil en días.
    reducer : str
        "median" o "mean".

    Retorna
    -------
    ee.ImageCollection
        Colección de mosaicos temporales.
    """
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    window_days = int(window_days)

    total_days = end.difference(start, "day")
    steps = ee.List.sequence(0, total_days.subtract(1), window_days)

    def _make_one(day_offset, acc):
        acc = ee.List(acc)
        day_offset = ee.Number(day_offset)

        f_start = start.advance(day_offset, "day")
        f_end = f_start.advance(window_days, "day")

        subset = col.filterDate(f_start, f_end)

        def _build_image():
            recent_ms = ee.Number(subset.aggregate_max("system:time_start"))
            recent_date = ee.Date(recent_ms)

            if reducer == "mean":
                mosaic = subset.mean()
            else:
                mosaic = subset.median()

            mosaic = (
                ee.Image(mosaic)
                .clip(aoi)
                .set("system:time_start", recent_ms)
                .set("fecha_id", recent_date.format("YYYY-MM-dd"))
                .set("conteo_imagenes", subset.size())
                .set("window_start", f_start.format("YYYY-MM-dd"))
                .set("window_end", f_end.format("YYYY-MM-dd"))
                .set("window_days", window_days)
                .set("aggregation", reducer)
            )
            return mosaic

        out = ee.Algorithms.If(subset.size().gt(0), acc.add(_build_image()), acc)
        return out

    mosaic_list = ee.List(steps.iterate(_make_one, ee.List([])))

    return ee.ImageCollection.fromImages(mosaic_list).sort("system:time_start")