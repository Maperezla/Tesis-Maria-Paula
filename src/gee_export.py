import logging
import ee

from dataclasses import dataclass
from typing import List, Optional

from .gee_l8_pipeline import build_monthly_collection


# ------------------------------------------------------------
# 1) Definición de ventanas bimestrales fijas
# ------------------------------------------------------------
# Cada tupla contiene:
# mes inicial, mes final, etiqueta para el nombre de salida.
BIMONTHLY_PERIODS = [
    (1, 2, "ene_feb"),
    (3, 4, "mar_abr"),
    (5, 6, "may_jun"),
    (7, 8, "jul_ago"),
    (9, 10, "sep_oct"),
    (11, 12, "nov_dic"),
]


# ------------------------------------------------------------
# 2) Información de cada tarea procesada
# ------------------------------------------------------------
@dataclass
class TaskInfo:
    year: int
    start_month: int
    end_month: int
    period_label: str
    desc: str
    img_count: int
    exported: bool
    status: str


# ------------------------------------------------------------
# 3) Configuración de logger
# ------------------------------------------------------------
def setup_logger(log_path: str, name: str = "gee_exports") -> logging.Logger:
    """
    Configura un logger para registrar el estado de las exportaciones.

    Parameters
    ----------
    log_path : str
        Ruta del archivo .log.
    name : str
        Nombre interno del logger.

    Returns
    -------
    logging.Logger
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita duplicar handlers si el notebook se ejecuta varias veces.
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
        for h in logger.handlers
    ):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(fh)

    return logger


# ------------------------------------------------------------
# 4) Crear mosaico bimestral
# ------------------------------------------------------------
def create_monthly_composite(aoi, year: int, start_month: int):
    """
    Crea un mosaico bimestral fijo de Landsat 8.

    Nota:
    Se conserva el nombre create_monthly_composite por compatibilidad
    con el flujo original, pero ahora la ventana temporal es de dos meses.

    Parameters
    ----------
    aoi : ee.Geometry
        Área de interés en formato Earth Engine.
    year : int
        Año del mosaico.
    start_month : int
        Mes inicial de la ventana bimestral.
        Debe ser uno de: 1, 3, 5, 7, 9, 11.

    Returns
    -------
    tuple
        comp2 : ee.Image
            Imagen compuesta mediante mediana, con banda IMG_COUNT.
        count : ee.Number
            Número de imágenes disponibles en la ventana bimestral.
        start : ee.Date
            Fecha inicial de la ventana.
        end : ee.Date
            Fecha final de la ventana.
    """
    start = ee.Date.fromYMD(year, start_month, 1)

    # Ventana bimestral fija:
    # ene-feb, mar-abr, may-jun, jul-ago, sep-oct, nov-dic.
    end = start.advance(2, "month")

    # La función se llama build_monthly_collection,
    # pero puede recibir cualquier rango start-end.
    col = build_monthly_collection(aoi, start, end)

    # Número de imágenes disponibles en la ventana.
    count = col.size()

    # Composición por mediana.
    comp = col.median()

    # Banda de control con el número de imágenes usadas.
    count_band = ee.Image.constant(count).rename("IMG_COUNT")

    # Imagen final.
    comp2 = comp.addBands(count_band).clip(aoi).float()

    return comp2, count, start, end


# ------------------------------------------------------------
# 5) Exportar imagen a Google Drive
# ------------------------------------------------------------
def export_to_drive(
    image,
    aoi,
    desc: str,
    folder: str,
    scale: int,
    max_pixels: float = 1e13
):
    """
    Lanza una tarea de exportación de imagen a Google Drive.

    Parameters
    ----------
    image : ee.Image
        Imagen a exportar.
    aoi : ee.Geometry
        Área de exportación.
    desc : str
        Nombre de la tarea y del archivo.
    folder : str
        Carpeta de Google Drive.
    scale : int
        Resolución espacial de exportación.
    max_pixels : float
        Máximo número de píxeles permitido.

    Returns
    -------
    ee.batch.Task
        Tarea de exportación iniciada.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        region=aoi,
        scale=scale,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF"
    )

    task.start()

    return task


# ------------------------------------------------------------
# 6) Ejecutar exportaciones bimestrales
# ------------------------------------------------------------
def run_monthly_exports(
    aoi,
    year_start: int,
    year_end: int,
    folder: str,
    scale: int,
    logger: logging.Logger,
    dry_run: bool = False
) -> List[TaskInfo]:
    """
    Ejecuta exportaciones bimestrales fijas de Landsat 8.

    Nota importante:
    Se conserva el nombre run_monthly_exports para no cambiar el notebook,
    pero su lógica interna ya no genera mosaicos mensuales, sino bimestrales.

    Para cada año genera como máximo 6 imágenes:

    - ene_feb
    - mar_abr
    - may_jun
    - jul_ago
    - sep_oct
    - nov_dic

    No lanza exportación si IMG_COUNT = 0.

    Parameters
    ----------
    aoi : ee.Geometry
        Área de interés.
    year_start : int
        Año inicial.
    year_end : int
        Año final.
    folder : str
        Carpeta de Google Drive.
    scale : int
        Resolución espacial de exportación.
    logger : logging.Logger
        Logger para registrar el proceso.
    dry_run : bool
        Si True, no lanza exportaciones. Solo registra el flujo.

    Returns
    -------
    List[TaskInfo]
        Lista con el estado de cada ventana bimestral procesada.
    """
    tasks: List[TaskInfo] = []

    for year in range(year_start, year_end + 1):

        for start_month, end_month, period_label in BIMONTHLY_PERIODS:

            comp, count, start, end = create_monthly_composite(
                aoi=aoi,
                year=year,
                start_month=start_month
            )

            # Nombre de salida solicitado:
            # L8_2014_ene_feb
            desc = f"L8_{year}_{period_label}"

            # Evaluar número de imágenes disponibles.
            # Esta llamada a getInfo() es necesaria para decidir
            # si se lanza o no la exportación.
            try:
                n = int(count.getInfo())
            except Exception as e:
                n = -1
                status = "count_error"
                exported = False

                logger.error(
                    f"{desc} | img_count={n} | status={status} | "
                    f"dry_run={dry_run} | error={repr(e)}"
                )

                tasks.append(
                    TaskInfo(
                        year=year,
                        start_month=start_month,
                        end_month=end_month,
                        period_label=period_label,
                        desc=desc,
                        img_count=n,
                        exported=exported,
                        status=status
                    )
                )

                # Si no se pudo obtener el conteo, no se exporta.
                continue

            # Caso 1: no hay imágenes en la ventana bimestral.
            if n == 0:
                status = "skipped_no_images"
                exported = False

                logger.info(
                    f"{desc} | img_count={n} | status={status} | "
                    f"dry_run={dry_run}"
                )

                tasks.append(
                    TaskInfo(
                        year=year,
                        start_month=start_month,
                        end_month=end_month,
                        period_label=period_label,
                        desc=desc,
                        img_count=n,
                        exported=exported,
                        status=status
                    )
                )

                continue

            # Caso 2: hay imágenes, pero dry_run=True.
            if dry_run:
                status = "dry_run_not_exported"
                exported = False

                logger.info(
                    f"{desc} | img_count={n} | status={status} | "
                    f"dry_run={dry_run}"
                )

                tasks.append(
                    TaskInfo(
                        year=year,
                        start_month=start_month,
                        end_month=end_month,
                        period_label=period_label,
                        desc=desc,
                        img_count=n,
                        exported=exported,
                        status=status
                    )
                )

                continue

            # Caso 3: hay imágenes y dry_run=False.
            export_to_drive(
                image=comp,
                aoi=aoi,
                desc=desc,
                folder=folder,
                scale=scale
            )

            status = "export_started"
            exported = True

            logger.info(
                f"{desc} | img_count={n} | status={status} | "
                f"dry_run={dry_run}"
            )

            tasks.append(
                TaskInfo(
                    year=year,
                    start_month=start_month,
                    end_month=end_month,
                    period_label=period_label,
                    desc=desc,
                    img_count=n,
                    exported=exported,
                    status=status
                )
            )

    return tasks