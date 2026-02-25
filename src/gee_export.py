import logging
import ee
from dataclasses import dataclass
from typing import List, Tuple

from .gee_l8_pipeline import build_monthly_collection


@dataclass
class TaskInfo:
    year: int
    month: int
    desc: str


def setup_logger(log_path: str, name: str = "gee_exports") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == log_path
               for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger


def create_monthly_composite(aoi, year: int, month: int):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    col = build_monthly_collection(aoi, start, end)
    count = col.size()
    comp = col.median()
    count_band = ee.Image.constant(count).rename('IMG_COUNT')

    comp2 = comp.addBands(count_band).clip(aoi).float()
    return comp2, count


def export_to_drive(image, aoi, desc: str, folder: str, scale: int, max_pixels: float = 1e13):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=desc,
        folder=folder,
        fileNamePrefix=desc,
        region=aoi,
        scale=scale,
        maxPixels=max_pixels,
        fileFormat='GeoTIFF'
    )
    task.start()
    return task


def run_monthly_exports(
    aoi,
    year_start: int,
    year_end: int,
    folder: str,
    scale: int,
    logger: logging.Logger,
    dry_run: bool = False
) -> List[TaskInfo]:
    tasks: List[TaskInfo] = []

    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            comp, count = create_monthly_composite(aoi, year, month)

            month_str = str(month).zfill(2)
            desc = f"L8_{year}_{month_str}"

            # Evitar getInfo masivo si no lo quieres (lento): opcional
            try:
                n = int(count.getInfo())
            except Exception:
                n = -1

            logger.info(f"{desc} | img_count={n} | dry_run={dry_run}")

            if not dry_run:
                export_to_drive(comp, aoi, desc, folder=folder, scale=scale)

            tasks.append(TaskInfo(year=year, month=month, desc=desc))

    return tasks