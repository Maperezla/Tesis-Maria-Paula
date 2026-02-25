import ee
import logging

def export_batch(col, aoi, year, start_idx, batch_size, folder, scale, max_pixels, logger=None):
    n = col.size().getInfo()

    end_idx = min(start_idx + batch_size, n)

    img_list = col.toList(n)

    tasks = []

    for i in range(start_idx, end_idx):
        img = ee.Image(img_list.get(i))

        desc = f"S1_{year}_{i}"

        task = ee.batch.Export.image.toDrive(
            image=img.toFloat(),
            description=desc,
            folder=folder,
            fileNamePrefix=desc,
            region=aoi,
            scale=scale,
            maxPixels=max_pixels,
            fileFormat="GeoTIFF"
        )

        task.start()
        tasks.append(desc)

        if logger:
            logger.info(f"Export iniciado: {desc}")

    return tasks