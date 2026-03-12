from pathlib import Path
from typing import Dict, Any, List
import logging
import pandas as pd
import rasterio

from .config_qc_s1 import QcConfig
from .io_qc_s1 import load_and_clean_aoi, list_tiffs, mask_raster_to_aoi
from .domains_qc_s1 import get_fixed_band_mapping, compute_domain_metrics
from .report_qc_s1 import build_ranked_table, write_report


def run_qc(cfg: QcConfig, logger: logging.Logger) -> Dict[str, Any]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    domains_csv_path = cfg.out_dir / cfg.domains_csv
    report_txt_path = cfg.out_dir / cfg.report_txt
    skipped_csv_path = cfg.out_dir / cfg.skipped_csv

    aoi_gdf = load_and_clean_aoi(cfg.aoi_path)
    tiffs = list_tiffs(cfg.tiff_dir, cfg.tiff_glob)
    logger.info(f"TIFF encontrados: {len(tiffs)}")

    band_mapping = get_fixed_band_mapping()

    records: List[Dict[str, Any]] = []
    skipped_records: List[Dict[str, Any]] = []

    geom_ref = None

    for fp in tiffs:
        base = fp.name
        logger.info(f"Procesando: {base}")

        try:
            with rasterio.open(fp) as src:
                if src.count != 5:
                    msg = f"{base} omitido: tiene {src.count} bandas, se esperaban 5."
                    logger.warning(msg)
                    skipped_records.append({
                        "file": base,
                        "path": str(fp),
                        "reason": f"band_count={src.count}"
                    })
                    continue

                arr = mask_raster_to_aoi(src, aoi_gdf, all_touched=False)

                geom_fp = {
                    "crs": str(src.crs),
                    "res": tuple(src.res),
                    "transform": tuple(src.transform)
                }

                if geom_ref is None:
                    geom_ref = geom_fp

                geom_match = (
                    geom_fp["crs"] == geom_ref["crs"]
                    and geom_fp["res"] == geom_ref["res"]
                    and geom_fp["transform"] == geom_ref["transform"]
                )

                row = {
                    "file": base,
                    "path": str(fp),
                    "geom_match_ref": bool(geom_match),
                }

                metrics = compute_domain_metrics(arr, cfg.domains, band_mapping)
                row.update(metrics)

                records.append(row)

        except Exception as e:
            logger.exception(f"Error procesando {base}: {e}")
            skipped_records.append({
                "file": base,
                "path": str(fp),
                "reason": str(e)
            })
            continue

    if len(records) == 0:
        raise RuntimeError("No se pudo procesar ningún TIFF válido.")

    df_dom = pd.DataFrame(records)
    df_rank = build_ranked_table(df_dom, cfg.domains)
    df_dom.to_csv(domains_csv_path, index=False, encoding="utf-8-sig")

    if skipped_records:
        pd.DataFrame(skipped_records).to_csv(skipped_csv_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["file", "path", "reason"]).to_csv(
            skipped_csv_path, index=False, encoding="utf-8-sig"
        )

    write_report(df_rank, cfg.domains, report_txt_path)

    logger.info(f"Escenas válidas procesadas: {len(df_dom)}")
    logger.info(f"Escenas omitidas: {len(skipped_records)}")
    logger.info(f"CSV dominios: {domains_csv_path}")
    logger.info(f"Reporte: {report_txt_path}")
    logger.info(f"CSV omitidos: {skipped_csv_path}")

    return {
        "df_dom": df_dom,
        "df_rank": df_rank,
        "df_skipped": pd.DataFrame(skipped_records),
        "paths": {
            "domains_csv": str(domains_csv_path),
            "report_txt": str(report_txt_path),
            "skipped_csv": str(skipped_csv_path),
        }
    }