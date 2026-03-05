import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio

from .config import QcConfig
from .io import load_and_clean_aoi, list_tiffs, mask_raster_to_aoi
from .band_inference import infer_band_order
from .stats import finite_mask, descriptive_stats, plot_histogram
from .report import write_report

def run_qc(cfg: QcConfig, logger: logging.Logger) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

    aoi_gdf = load_and_clean_aoi(cfg.aoi_path)
    tiffs = list_tiffs(cfg.tiff_dir, cfg.tiff_glob)

    logger.info(f"Encontrados {len(tiffs)} TIFF -> pattern={cfg.tiff_glob}")

    records_metrics = []
    records_stats = []
    bandmap_log = []

    geom_fingerprint_ref = None

    for fp in tiffs:
        base = fp.name
        logger.info(f"Procesando {base}")

        with rasterio.open(fp) as src:
            if src.count != 5:
                raise ValueError(f"{base} tiene {src.count} bandas; se esperaban 5.")

            arr, _ = mask_raster_to_aoi(src, aoi_gdf, all_touched=False)
            bi = infer_band_order(arr)

            bandmap_log.append({
                "file": base,
                "mapping": bi.mapping,
                "confidence": bi.confidence,
                "notes": "; ".join(bi.notes) if bi.notes else ""
            })

            geom_fp = {"crs": str(src.crs), "res": tuple(src.res), "transform": tuple(src.transform)}
            if geom_fingerprint_ref is None:
                geom_fingerprint_ref = geom_fp

            geom_match = (
                geom_fp["crs"] == geom_fingerprint_ref["crs"]
                and geom_fp["res"] == geom_fingerprint_ref["res"]
                and geom_fp["transform"] == geom_fingerprint_ref["transform"]
            )

            # asegurar bandas requeridas
            required = cfg.bands_used
            all_idx = set(range(1, 6))
            used_idx = set(bi.mapping.values())
            remaining_idx = sorted(list(all_idx - used_idx))

            for key in required:
                if key not in bi.mapping:
                    if not remaining_idx:
                        raise ValueError(f"No hay bandas disponibles para asignar '{key}'. Mapping={bi.mapping}")
                    bi.mapping[key] = remaining_idx.pop(0)
                    bi.confidence[key] = 0.05

            named = {name: arr[bi.mapping[name] - 1] for name in required}

            total_pixels = int(next(iter(named.values())).size)
            per_band = {}
            for name in required:
                m = finite_mask(named[name])
                valid = int(np.sum(m))
                nodata = total_pixels - valid
                per_band[f"{name}_valid_px"] = valid
                per_band[f"{name}_valid_pct"] = 100.0 * valid / total_pixels
                per_band[f"{name}_null_px"] = nodata
                per_band[f"{name}_null_pct"] = 100.0 * nodata / total_pixels

            valid_all = np.ones_like(next(iter(named.values())), dtype=bool)
            for name in required:
                valid_all &= finite_mask(named[name])

            valid_all_px = int(np.sum(valid_all))
            null_any_px = total_pixels - valid_all_px

            records_metrics.append({
                "file": base,
                "path": str(fp),
                "total_px_aoi": total_pixels,
                "bands_used": ",".join(required),
                "valid_allbands_px": valid_all_px,
                "valid_allbands_pct": 100.0 * valid_all_px / total_pixels,
                "null_anyband_px": null_any_px,
                "null_anyband_pct": 100.0 * null_any_px / total_pixels,
                "geom_match_ref": bool(geom_match),
                "band_mapping": json.dumps(bi.mapping, ensure_ascii=False),
                "band_confidence": json.dumps(bi.confidence, ensure_ascii=False),
                **per_band
            })

            for name in required:
                v = named[name].ravel()
                stats = descriptive_stats(v)
                records_stats.append({"file": base, "band": name, **stats})

                out_png = cfg.plots_dir / f"{fp.stem}__{name}.png"
                plot_histogram(v, f"{base} | {name}", out_png, cfg.nbins, cfg.p_lo, cfg.p_hi)

    df_metrics = pd.DataFrame(records_metrics)
    df_stats = pd.DataFrame(records_stats)
    df_bandmap = pd.DataFrame(bandmap_log)

    df_metrics.to_csv(cfg.metrics_csv, index=False, encoding="utf-8-sig")
    df_stats.to_csv(cfg.stats_csv, index=False, encoding="utf-8-sig")
    df_bandmap.to_csv(cfg.bandmap_csv, index=False, encoding="utf-8-sig")

    # ranking + reporte
    df_rank = df_metrics.sort_values(
        by=["valid_allbands_pct", "null_anyband_pct", "geom_match_ref"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    df_rank.to_csv(cfg.rank_csv, index=False, encoding="utf-8-sig")
    write_report(df_rank, cfg.bands_used, cfg.report_txt)

    return {
        "df_metrics": df_metrics,
        "df_stats": df_stats,
        "df_rank": df_rank,
        "paths": {
            "metrics_csv": str(cfg.metrics_csv),
            "stats_csv": str(cfg.stats_csv),
            "bandmap_csv": str(cfg.bandmap_csv),
            "rank_csv": str(cfg.rank_csv),
            "report_txt": str(cfg.report_txt),
            "plots_dir": str(cfg.plots_dir),
        }
    }