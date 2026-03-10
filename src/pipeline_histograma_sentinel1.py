import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio

from .config_histogra_sentinel1 import QcConfig
from .io import load_and_clean_aoi, mask_raster_to_aoi
from .band_inference import infer_band_order
from .stats import finite_mask, descriptive_stats, plot_histogram
from .report import write_report


def list_subfolders(root_dir: Path):
    subfolders = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not subfolders:
        raise FileNotFoundError(f"No se encontraron subcarpetas en: {root_dir}")
    return subfolders


def list_tiffs_in_folder(folder: Path, extension: str = ".tif"):
    files = sorted(folder.glob(f"*{extension}"))
    if not files:
        raise FileNotFoundError(f"La subcarpeta está vacía o no contiene '{extension}': {folder}")
    return files


def run_qc_single_folder(folder: Path, cfg: QcConfig, logger: logging.Logger) -> dict:
    aoi_gdf = load_and_clean_aoi(cfg.aoi_path)

    tiffs = list_tiffs_in_folder(folder, cfg.tiff_extension)
    logger.info(f"Procesando subcarpeta {folder.name} con {len(tiffs)} TIFF")

    folder_out_dir = cfg.out_dir / folder.name
    folder_out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = folder_out_dir / cfg.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = folder_out_dir / "metricas_qc.csv"
    stats_csv = folder_out_dir / "estadistica_descriptiva.csv"
    bandmap_csv = folder_out_dir / "band_mapping_inferido.csv"
    rank_csv = folder_out_dir / "ranking_final.csv"
    report_txt = folder_out_dir / "justificacion_seleccion.txt"

    records_metrics = []
    records_stats = []
    bandmap_log = []

    geom_fingerprint_ref = None

    for fp in tiffs:
        base = fp.name
        logger.info(f"Procesando archivo {base}")

        with rasterio.open(fp) as src:
            if src.count != 5:
                raise ValueError(f"{base} tiene {src.count} bandas; se esperaban 5.")

            arr, _ = mask_raster_to_aoi(src, aoi_gdf, all_touched=False)
            bi = infer_band_order(arr)

            bandmap_log.append({
                "folder": folder.name,
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
                "folder": folder.name,
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
                records_stats.append({"folder": folder.name, "file": base, "band": name, **stats})

                out_png = plots_dir / f"{fp.stem}__{name}.png"
                plot_histogram(v, f"{folder.name} | {base} | {name}", out_png,
                               cfg.nbins, cfg.p_lo, cfg.p_hi)

    df_metrics = pd.DataFrame(records_metrics)
    df_stats = pd.DataFrame(records_stats)
    df_bandmap = pd.DataFrame(bandmap_log)

    df_metrics.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    df_stats.to_csv(stats_csv, index=False, encoding="utf-8-sig")
    df_bandmap.to_csv(bandmap_csv, index=False, encoding="utf-8-sig")

    df_rank = df_metrics.sort_values(
        by=["valid_allbands_pct", "null_anyband_pct", "geom_match_ref"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    df_rank.to_csv(rank_csv, index=False, encoding="utf-8-sig")
    write_report(df_rank, cfg.bands_used, report_txt)

    return {
        "folder": folder.name,
        "df_metrics": df_metrics,
        "df_stats": df_stats,
        "df_rank": df_rank,
        "paths": {
            "metrics_csv": str(metrics_csv),
            "stats_csv": str(stats_csv),
            "bandmap_csv": str(bandmap_csv),
            "rank_csv": str(rank_csv),
            "report_txt": str(report_txt),
            "plots_dir": str(plots_dir),
        }
    }


def run_qc_batch(cfg: QcConfig, logger: logging.Logger) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    subfolders = list_subfolders(cfg.root_tiff_dir)

    all_rankings = []
    all_metrics = []
    all_stats = []
    processed = []

    for folder in subfolders:
        result = run_qc_single_folder(folder, cfg, logger)
        processed.append(result)
        all_rankings.append(result["df_rank"])
        all_metrics.append(result["df_metrics"])
        all_stats.append(result["df_stats"])

    df_consolidated_rank = pd.concat(all_rankings, ignore_index=True)
    df_consolidated_metrics = pd.concat(all_metrics, ignore_index=True)
    df_consolidated_stats = pd.concat(all_stats, ignore_index=True)

    df_consolidated_rank = df_consolidated_rank.sort_values(
        by=["valid_allbands_pct", "null_anyband_pct", "geom_match_ref"],
        ascending=[False, True, False]
    ).reset_index(drop=True)

    global_rank_csv = cfg.out_dir / "ranking_consolidado.csv"
    global_metrics_csv = cfg.out_dir / "metricas_consolidadas.csv"
    global_stats_csv = cfg.out_dir / "estadisticas_consolidadas.csv"
    global_report_txt = cfg.out_dir / "reporte_consolidado.txt"

    df_consolidated_rank.to_csv(global_rank_csv, index=False, encoding="utf-8-sig")
    df_consolidated_metrics.to_csv(global_metrics_csv, index=False, encoding="utf-8-sig")
    df_consolidated_stats.to_csv(global_stats_csv, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("REPORTE CONSOLIDADO SENTINEL-1")
    lines.append("=" * 60)
    lines.append(f"Subcarpetas procesadas: {len(processed)}")
    lines.append("")
    lines.append("Top 10 escenas globales:")
    for i, row in df_consolidated_rank.head(10).iterrows():
        lines.append(
            f"{i+1}. {row['folder']} | {row['file']} | "
            f"valid_all={row['valid_allbands_pct']:.2f}% | "
            f"null_any={row['null_anyband_pct']:.2f}% | "
            f"geom_ok={row['geom_match_ref']}"
        )

    global_report_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "processed": processed,
        "df_consolidated_rank": df_consolidated_rank,
        "df_consolidated_metrics": df_consolidated_metrics,
        "df_consolidated_stats": df_consolidated_stats,
        "paths": {
            "global_rank_csv": str(global_rank_csv),
            "global_metrics_csv": str(global_metrics_csv),
            "global_stats_csv": str(global_stats_csv),
            "global_report_txt": str(global_report_txt),
        }
    }