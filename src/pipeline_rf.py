import os
import glob
import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.errors import RasterioIOError

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from .points_loader import load_points, assign_dates_to_absences


# =========================================================
# CONFIG FIJA DE FEATURES
# =========================================================
L8_BAND_MAP: Dict[str, str] = {
    "SR_B4": "RED",
    "SR_B5": "NIR",
    "SR_B6": "SWIR_1",
    "SR_B7": "SWIR_2",
    "NDVI": "NDVI",
    "EVI": "EVI",
    "NBR": "NBR",
    "IMG_COUNT": "IMG_COUNT",
}
L8_BANDS_IN = list(L8_BAND_MAP.keys())
L8_BANDS_OUT = list(L8_BAND_MAP.values())

S1_FEATURES = ["VV", "VH", "angle"]
FEATURES = L8_BANDS_OUT + S1_FEATURES


# =========================================================
# HELPERS FECHAS
# =========================================================
def ym_neighbors(y: int, m: int) -> List[Tuple[int, int]]:
    prev_y, prev_m = (y, m - 1) if m > 1 else (y - 1, 12)
    next_y, next_m = (y, m + 1) if m < 12 else (y + 1, 1)
    return [(y, m), (prev_y, prev_m), (next_y, next_m)]


def median_lower_index(n: int) -> int:
    return (n - 1) // 2


# =========================================================
# INDEXACIÓN RASTERS
# =========================================================
def index_l8_rasters(l8_dir: str) -> Dict[Tuple[int, int], str]:
    d = {}
    for p in glob.glob(os.path.join(l8_dir, "L8_*.tif")):
        base = os.path.splitext(os.path.basename(p))[0]
        parts = base.split("_")
        if len(parts) < 3:
            continue
        try:
            y = int(parts[1])
            m = int(parts[2])
            d[(y, m)] = p
        except Exception:
            continue
    return d


def index_s1_rasters_with_dates(s1_dir: str, meta_xlsx: str, logger: logging.Logger) -> pd.DataFrame:
    meta = pd.read_excel(meta_xlsx)
    needed = {"filename_prefix", "date_yyyymmdd"}
    if not needed.issubset(meta.columns):
        raise ValueError(f"Faltan columnas {needed}. Columnas: {list(meta.columns)}")

    meta = meta.copy()
    meta["filename_prefix"] = meta["filename_prefix"].astype(str).str.strip()
    meta["filename_prefix"] = meta["filename_prefix"].str.replace(r"\.0$", "", regex=True)

    def _clean_yyyymmdd(x):
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        s = s.replace("-", "").replace("/", "")
        return s

    meta["date_str"] = meta["date_yyyymmdd"].apply(_clean_yyyymmdd)
    meta["date"] = pd.to_datetime(meta["date_str"], format="%Y%m%d", errors="coerce")

    n_meta = len(meta)
    n_parsed = int(meta["date"].notna().sum())
    meta = meta.dropna(subset=["date"]).copy()

    tif_paths = glob.glob(os.path.join(s1_dir, "S1_*.tif"))
    df_paths = pd.DataFrame({
        "filename_prefix": [os.path.splitext(os.path.basename(p))[0] for p in tif_paths],
        "path": tif_paths
    })
    n_disk = len(df_paths)

    df = df_paths.merge(meta[["filename_prefix", "date"]], on="filename_prefix", how="left")
    n_match = int(df["date"].notna().sum())

    logger.info(f"[S1] TIFF en disco: {n_disk}")
    logger.info(f"[S1] Filas Excel: {n_meta}")
    logger.info(f"[S1] Fechas parseadas OK (Excel): {n_parsed}")
    logger.info(f"[S1] Matches exactos (disco↔excel con fecha): {n_match}")

    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df = df.sort_values("date").reset_index(drop=True)
    return df


# =========================================================
# EXTRACCIÓN FEATURES
# =========================================================
def get_l8_path_for_point(t0: pd.Timestamp, l8_index: Dict[Tuple[int, int], str], month_offset: int):
    tL8 = pd.Timestamp(t0) + pd.DateOffset(months=month_offset)
    y, m = int(tL8.year), int(tL8.month)
    return l8_index.get((y, m), None), y, m


def select_s1_scene_for_point(
    t0,
    s1_df: pd.DataFrame,
    start_offset_days: int,
    window_days: int
) -> Optional[pd.Series]:
    if t0 is None or pd.isna(t0):
        return None

    t0 = pd.to_datetime(t0, errors="coerce")
    if pd.isna(t0):
        return None

    if getattr(t0, "tzinfo", None) is not None:
        t0 = t0.tz_convert(None)

    if "date" not in s1_df.columns:
        raise ValueError("s1_df no tiene columna 'date'.")

    dates = pd.to_datetime(s1_df["date"], errors="coerce")
    try:
        if hasattr(dates.dt, "tz") and dates.dt.tz is not None:
            dates = dates.dt.tz_convert(None)
    except Exception:
        pass

    start = t0 + pd.Timedelta(days=start_offset_days)
    end = t0 + pd.Timedelta(days=window_days)

    mask = (dates >= start) & (dates <= end)
    cand = s1_df.loc[mask].copy()
    if cand.empty:
        return None

    cand = cand.sort_values("date").reset_index(drop=True)
    idx = median_lower_index(len(cand))
    return cand.iloc[idx]


def sample_raster_at_points(
    raster_path: str,
    pts: gpd.GeoDataFrame,
    band_indices_1based: List[int]
) -> np.ndarray:
    with rasterio.open(raster_path) as src:
        pts_proj = pts.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in pts_proj.geometry]
        samples = list(src.sample(coords, indexes=band_indices_1based))
        arr = np.array(samples, dtype="float64")

        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan

        arr[~np.isfinite(arr)] = np.nan
        return arr


def get_l8_band_indices(src: rasterio.io.DatasetReader, band_names: List[str]) -> List[int]:
    desc = list(src.descriptions)
    if any(d is None for d in desc):
        raise ValueError("El raster L8 no tiene descriptions en todas las bandas.")

    idxs = []
    for bn in band_names:
        if bn not in desc:
            raise ValueError(f"No se encontró banda '{bn}' en descriptions. Disponibles: {desc}")
        idxs.append(desc.index(bn) + 1)
    return idxs


def build_master_features(
    g: gpd.GeoDataFrame,
    l8_index: Dict[Tuple[int, int], str],
    s1_df: pd.DataFrame,
    date_col: str,
    l8_month_offset: int,
    s1_start_offset_days: int,
    s1_window_days: int
) -> gpd.GeoDataFrame:
    g = g.copy()
    g[date_col] = pd.to_datetime(g[date_col], errors="coerce")

    s1_df = s1_df.copy()
    s1_df["date"] = pd.to_datetime(s1_df["date"], errors="coerce")
    s1_df = s1_df.dropna(subset=["date"]).copy()

    for col in L8_BANDS_OUT + S1_FEATURES:
        g[col] = np.nan

    g["l8_found"] = False
    g["l8_file"] = None
    g["l8_year"] = pd.NA
    g["l8_month"] = pd.NA

    g["s1_found"] = False
    g["s1_file"] = None
    g["s1_date"] = pd.NaT

    l8_targets = []
    for t0 in g[date_col]:
        if pd.isna(t0):
            l8_targets.append((None, pd.NA, pd.NA))
            continue
        p, y, m = get_l8_path_for_point(pd.Timestamp(t0), l8_index, l8_month_offset)
        l8_targets.append((p, y, m))

    g["_l8_path"] = [x[0] for x in l8_targets]
    g["l8_year"] = [x[1] for x in l8_targets]
    g["l8_month"] = [x[2] for x in l8_targets]

    g["l8_year"] = g["l8_year"].astype("Int64")
    g["l8_month"] = g["l8_month"].astype("Int64")

    g_valid_l8 = g.dropna(subset=["l8_year", "l8_month"]).copy()
    for (y, m), grp_idx in g_valid_l8.groupby(["l8_year", "l8_month"]).groups.items():
        path = l8_index.get((int(y), int(m)), None)
        if path is None:
            continue

        try:
            with rasterio.open(path) as src:
                band_idxs = get_l8_band_indices(src, L8_BANDS_IN)
        except RasterioIOError:
            continue

        vals = sample_raster_at_points(path, g.loc[grp_idx], band_idxs)

        for j, bn_in in enumerate(L8_BANDS_IN):
            bn_out = L8_BAND_MAP[bn_in]
            g.loc[grp_idx, bn_out] = vals[:, j]

        g.loc[grp_idx, "l8_found"] = True
        g.loc[grp_idx, "l8_file"] = os.path.basename(path)

    for i, row in g.iterrows():
        t0 = row[date_col]
        sel = select_s1_scene_for_point(
            t0=t0,
            s1_df=s1_df,
            start_offset_days=s1_start_offset_days,
            window_days=s1_window_days
        )
        if sel is None:
            continue

        path = sel["path"]
        try:
            vals = sample_raster_at_points(path, g.loc[[i]], [1, 2, 3])
        except RasterioIOError:
            continue

        g.at[i, "VV"] = vals[0, 0]
        g.at[i, "VH"] = vals[0, 1]
        g.at[i, "angle"] = vals[0, 2]
        g.at[i, "s1_found"] = True
        g.at[i, "s1_file"] = os.path.basename(path)
        g.at[i, "s1_date"] = pd.Timestamp(sel["date"])

    g.drop(columns=["_l8_path"], inplace=True)
    return g


def coverage_report(g: gpd.GeoDataFrame, features: List[str]) -> pd.DataFrame:
    rep = []
    for col in features:
        rep.append({
            "feature": col,
            "missing_n": int(g[col].isna().sum()),
            "missing_pct": float(g[col].isna().mean() * 100.0),
        })
    return pd.DataFrame(rep).sort_values("missing_pct", ascending=False)


# =========================================================
# DATASETS A / B
# =========================================================
def build_dataset_A_balanced_exact(
    g_master: gpd.GeoDataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    y_col: str = "l8_year",
    m_col: str = "l8_month",
    seed: int = 42,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    elig = g_master.dropna(subset=feature_cols).copy()
    elig[y_col] = elig[y_col].astype(int)
    elig[m_col] = elig[m_col].astype(int)

    pos = elig[elig[label_col] == 1].copy()
    neg = elig[elig[label_col] == 0].copy()

    selected_pos = []
    selected_neg = []
    used_neg = set()

    for (y, m), pos_g in pos.groupby([y_col, m_col]):
        n_need = len(pos_g)
        if n_need == 0:
            continue

        selected_pos.append(pos_g)

        ym_list = ym_neighbors(int(y), int(m))
        cand = neg[neg[[y_col, m_col]].apply(tuple, axis=1).isin(ym_list)].copy()

        if len(cand) == 0:
            if allow_replacement:
                cand = neg.copy()
            else:
                selected_pos[-1] = pos_g.iloc[0:0]
                continue

        cand_free = cand.loc[~cand.index.isin(used_neg)]

        if len(cand_free) >= n_need:
            chosen_idx = rng.choice(cand_free.index.to_numpy(), size=n_need, replace=False)
        else:
            if not allow_replacement:
                take = len(cand_free)
                selected_pos[-1] = pos_g.sample(n=take, random_state=seed) if take > 0 else pos_g.iloc[0:0]
                chosen_idx = cand_free.index.to_numpy()
            else:
                idx_list = cand_free.index.to_list()
                remaining = n_need - len(idx_list)
                extra_idx = rng.choice(cand.index.to_numpy(), size=remaining, replace=True)
                chosen_idx = np.array(idx_list + extra_idx.tolist())

        used_neg.update([i for i in chosen_idx if i in cand.index])
        selected_neg.append(neg.loc[chosen_idx].copy())

    df_pos = pd.concat(selected_pos, ignore_index=True) if selected_pos else pos.iloc[0:0].copy()
    df_neg = pd.concat(selected_neg, ignore_index=True) if selected_neg else neg.iloc[0:0].copy()

    out = pd.concat([df_pos, df_neg], ignore_index=True)
    c = out[label_col].value_counts().to_dict()
    assert c.get(1, 0) == c.get(0, 0), "No quedó balance 1:1."
    return out.drop(columns=["geometry"], errors="ignore")


def impute_by_class_median(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    fallback: str = "global"
) -> pd.DataFrame:
    df = df.copy()

    for c in feature_cols:
        med_pos = df.loc[df[label_col] == 1, c].median(skipna=True)
        med_neg = df.loc[df[label_col] == 0, c].median(skipna=True)

        if fallback == "global":
            med_global = df[c].median(skipna=True)
            if pd.isna(med_pos):
                med_pos = med_global
            if pd.isna(med_neg):
                med_neg = med_global

        mask_pos = (df[label_col] == 1) & (df[c].isna())
        mask_neg = (df[label_col] == 0) & (df[c].isna())

        df.loc[mask_pos, c] = med_pos
        df.loc[mask_neg, c] = med_neg

    return df


def build_dataset_B_balanced_exact(
    g_master: gpd.GeoDataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    y_col: str = "l8_year",
    m_col: str = "l8_month",
    seed: int = 42,
    allow_replacement: bool = True,
) -> pd.DataFrame:
    df = g_master.drop(columns=["geometry"], errors="ignore").copy()
    df = impute_by_class_median(df, feature_cols, label_col=label_col, fallback="global")

    df[y_col] = df[y_col].astype(int)
    df[m_col] = df[m_col].astype(int)

    pos = df[df[label_col] == 1].copy()
    neg = df[df[label_col] == 0].copy()

    rng = np.random.default_rng(seed)
    used_neg = set()

    selected_pos = []
    selected_neg = []

    for (y, m), pos_g in pos.groupby([y_col, m_col]):
        n_need = len(pos_g)
        if n_need == 0:
            continue

        selected_pos.append(pos_g)

        ym_list = ym_neighbors(int(y), int(m))
        cand = neg[neg[[y_col, m_col]].apply(tuple, axis=1).isin(ym_list)].copy()

        if len(cand) == 0:
            cand = neg.copy()

        cand_free = cand.loc[~cand.index.isin(used_neg)]

        if len(cand_free) >= n_need:
            chosen_idx = rng.choice(cand_free.index.to_numpy(), size=n_need, replace=False)
        else:
            if not allow_replacement:
                take = len(cand_free)
                selected_pos[-1] = pos_g.sample(n=take, random_state=seed) if take > 0 else pos_g.iloc[0:0]
                chosen_idx = cand_free.index.to_numpy()
            else:
                idx_list = cand_free.index.to_list()
                remaining = n_need - len(idx_list)
                extra_idx = rng.choice(cand.index.to_numpy(), size=remaining, replace=True)
                chosen_idx = np.array(idx_list + extra_idx.tolist())

        used_neg.update([i for i in chosen_idx if i in cand.index])
        selected_neg.append(neg.loc[chosen_idx].copy())

    df_pos = pd.concat(selected_pos, ignore_index=True) if selected_pos else pos.iloc[0:0].copy()
    df_neg = pd.concat(selected_neg, ignore_index=True) if selected_neg else neg.iloc[0:0].copy()

    out = pd.concat([df_pos, df_neg], ignore_index=True)
    counts = out[label_col].value_counts().to_dict()
    assert counts.get(1, 0) == counts.get(0, 0), "Dataset B no quedó 1:1."

    return out


# =========================================================
# CV RF
# =========================================================
def run_grouped_cv_rf(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    group_col: str = "point_id",
    n_splits: int = 10,
    random_state: int = 42,
) -> Tuple[RandomForestClassifier, pd.DataFrame, str]:
    if group_col not in df.columns:
        raise ValueError(f"Falta columna '{group_col}' en df.")

    X = df[feature_cols].to_numpy()
    y = df[label_col].astype(int).to_numpy()
    groups = df[group_col].astype(int).to_numpy()

    if HAS_SGK:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X, y, groups=groups)
        cv_name = "StratifiedGroupKFold"
    else:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X, y, groups=groups)
        cv_name = "GroupKFold"

    fold_rows = []
    cms = []
    reports = []
    importances = []

    for fold, (tr, te) in enumerate(split_iter, start=1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)

        cm = confusion_matrix(yte, yp, labels=[0, 1])
        cms.append(cm)

        fold_rows.append({
            "fold": fold,
            "cv": cv_name,
            "n_train": len(tr),
            "n_test": len(te),
            "balanced_accuracy": balanced_accuracy_score(yte, yp),
            "precision_pos": precision_score(yte, yp, pos_label=1, zero_division=0),
            "recall_pos": recall_score(yte, yp, pos_label=1, zero_division=0),
            "f1_pos": f1_score(yte, yp, pos_label=1, zero_division=0),
            "precision_neg": precision_score(yte, yp, pos_label=0, zero_division=0),
            "recall_neg": recall_score(yte, yp, pos_label=0, zero_division=0),
            "f1_neg": f1_score(yte, yp, pos_label=0, zero_division=0),
            "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1],
        })

        reports.append(
            f"--- Fold {fold} ({cv_name}) ---\n"
            + classification_report(yte, yp, target_names=["absence(0)", "presence(1)"], zero_division=0)
        )

        importances.append(model.feature_importances_)

    df_folds = pd.DataFrame(fold_rows)
    cm_sum = np.sum(np.stack(cms, axis=0), axis=0)

    metric_cols = [
        "balanced_accuracy", "precision_pos", "recall_pos", "f1_pos",
        "precision_neg", "recall_neg", "f1_neg"
    ]
    summary = df_folds[metric_cols].agg(["mean", "std"]).T

    imp_arr = np.vstack(importances)
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": imp_arr.mean(axis=0),
        "importance_std": imp_arr.std(axis=0)
    }).sort_values("importance_mean", ascending=False)

    final_model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    final_model.fit(X, y)

    txt = []
    txt.append(f"========== CV REPORT ({cv_name}) fold-by-fold ==========\n")
    txt.extend([r + "\n" for r in reports])
    txt.append("\n========== METRICS PER FOLD ==========\n")
    txt.append(df_folds.to_string(index=False))
    txt.append("\n\n========== SUMMARY (mean ± std) ==========\n")
    txt.append(summary.to_string())
    txt.append("\n\n========== AGGREGATED CONFUSION MATRIX (sum over folds) ==========\n")
    txt.append(str(cm_sum))
    txt.append("\n\n========== FEATURE IMPORTANCE (mean ± std over folds) ==========\n")
    txt.append(imp_df.to_string(index=False))

    return final_model, df_folds, "\n".join(txt)


# =========================================================
# EXPORT
# =========================================================
def _dedupe_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        key = c.lower()
        if key not in seen:
            seen[key] = 1
            new_cols.append(c)
        else:
            seen[key] += 1
            new_cols.append(f"{c}_{seen[key]}")
    df = df.copy()
    df.columns = new_cols
    return df


def export_outputs(
    g_master: gpd.GeoDataFrame,
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    modelA,
    modelB,
    reportA: str,
    reportB: str,
    out_gpkg: str,
    out_layer: str,
    out_csv_a: str,
    out_csv_b: str,
    out_model_a: str,
    out_model_b: str,
    out_report: str,
):
    g_master_out = g_master.copy()

    g_master_out["acq_date"] = pd.to_datetime(g_master_out["acq_date"], errors="coerce")
    g_master_out["s1_date"] = pd.to_datetime(g_master_out["s1_date"], errors="coerce")

    try:
        g_master_out["acq_date"] = g_master_out["acq_date"].dt.tz_localize(None)
    except Exception:
        pass
    try:
        g_master_out["s1_date"] = g_master_out["s1_date"].dt.tz_localize(None)
    except Exception:
        pass

    for c in g_master_out.columns:
        if c != "geometry" and g_master_out[c].dtype == "object":
            g_master_out[c] = g_master_out[c].astype(str)

    for cand in ["Label", "LABEL"]:
        if cand in g_master_out.columns:
            g_master_out = g_master_out.rename(columns={cand: f"{cand}_src"})

    g_master_out = _dedupe_columns_case_insensitive(g_master_out)

    g_master_out.to_file(out_gpkg, layer=out_layer, driver="GPKG")
    dfA.to_csv(out_csv_a, index=False)
    dfB.to_csv(out_csv_b, index=False)
    joblib.dump(modelA, out_model_a)
    joblib.dump(modelB, out_model_b)

    with open(out_report, "w", encoding="utf-8") as f:
        f.write("########################\n# DATASET A\n########################\n\n")
        f.write(reportA)
        f.write("\n\n\n########################\n# DATASET B\n########################\n\n")
        f.write(reportB)

# =========================================================
# PIPELINE PRINCIPAL
# =========================================================
def run_pipeline(cfg, logger: logging.Logger):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    l8_dir = str(cfg.base_tiff / "L8")
    s1_dir = str(cfg.base_tiff / "S1_V2")   # <- corregir si esta es la carpeta real

    g_points = load_points(
        path_fire_shp=str(cfg.path_fire_shp),
        path_abs_shp=str(cfg.path_abs_shp),
        date_col=cfg.date_col,
        epsg_work=cfg.epsg_work,
    )

    logger.info(f"Ruta incendios: {cfg.path_fire_shp}")
    logger.info(f"Ruta ausencias: {cfg.path_abs_shp}")
    logger.info(f"CRS puntos combinados: {g_points.crs}")
    logger.info(f"Incendios: {(g_points['label'] == 1).sum()}")
    logger.info(f"Ausencias: {(g_points['label'] == 0).sum()}")
    logger.info(f"Total puntos: {len(g_points)}")

    g_points = assign_dates_to_absences(
        g_points,
        date_col=cfg.date_col,
        seed=cfg.seed_abs_dates
    )

    l8_index = index_l8_rasters(l8_dir)
    s1_index = index_s1_rasters_with_dates(s1_dir, str(cfg.s1_meta_xlsx), logger)

    logger.info(f"L8 meses indexados: {len(l8_index)}")
    logger.info(f"S1 escenas indexadas: {len(s1_index)}")

    g_master = build_master_features(
        g=g_points,
        l8_index=l8_index,
        s1_df=s1_index,
        date_col=cfg.date_col,
        l8_month_offset=cfg.l8_month_offset,
        s1_start_offset_days=cfg.s1_start_offset_days,
        s1_window_days=cfg.s1_window_days,
    )

    cov = coverage_report(g_master, FEATURES)
    logger.info("Coverage report:\n" + cov.to_string(index=False))

    dfA = build_dataset_A_balanced_exact(
        g_master,
        feature_cols=FEATURES,
        seed=cfg.seed_dataset_a,
        allow_replacement=True
    )

    dfB = build_dataset_B_balanced_exact(
        g_master,
        feature_cols=FEATURES,
        seed=cfg.seed_dataset_b,
        allow_replacement=True
    )

    logger.info(f"Dataset A shape: {dfA.shape}")
    logger.info(f"Dataset B shape: {dfB.shape}")

    print("=== DEBUG PRE-CV ===")
    print("g_points:", g_points.shape)
    print("g_master:", g_master.shape)
    print("FEATURES:", FEATURES)

    print("\nConteo label en g_master:")
    print(g_master["label"].value_counts(dropna=False))

    print("\nL8 found:")
    print(g_master["l8_found"].value_counts(dropna=False))

    print("\nS1 found:")
    print(g_master["s1_found"].value_counts(dropna=False))

    print("\nNaN por feature en g_master:")
    print(g_master[FEATURES].isna().sum())

    print("\nComplete cases en FEATURES:")
    tmp_complete = g_master.dropna(subset=FEATURES)
    print(tmp_complete.shape)
    if len(tmp_complete) > 0:
        print(tmp_complete["label"].value_counts(dropna=False))

    print("\ndfA shape:", dfA.shape)
    if len(dfA) > 0:
        print(dfA["label"].value_counts(dropna=False))
        print("point_id únicos en dfA:", dfA["point_id"].nunique())

    print("\ndfB shape:", dfB.shape)
    if len(dfB) > 0:
        print(dfB["label"].value_counts(dropna=False))
        print("point_id únicos en dfB:", dfB["point_id"].nunique())

    # =========================
    # VALIDACIONES PREVIAS A CV
    # =========================
    if dfA.empty:
        raise ValueError(
            "dfA quedó vacío antes de CV. "
            "Revisa cobertura de FEATURES, balanceo por mes y disponibilidad de L8/S1."
        )

    if dfB.empty:
        raise ValueError(
            "dfB quedó vacío antes de CV. "
            "Revisa imputación, columnas temporales (l8_year/l8_month) y balanceo."
        )

    if dfA["label"].nunique() < 2:
        raise ValueError(
            f"dfA no tiene ambas clases. Conteo actual: {dfA['label'].value_counts(dropna=False).to_dict()}"
        )

    if dfB["label"].nunique() < 2:
        raise ValueError(
            f"dfB no tiene ambas clases. Conteo actual: {dfB['label'].value_counts(dropna=False).to_dict()}"
        )

    if dfA["point_id"].nunique() < 10:
        raise ValueError(
            f"dfA tiene menos de 10 grupos únicos para CV 10-fold. point_id únicos: {dfA['point_id'].nunique()}"
        )

    if dfB["point_id"].nunique() < 10:
        raise ValueError(
            f"dfB tiene menos de 10 grupos únicos para CV 10-fold. point_id únicos: {dfB['point_id'].nunique()}"
        )

    # =========================
    # ENTRENAMIENTO / CV
    # =========================
    modelA, foldsA, reportA = run_grouped_cv_rf(
        dfA,
        FEATURES,
        label_col="label",
        group_col="point_id",
        n_splits=10,
        random_state=cfg.seed_cv
    )

    modelB, foldsB, reportB = run_grouped_cv_rf(
        dfB,
        FEATURES,
        label_col="label",
        group_col="point_id",
        n_splits=10,
        random_state=cfg.seed_cv
    )

    # =========================
    # EXPORTACIÓN
    # =========================
    export_outputs(
        g_master=g_master,
        dfA=dfA,
        dfB=dfB,
        modelA=modelA,
        modelB=modelB,
        reportA=reportA,
        reportB=reportB,
        out_gpkg=str(cfg.out_dir / cfg.out_gpkg),
        out_layer=cfg.out_layer,
        out_csv_a=str(cfg.out_dir / cfg.out_csv_a),
        out_csv_b=str(cfg.out_dir / cfg.out_csv_b),
        out_model_a=str(cfg.out_dir / cfg.out_model_a),
        out_model_b=str(cfg.out_dir / cfg.out_model_b),
        out_report=str(cfg.out_dir / cfg.out_report),
    )

    logger.info(f"Outputs generados en: {cfg.out_dir}")

    return {
        "g_points": g_points,
        "g_master": g_master,
        "dfA": dfA,
        "dfB": dfB,
        "foldsA": foldsA,
        "foldsB": foldsB,
        "features": FEATURES,
        "coverage": cov,
        "out_dir": str(cfg.out_dir),
    }
