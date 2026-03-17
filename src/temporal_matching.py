from pathlib import Path
import pandas as pd


def build_image_inventory(raster_paths: list[Path]) -> pd.DataFrame:
    return pd.DataFrame({
        "raster_path": [str(p) for p in raster_paths],
        "image_name": [p.stem for p in raster_paths]
    })


def build_temporal_matches(raster_paths: list[Path], meta: pd.DataFrame) -> pd.DataFrame:
    inv = build_image_inventory(raster_paths)

    df = inv.merge(
        meta[["filename_prefix", "img_date"]],
        left_on="image_name",
        right_on="filename_prefix",
        how="left"
    ).rename(columns={"img_date": "date_actual"})

    if df["date_actual"].isna().any():
        bad = df.loc[df["date_actual"].isna(), ["image_name"]]
        raise ValueError(
            "Hay imágenes en la carpeta que no aparecen en metadata.xlsx:\n"
            f"{bad.to_string(index=False)}"
        )

    meta_ref = meta[["filename_prefix", "img_date"]].copy()

    rows = []
    for _, row in df.iterrows():
        image_name = row["image_name"]
        raster_path = row["raster_path"]
        date_actual = row["date_actual"]
        target_date = date_actual - pd.Timedelta(days=365)

        cand = meta_ref.copy()
        cand["abs_diff_days"] = (cand["img_date"] - target_date).abs().dt.days
        cand = cand.sort_values(["abs_diff_days", "img_date", "filename_prefix"]).reset_index(drop=True)

        best = cand.iloc[0]

        rows.append({
            "image_name_actual": image_name,
            "raster_path_actual": raster_path,
            "date_actual": date_actual,
            "image_name_pre": best["filename_prefix"],
            "date_pre": best["img_date"],
            "delta_days_to_target": int(best["abs_diff_days"])
        })

    return pd.DataFrame(rows)