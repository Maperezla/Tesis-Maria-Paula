from pathlib import Path
import re
import pandas as pd


# Acepta:
# S1_2015_2
# S1_2015_02
# S1_2015_2.0
PAT = re.compile(r"^(S1)_(\d{4})_([0-9]+(?:\.0+)?)$", re.IGNORECASE)


def normalize_image_name(name: str) -> str:
    """
    Normaliza nombres de imagen para permitir el cruce entre:
    - nombres de archivo en carpeta (ej. S1_2015_2.tif)
    - filename_prefix en metadata (ej. S1_2015_2.0 o S1_2015_02)

    Reglas:
    - elimina espacios
    - elimina extensión .tif/.TIF
    - normaliza el número final como entero
    - devuelve formato canónico: S1_YYYY_N
    """
    if pd.isna(name):
        return None

    s = str(name).strip()
    s = s.replace(".tif", "").replace(".TIF", "")

    m = PAT.match(s)
    if not m:
        return s.upper()

    pref = m.group(1).upper()
    year = m.group(2)
    nn = int(float(m.group(3)))  # convierte "2" o "2.0" o "02" -> 2

    return f"{pref}_{year}_{nn}"


def build_image_inventory(raster_paths: list[Path]) -> pd.DataFrame:
    """
    Construye inventario de imágenes disponibles en la carpeta.
    """
    return pd.DataFrame({
        "raster_path_actual": [str(p) for p in raster_paths],
        "image_name": [p.stem for p in raster_paths]
    })


def build_temporal_matches(raster_paths: list[Path], meta: pd.DataFrame) -> pd.DataFrame:
    """
    Construye tabla de emparejamiento temporal entre:
    - imagen actual en carpeta
    - imagen pre con fecha más cercana a (fecha_actual - 365 días)

    Requiere metadata con columnas:
    - filename_prefix
    - img_date

    Devuelve columnas:
    - image_name_actual
    - raster_path_actual
    - date_actual
    - image_name_pre
    - date_pre
    - delta_days_to_target
    """
    inv = build_image_inventory(raster_paths)

    # Normalizar nombres de imágenes de carpeta
    inv["image_name_norm"] = inv["image_name"].apply(normalize_image_name)

    # Copia local de metadata
    meta = meta.copy()

    # Normalizar nombres de metadata
    meta["filename_prefix_norm"] = meta["filename_prefix"].apply(normalize_image_name)

    # Cruce carpeta -> metadata para obtener fecha de cada imagen actual
    df = inv.merge(
        meta[["filename_prefix", "filename_prefix_norm", "img_date"]],
        left_on="image_name_norm",
        right_on="filename_prefix_norm",
        how="left"
    ).rename(columns={"img_date": "date_actual"})

    # Validar imágenes que siguen sin aparecer en metadata
    if df["date_actual"].isna().any():
        bad = df.loc[df["date_actual"].isna(), ["image_name"]]
        raise ValueError(
            "Hay imágenes en la carpeta que no aparecen en metadata.xlsx:\n"
            f"{bad.to_string(index=False)}"
        )

    # Tabla de referencia para buscar imagen pre
    meta_ref = meta[["filename_prefix", "filename_prefix_norm", "img_date"]].copy()

    rows = []

    for _, row in df.iterrows():
        image_name_actual = row["image_name"]
        raster_path_actual = row["raster_path_actual"]
        date_actual = row["date_actual"]

        target_date = date_actual - pd.Timedelta(days=365)

        cand = meta_ref.copy()
        cand["abs_diff_days"] = (cand["img_date"] - target_date).abs().dt.days
        cand = cand.sort_values(
            ["abs_diff_days", "img_date", "filename_prefix_norm"]
        ).reset_index(drop=True)

        best = cand.iloc[0]

        rows.append({
            "image_name_actual": image_name_actual,
            "raster_path_actual": raster_path_actual,
            "date_actual": date_actual,
            "image_name_pre": best["filename_prefix_norm"],
            "date_pre": best["img_date"],
            "delta_days_to_target": int(best["abs_diff_days"])
        })

    return pd.DataFrame(rows)