import pandas as pd


def load_metadata(metadata_xlsx: str) -> pd.DataFrame:
    meta = pd.read_excel(metadata_xlsx)

    required = {"filename_prefix", "date_yyyymmdd"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en metadata: {missing}")

    meta = meta.copy()
    meta["filename_prefix"] = meta["filename_prefix"].astype(str).str.strip()
    meta["img_date"] = pd.to_datetime(
        meta["date_yyyymmdd"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )

    if meta["img_date"].isna().any():
        bad = meta.loc[meta["img_date"].isna(), ["filename_prefix", "date_yyyymmdd"]].head(10)
        raise ValueError(
            "Hay filas inválidas en date_yyyymmdd. Ejemplos:\n"
            f"{bad.to_string(index=False)}"
        )

    return meta.sort_values("img_date").reset_index(drop=True)