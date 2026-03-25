from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RFPipelineConfig:
    path_fire_shp: Path
    path_abs_shp: Path
    base_tiff: Path
    s1_meta_xlsx: Path
    out_dir: Path

    out_gpkg: str
    out_layer: str
    out_csv_a: str
    out_csv_b: str
    out_model_a: str
    out_model_b: str
    out_report: str

    date_col: str
    epsg_work: int
    s1_window_days: int
    s1_start_offset_days: int
    l8_month_offset: int

    seed_abs_dates: int
    seed_dataset_a: int
    seed_dataset_b: int
    seed_cv: int


def load_config(path: Path) -> RFPipelineConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return RFPipelineConfig(
        path_fire_shp=Path(cfg["path_fire_shp"]),
        path_abs_shp=Path(cfg["path_abs_shp"]),
        base_tiff=Path(cfg["base_tiff"]),
        s1_meta_xlsx=Path(cfg["s1_meta_xlsx"]),
        out_dir=Path(cfg["out_dir"]),
        out_gpkg=str(cfg["out_gpkg"]),
        out_layer=str(cfg["out_layer"]),
        out_csv_a=str(cfg["out_csv_a"]),
        out_csv_b=str(cfg["out_csv_b"]),
        out_model_a=str(cfg["out_model_a"]),
        out_model_b=str(cfg["out_model_b"]),
        out_report=str(cfg["out_report"]),
        date_col=str(cfg["date_col"]),
        epsg_work=int(cfg["epsg_work"]),
        s1_window_days=int(cfg["s1_window_days"]),
        s1_start_offset_days=int(cfg["s1_start_offset_days"]),
        l8_month_offset=int(cfg["l8_month_offset"]),
        seed_abs_dates=int(cfg["seed_abs_dates"]),
        seed_dataset_a=int(cfg["seed_dataset_a"]),
        seed_dataset_b=int(cfg["seed_dataset_b"]),
        seed_cv=int(cfg["seed_cv"]),
    )