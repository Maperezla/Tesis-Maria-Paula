from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RFPipelineConfig:
    path_fire_shp: Path
    path_abs_shp: Path

    base_tiff: Path
    l8_dir: Path
    s1_dir: Path
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

    # Landsat 8
    l8_use_bimonth_logic: bool
    l8_switch_day: int
    l8_fallback_next_available: bool

    # Sentinel-1 general
    s1_window_days: int
    s1_start_offset_days: int
    s1_select_mode: str

    # Modelo C
    model_c_enabled: bool
    s1_difference_same_month: bool
    s1_difference_selection_strategy: str

    out_gpkg_c: str
    out_layer_c: str
    out_csv_c: str
    out_model_c: str
    out_report_c: str

    # Seeds
    seed_abs_dates: int
    seed_dataset_a: int
    seed_dataset_b: int
    seed_dataset_c: int
    seed_cv: int


def load_config(path: Path) -> RFPipelineConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))

    base_tiff = Path(cfg["base_tiff"])

    return RFPipelineConfig(
        path_fire_shp=Path(cfg["path_fire_shp"]),
        path_abs_shp=Path(cfg["path_abs_shp"]),

        base_tiff=base_tiff,
        l8_dir=Path(cfg["l8_dir"]),
        s1_dir=Path(cfg["s1_dir"]),
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

        l8_use_bimonth_logic=bool(cfg.get("l8_use_bimonth_logic", True)),
        l8_switch_day=int(cfg.get("l8_switch_day", 14)),
        l8_fallback_next_available=bool(cfg.get("l8_fallback_next_available", True)),

        s1_window_days=int(cfg["s1_window_days"]),
        s1_start_offset_days=int(cfg["s1_start_offset_days"]),
        s1_select_mode=str(cfg.get("s1_select_mode", "next_available_after_fire")),

        model_c_enabled=bool(cfg.get("model_c_enabled", True)),
        s1_difference_same_month=bool(cfg.get("s1_difference_same_month", True)),
        s1_difference_selection_strategy=str(
            cfg.get("s1_difference_selection_strategy", "closest_to_fire_date")
        ),

        out_gpkg_c=str(cfg.get("out_gpkg_c", "puntos_features_model_C_260507.gpkg")),
        out_layer_c=str(cfg.get("out_layer_c", "puntos_features_model_C_260507")),
        out_csv_c=str(cfg.get("out_csv_c", "dataset_C_complete_case_260507.csv")),
        out_model_c=str(cfg.get("out_model_c", "modelo_RF_C_260507.pkl")),
        out_report_c=str(cfg.get("out_report_c", "reporte_cv_C_260507.txt")),

        seed_abs_dates=int(cfg["seed_abs_dates"]),
        seed_dataset_a=int(cfg["seed_dataset_a"]),
        seed_dataset_b=int(cfg["seed_dataset_b"]),
        seed_dataset_c=int(cfg.get("seed_dataset_c", cfg["seed_dataset_a"])),
        seed_cv=int(cfg["seed_cv"]),
    )