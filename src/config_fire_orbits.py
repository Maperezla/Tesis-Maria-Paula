from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class FireOrbitConfig:
    points_shp: Path
    meta_xlsx: Path
    out_dir: Path
    out_name: str
    col_year: str
    col_month: str
    col_day: str
    col_img_date: str
    col_orbit: str
    col_sysindex: str
    col_slice: str
    slice_preference: list
    prefer_platform: str
    post_min_days: int
    post_max_days: int
    pre_min_days: int
    pre_target_days: int

def load_config(path: Path) -> FireOrbitConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return FireOrbitConfig(
        points_shp=Path(cfg["points_shp"]),
        meta_xlsx=Path(cfg["meta_xlsx"]),
        out_dir=Path(cfg["out_dir"]),
        out_name=str(cfg["out_name"]),
        col_year=str(cfg["col_year"]),
        col_month=str(cfg["col_month"]),
        col_day=str(cfg["col_day"]),
        col_img_date=str(cfg["col_img_date"]),
        col_orbit=str(cfg["col_orbit"]),
        col_sysindex=str(cfg["col_sysindex"]),
        col_slice=str(cfg["col_slice"]),
        slice_preference=list(cfg["slice_preference"]),
        prefer_platform=str(cfg["prefer_platform"]),
        post_min_days=int(cfg["post_min_days"]),
        post_max_days=int(cfg["post_max_days"]),
        pre_min_days=int(cfg["pre_min_days"]),
        pre_target_days=int(cfg["pre_target_days"]),
    )