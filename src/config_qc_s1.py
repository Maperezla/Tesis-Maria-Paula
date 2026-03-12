from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


@dataclass(frozen=True)
class QcConfig:
    tiff_dir: Path
    aoi_path: Path
    tiff_glob: str
    out_dir: Path
    domains_csv: str
    report_txt: str
    skipped_csv: str
    domains: Dict[str, List[str]]


def load_config(path: Path) -> QcConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return QcConfig(
        tiff_dir=Path(cfg["tiff_dir"]),
        aoi_path=Path(cfg["aoi_path"]),
        tiff_glob=str(cfg["tiff_glob"]),
        out_dir=Path(cfg["out_dir"]),
        domains_csv=str(cfg["domains_csv"]),
        report_txt=str(cfg["report_txt"]),
        skipped_csv=str(cfg["skipped_csv"]),
        domains=dict(cfg["domains"]),
    )