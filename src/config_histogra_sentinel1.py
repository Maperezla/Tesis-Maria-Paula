from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

@dataclass(frozen=True)
class QcConfig:
    root_tiff_dir: Path
    aoi_path: Path
    tiff_extension: str
    out_dir: Path
    plots_subdir: str
    nbins: int
    p_lo: float
    p_hi: float
    bands_used: List[str]

def load_config(path: Path) -> QcConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return QcConfig(
        root_tiff_dir=Path(cfg["root_tiff_dir"]),
        aoi_path=Path(cfg["aoi_path"]),
        tiff_extension=str(cfg["tiff_extension"]),
        out_dir=Path(cfg["out_dir"]),
        plots_subdir=str(cfg["plots_subdir"]),
        nbins=int(cfg["nbins"]),
        p_lo=float(cfg["p_lo"]),
        p_hi=float(cfg["p_hi"]),
        bands_used=list(cfg["bands_used"]),
    )