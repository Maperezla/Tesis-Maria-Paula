from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

@dataclass(frozen=True)
class QcConfig:
    tiff_dir: Path
    aoi_path: Path
    tiff_glob: str
    out_dir: Path
    plots_subdir: str
    nbins: int
    p_lo: float
    p_hi: float
    bands_used: List[str]

    @property
    def plots_dir(self) -> Path:
        return self.out_dir / self.plots_subdir

    @property
    def metrics_csv(self) -> Path:
        return self.out_dir / "metricas_qc.csv"

    @property
    def stats_csv(self) -> Path:
        return self.out_dir / "estadistica_descriptiva.csv"

    @property
    def bandmap_csv(self) -> Path:
        return self.out_dir / "band_mapping_inferido.csv"

    @property
    def rank_csv(self) -> Path:
        return self.out_dir / "ranking_final.csv"

    @property
    def report_txt(self) -> Path:
        return self.out_dir / "justificacion_seleccion.txt"

def load_config(path: Path) -> QcConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return QcConfig(
        tiff_dir=Path(cfg["tiff_dir"]),
        aoi_path=Path(cfg["aoi_path"]),
        tiff_glob=str(cfg["tiff_glob"]),
        out_dir=Path(cfg["out_dir"]),
        plots_subdir=str(cfg["plots_subdir"]),
        nbins=int(cfg["nbins"]),
        p_lo=float(cfg["p_lo"]),
        p_hi=float(cfg["p_hi"]),
        bands_used=list(cfg["bands_used"]),
    )