from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class S1TimeseriesConfig:
    gpkg_path: Path
    layer_name: str
    raster_dir: Path
    metadata_xlsx: Path
    out_dir: Path
    epsilon: float
    n_sample: int
    random_state: int
    min_valid_frac: float
    band_names: list[str]
    derived_bands: list[str]


def load_config(path: Path) -> S1TimeseriesConfig:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return S1TimeseriesConfig(
        gpkg_path=Path(cfg["gpkg_path"]),
        layer_name=str(cfg["layer_name"]),
        raster_dir=Path(cfg["raster_dir"]),
        metadata_xlsx=Path(cfg["metadata_xlsx"]),
        out_dir=Path(cfg["out_dir"]),
        epsilon=float(cfg["epsilon"]),
        n_sample=int(cfg["n_sample"]),
        random_state=int(cfg["random_state"]),
        min_valid_frac=float(cfg["min_valid_frac"]),
        band_names=list(cfg["band_names"]),
        derived_bands=list(cfg["derived_bands"]),
    )