from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ConfigError(Exception):
    """Error de configuración."""


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Carga un archivo JSON de configuración."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"No existe el archivo de configuración: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ConfigError("La configuración debe ser un objeto JSON (dict).")
    return cfg


def ensure_required_keys(cfg: Dict[str, Any]) -> None:
    """Valida la presencia de claves mínimas requeridas."""
    required_top = ["paths", "features", "threshold", "outputs", "validation", "window"]
    for key in required_top:
        if key not in cfg:
            raise ConfigError(f"Falta la clave requerida en config: '{key}'")

    required_paths = ["model", "l8", "s1"]
    for key in required_paths:
        if key not in cfg["paths"]:
            raise ConfigError(f"Falta paths['{key}'] en la configuración")

    required_outputs = ["prob", "bin"]
    for key in required_outputs:
        if key not in cfg["outputs"]:
            raise ConfigError(f"Falta outputs['{key}'] en la configuración")


def resolve_config_paths(cfg: Dict[str, Any], base_dir: str | Path | None = None) -> Dict[str, Any]:
    """Resuelve rutas relativas respecto a una carpeta base.

    Devuelve una copia del diccionario con rutas normalizadas a string absoluto.
    """
    if base_dir is None:
        return cfg

    base = Path(base_dir).resolve()
    out = json.loads(json.dumps(cfg))

    def _resolve(value: Any) -> Any:
        if value is None or not isinstance(value, str):
            return value
        p = Path(value)
        if p.is_absolute() or (len(value) > 1 and value[1] == ":"):
            return str(p)
        return str((base / p).resolve())

    if "paths" in out:
        for k, v in out["paths"].items():
            out["paths"][k] = _resolve(v)

    if "outputs" in out:
        for k, v in out["outputs"].items():
            out["outputs"][k] = _resolve(v)

    if "logging" in out:
        for k, v in out["logging"].items():
            out["logging"][k] = _resolve(v)

    return out