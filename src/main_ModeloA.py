from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ensure_required_keys, load_config, resolve_config_paths
from src.logging_utils import setup_logger
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta inferencia espacial RF para Landsat 8 + Sentinel-1")
    parser.add_argument(
        "--config",
        required=True,
        help="Ruta al archivo config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent

    cfg = load_config(config_path)
    cfg = resolve_config_paths(cfg, base_dir=project_root)
    ensure_required_keys(cfg)

    log_file = cfg.get("logging", {}).get("log_file", project_root / "logs" / "pipeline.log")
    logger = setup_logger(log_file=str(log_file))
    logger.info("Usando configuración: %s", config_path)

    result = run_pipeline(cfg, logger)
    logger.info("Resultado final: %s", result)


if __name__ == "__main__":
    main()