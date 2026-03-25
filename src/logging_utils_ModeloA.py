from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str = "tesis_rf", log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configura logger con salida a consola y archivo."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Evitar handlers duplicados al ejecutar varias veces desde notebook.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger