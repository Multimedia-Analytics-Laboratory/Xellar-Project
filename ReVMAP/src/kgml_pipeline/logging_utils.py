"""Logging setup helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import ensure_dir


def configure_logging(log_dir, stage_name):
    try:
        resolved_log_dir = ensure_dir(log_dir)
    except Exception:
        resolved_log_dir = ensure_dir(Path.cwd() / "logs")

    logger = logging.getLogger("kgml_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = Path(resolved_log_dir) / ("%s.log" % stage_name)
    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
