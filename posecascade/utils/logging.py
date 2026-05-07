"""Logging configuration and accessors.

Use :func:`get_logger` to obtain a module-scoped logger; never call ``print``
in production code.
"""
from __future__ import annotations

import logging
from logging import Logger

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Install the root handler. Idempotent — safe to call multiple times."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> Logger:
    """Return the module-scoped logger for ``name``."""
    return logging.getLogger(name)
