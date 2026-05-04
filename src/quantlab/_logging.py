"""Centralised logging configuration for quantlab.

Library code uses ``logging.getLogger(__name__)`` and never calls
``basicConfig`` itself. The CLI and notebooks call :func:`configure` once at
startup to install a handler.
"""

from __future__ import annotations

import logging
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    """Install a stream handler on the root logger if none is configured.

    Args:
        level: Logging level (e.g. ``logging.INFO``, ``logging.DEBUG``).
        fmt: Optional log-record format string. Falls back to the package default.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt or _DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Thin wrapper kept so library code never imports :mod:`logging` directly.
    """
    return logging.getLogger(name)
