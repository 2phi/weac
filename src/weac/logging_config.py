"""
Logging configuration for weak layer anticrack nucleation model.
"""

import os
from logging.config import dictConfig


def setup_logging(level: str | None = None) -> None:
    """
    Initialise the global logging configuration exactly once.
    The level is taken from the env var WEAC_LOG_LEVEL (default WARNING).
    """
    if level is None:
        level = os.getenv("WEAC_LOG_LEVEL", "WARNING").upper()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # keep third-party loggers alive
            "formatters": {
                "console": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "console",
                    "level": level,
                },
            },
            "root": {  # applies to *all* loggers
                "handlers": ["console"],
                "level": level,
            },
        }
    )
