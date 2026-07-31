# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

_ENV_KUMO_LOG = "KUMO_LOG"

_HANDLER_ATTR = '_kumorfm_handler'


def _install_handler(logger: logging.Logger) -> None:
    r"""Attaches our formatter to the ``kumorfm`` logger, and only to it.

    A library must not call :func:`logging.basicConfig`: that attaches a
    handler to the *root* logger and silently reconfigures the host
    application's logging. Handling our own records here and not propagating
    them keeps the previous console output without touching anyone else's.
    """
    if any(getattr(h, _HANDLER_ATTR, False) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt=("[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] "
                 "%(message)s"),
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    setattr(handler, _HANDLER_ATTR, True)
    logger.addHandler(handler)
    logger.propagate = False


def initialize_logging() -> None:
    r"""Initializes Kumo logging."""
    logger: logging.Logger = logging.getLogger('kumorfm')

    _install_handler(logger)

    default_level = os.getenv(_ENV_KUMO_LOG, "INFO")
    try:
        logger.setLevel(default_level)
    except (TypeError, ValueError):
        logger.setLevel(logging.INFO)
        logger.warning(
            "Logging level %s could not be properly parsed. "
            "Defaulting to INFO log level.", default_level)

    for name in ["matplotlib", "urllib3", "snowflake"]:
        logging.getLogger(name).setLevel(logging.ERROR)
