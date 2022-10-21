"""Benchmark logging."""
import logging

FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(FORMATTER)

LOGGER_TABLE = {}

def get_logger(name):
    """Attach to the default logger."""

    if name in LOGGER_TABLE:
        return LOGGER_TABLE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(STREAM_HANDLER)

    LOGGER_TABLE[name] = logger
    return logger
