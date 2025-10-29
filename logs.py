import logging

logger = logging.getLogger(__name__)


def log_info(message: str):
    """Log an info message."""
    logger.info(message)


def log_exception(message: str):
    """Log an exception message."""
    logger.exception(message)
