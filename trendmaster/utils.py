"""Utility functions and exceptions for TrendMaster."""

import logging
import os
from typing import Any, Optional


# Configure logger
def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Set up a logger with specified name and level.

    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger(__name__)


# Custom Exceptions
class TrendMasterError(Exception):
    """Base exception for TrendMaster library."""
    pass


class AuthenticationError(TrendMasterError):
    """Raised when authentication with data provider fails."""
    pass


class DataLoadError(TrendMasterError):
    """Raised when data loading or preprocessing fails."""
    pass


class InstrumentNotFoundError(DataLoadError):
    """Raised when a trading instrument is not found."""
    pass


class ModelError(TrendMasterError):
    """Raised when model operations fail."""
    pass


class ConfigurationError(TrendMasterError):
    """Raised when configuration is invalid."""
    pass


def get_env_var(env_var: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set

    Returns:
        Environment variable value or default

    Raises:
        ConfigurationError: If env var not set and no default provided
    """
    value = os.getenv(env_var)
    if value is None:
        if default is None:
            raise ConfigurationError(
                f"Environment variable '{env_var}' not set and no default provided"
            )
        return default
    return value
