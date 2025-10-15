"""
Utility functions for the Whisper+MFA dataset building pipeline.

This module provides common utility functions used throughout the pipeline:
- Logging configuration and setup
- System executable validation
- Time formatting utilities
- Safe file writing operations

These utilities help ensure consistent behaviour and error handling across
all pipeline components.
"""
import logging


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Configure basic logging with a reasonable format.

    Sets up the Python logging system with a consistent format for all
    pipeline components. This ensures uniform log output across the entire
    application by configuring the root logger.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG)

    Returns:
        A logger instance that can be used for logging. Each module should
        create its own logger with logging.getLogger(__name__) after calling
        this function.

    Example:
        >>> setup_logger(logging.DEBUG)
        >>> logger = logging.getLogger(__name__)
    """
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level=level)

    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Add formatter to ch
    ch.setFormatter(formatter)
    # Add ch to root logger
    root_logger.addHandler(ch)

    # Return a logger that modules can use (they should create their own with getLogger(__name__))
    return logging.getLogger(__name__)
