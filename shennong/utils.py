"""Provides some utilities functions"""

import logging
import sys


def null_logger():
    """Configures and returns a logger sending messages to nowhere

    This is used as default logger for some functions.

    Returns
    -------
    logging.Logger
        Logging instance ignoring all the messages.

    """
    log = logging.getLogger()
    log.addHandler(logging.NullHandler())
    return log


def get_logger(name=None, level=logging.INFO):
    """Configures and returns a logger sending messages to standard error

    Parameters
    ----------
    name : str
        Name of the created logger, to be displayed in the header of
        log messages.
    level : logging.level
        The minimum log level handled by the logger (any message above
        this level will be ignored).

    Returns
    -------
    logging.Logger
        Logging instance displaying messages to the standard error
        stream.

    """
    log = logging.getLogger(name)
    log.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    log.addHandler(handler)
    return log
