"""Provides some utilities functions"""

import logging
import numpy as np
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


def list2array(x):
    """Converts lists in `x` into numpy arrays"""
    if isinstance(x, list):
        return np.asarray(x)
    elif isinstance(x, dict):
        return {k: list2array(v) for k, v in x.items()}
    return x


def array2list(x):
    """Converts numpy arrays in `x` into lists"""
    if isinstance(x, dict):
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in x.items()}
    elif isinstance(x, np.ndarray):
        return x.tolist()
    return x


def dict_equal(x, y):
    """Returns True if `x` and `y` are equals

    The dictionnaries `x` and `y` can contain numpy arrays.

    Parameters
    ----------
    x : dict
        The first dictionnary to compare
    y : dict
        The second dictionnary to compare

    Returns
    -------
    equal : bool
        True if `x` == `y`, False otherwise

    """
    return array2list(x) == array2list(y)
