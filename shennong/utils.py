"""Provides some utilities functions used by the *shennong* library

Those fonctions are not designed to be used by the end-user.

"""

import logging
import multiprocessing
import numpy as np
import os
import re
import sys


_logger = logging.getLogger()


def null_logger():
    """Configures and returns a logger sending messages to nowhere

    This is used as default logger for some functions.

    Returns
    -------
    logging.Logger
        Logging instance ignoring all the messages.

    """
    _logger.handlers = []
    _logger.addHandler(logging.NullHandler())
    return _logger


def get_logger(name=None, level='info',
               formatter='%(levelname)s - %(message)s'):
    """Configures and returns a logger sending messages to standard error

    Parameters
    ----------
    name : str
        Name of the created logger, to be displayed in the header of
        log messages.
    level : str, optional
        The minimum log level handled by the logger (any message above
        this level will be ignored). Must be 'debug', 'info',
        'warning' or 'error'. Default to 'info'.
    formatter : str, optional
        A string to format the log messages, see
        https://docs.python.org/3/library/logging.html#formatter-objects. By
        default display level and message. Use '%(asctime)s -
        %(levelname)s - %(name)s - %(message)s' to display time,
        level, name and message.

    Returns
    -------
    logging.Logger
        A configured logging instance displaying messages to the
        standard error stream.

    Raises
    ------
    ValueError
        If the logging `level` is not 'debug', 'info', 'warning' or
        'error'.

    """
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR}

    _logger.name = name
    try:
        _logger.setLevel(levels[level])
    except KeyError:
        raise ValueError(
            'invalid logging level "{}", must be in {}'.format(
                level, ', '.join(levels.keys())))

    formatter = logging.Formatter(formatter)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    _logger.handlers = []
    _logger.addHandler(handler)
    return _logger


def get_njobs(njobs, log=null_logger()):
    """Returns the number of parallel jobs to run

    The returned number of jobs is adapted from the input `njobs`
    value, considering the number of CPU cores available on the
    machine.

    Parameters
    ----------
    njobs : int
        The desired number of jobs to use.
    log : logging.Logger, optional
        A logger where to send messages, no logging by default.

    Returns
    -------
    njobs : int
        The returned value is min(njobs, ncpus).

    Raises
    ------
    ValueError
        If `njobs` is not a strictly positive integer.

    """
    max_njobs = multiprocessing.cpu_count()
    if njobs is None:
        return max_njobs
    elif njobs <= 0:
        raise ValueError(
            'njobs must be strictly positive, it is {}'.format(njobs))
    elif njobs > max_njobs:
        log.warning(
            'asking %d CPU cores but reducing to %d (max available)',
            njobs, max_njobs)
        return max_njobs
    return njobs


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


def list_files_with_extension(
        directory, extension,
        abspath=False, realpath=True, recursive=True):
    """Return all files of given extension in directory hierarchy

    Parameters
    ----------
    directory : str
        The directory where to search for files
    extension : str
        The extension of the targeted files (e.g. '.wav')
    abspath : bool, optional
        If True, return the absolute path to the file/link, default to
        False.
    realpath : bool, optional
        If True, return resolved links, default to True.
    recursive : bool, optional
        If True, list files in the whole subdirectories tree, if False
        just list the top-level directory, default to True.

    Returns
    -------
    files : list
        The files are returned in a sorted list with a path relative
        to 'directory', except if `abspath` or `realpath` is True

    """
    # the regular expression to match in filenames
    expr = r'(.*)' + extension + '$'

    # build the list of matching files
    if recursive:
        matched = []
        for path, _, files in os.walk(directory):
            matched += [
                os.path.join(path, f) for f in files if re.match(expr, f)]
    else:
        matched = (
            os.path.join(directory, f)
            for f in os.listdir(directory) if re.match(expr, f))

    if abspath:
        matched = (os.path.abspath(m) for m in matched)
    if realpath:
        matched = (os.path.realpath(m) for m in matched)
    return sorted(matched)
