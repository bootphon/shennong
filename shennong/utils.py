"""Provides some utilities functions used by the *shennong* library

Those fonctions are not designed to be used by the end-user.

"""

import multiprocessing
import os
import re
import sys

import numpy as np
import pkg_resources

from shennong.logger import null_logger


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
    if njobs <= 0:
        raise ValueError(
            'njobs must be strictly positive, it is {}'.format(njobs))
    if njobs > max_njobs:
        log.warning(
            'asking %d CPU cores but reducing to %d (max available)',
            njobs, max_njobs)
        return max_njobs
    return njobs


def list2array(seq):
    """Converts lists in `seq` into numpy arrays"""
    if isinstance(seq, list):
        return np.asarray(seq)
    if isinstance(seq, dict):
        return {k: list2array(v) for k, v in seq.items()}
    return seq


def array2list(seq):
    """Converts numpy arrays in `seq` into lists"""
    if isinstance(seq, dict):
        return {
            k: array2list(v)
            for k, v in seq.items()}
    if isinstance(seq, np.ndarray):
        return seq.tolist()
    return seq


def dict_equal(dict1, dict2):
    """Returns True if `dict1` and `dict2` are equals

    The dictionnaries `dict1` and `dict2` can contain numpy arrays.

    Parameters
    ----------
    dict1 : dict
        The first dictionnary to compare
    dict2 : dict
        The second dictionnary to compare

    Returns
    -------
    equal : bool
        True if `dict1` == `dict2`, False otherwise

    """
    return array2list(dict1) == array2list(dict2)


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


class CatchExceptions:
    """Decorator wrapping a function in a try/except block

    When an exception occurs, display a user friendly message on
    standard output before exiting with error code 1.

    The detected exceptions are ValueError, OSError, RuntimeError,
    AssertionError, KeyboardInterrupt and
    pkg_resources.DistributionNotFound.

    Parameters
    ----------
    function :
        The function to wrap in a try/except block

    """
    def __init__(self, function):
        self.function = function

    def __call__(self):
        """Executes the wrapped function and catch common exceptions"""
        try:
            self.function()

        except (IOError, ValueError, OSError,
                RuntimeError, AssertionError) as err:
            self.exit('fatal error: {}'.format(err))

        except pkg_resources.DistributionNotFound:  # pragma: nocover
            self.exit(
                'fatal error: shennong package not found\n'
                'please install shennong on your system')

        except KeyboardInterrupt:
            self.exit('keyboard interruption, exiting')

    @staticmethod
    def exit(msg):
        """Write `msg` on stderr and exit with error code 1"""
        sys.stderr.write(msg.strip() + '\n')
        sys.exit(1)
