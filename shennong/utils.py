"""Provides some utilities functions"""

import os
import re


def list_files_with_extension(
        directory, extension,
        abspath=False, realpath=True, recursive=True):
    """Return all files of given `extension` in a `directory` hierarchy

    The files are returned as a sorted list with a path relative to
    `directory`, excepted if `abspath` or `realpath` are True

    Parameters
    ----------
    directory : str
        The directory where to look for files
    extension : str
        The files extension to look for (e.g. '.wav')
    abspath : bool, optional
        If True, return absolute path to the file/link, default to False
    realpath : bool, optional
        If True, return resolved links, default to True
    recursive : bool, optional
        If True, list files in the whole subdirectories structure, if
        False just list the top-level directory. Default to True.

    Returns
    -------
    files : list of str
        The sorted list of files found in `directory`

    """
    # the regular expression to match in filenames
    expr = r'(.*)' + extension + '$'

    # build the list of matching files
    matched = []
    if recursive:
        for path, _, files in os.walk(directory):
            matched += [os.path.join(path, f)
                        for f in files if re.match(expr, f)]
    else:
        matched = [os.path.join(directory, f)
                   for f in os.listdir(directory) if re.match(expr, f)]

    # resolve the paths if needed
    if abspath:
        matched = (os.path.abspath(m) for m in matched)
    if realpath:
        matched = (os.path.realpath(m) for m in matched)

    return sorted(matched)
