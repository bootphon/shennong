"""Unsupervised speech recognition toolbox for Python
==================================================

shennong is a Python module integrating unsupervised learning
algorithms applied to speech processing and recognition.

See https://coml.lscp.ens.fr/shennong for a complete documentation.

"""


__version__ = '0.0.1.dev0'


def version(type=str, full=False):
    """Returns the shennong version as a string or a tuple

    The version is in format (major, minor, patch, [pre-release])

    Parameters
    ----------
    type: str or tuple, optional
        The type of the returned version, default to str

    full: bool, optional
        If True returns the full version, else returns only (major,
        minor, patch), default to False.

    """
    if type not in (str, tuple, 'str', 'tuple'):
        raise ValueError(
            'version type must be str or tuple, it is {}'.format(type))

    v = tuple(__version__.split('.'))
    if not full:
        v = v[:3]

    return v if type in (tuple, 'tuple') else '.'.join(v)
