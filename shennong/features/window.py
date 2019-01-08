"""Implementation of different types of window functions

This is usefull when computing frames for features extraction. Uses
the kaldi implementation.

The implemented window functions :math:`w(n)` are, with `length` noted
:math:`N`:

* **rectangular**:

  .. math::

     w(n) = 1

* **hanning**:

  .. math::

     w(n) = \\frac{1}{2} - \\frac{1}{2} cos(\\frac{2\\pi n}{N-1})

* **hamming**:

  .. math::

     w(n) = 0.54 - 0.46 cos(\\frac{2\\pi n}{N-1})

* **povey** (like `hamming` but goes to zero at edges):

  .. math::

     w(n) = (\\frac{1}{2} - \\frac{1}{2} cos(\\frac{2\\pi n}{N-1}))^{0.85}

* **blackman**, with `blackman_coeff` noted as :math:`\\alpha`:

  .. math::

     w(n) = \\alpha - \\frac{1}{2} cos(\\frac{2\\pi n}{N-1}) + \
            (\\frac{1}{2} - \\alpha) cos(\\frac{4\\pi n}{N-1})

Examples
--------

>>> from shennong.features.window import window
>>> window(5, type='hamming')
array([0.08, 0.54, 1.  , 0.54, 0.08], dtype=float32)
>>> window(5, type='rectangular')
array([1., 1., 1., 1., 1.], dtype=float32)
>>> window(5, type='povey').tolist()
[0.0, 0.5547847151756287, 1.0, 0.5547847151756287, 0.0]

"""

import numpy as np

import kaldi.matrix
import kaldi.feat.window


def types():
    """Returns the supported window functions as a list"""
    return sorted(['povey', 'hanning', 'hamming', 'rectangular', 'blackman'])


def window(length, type='povey', blackman_coeff=0.42):
    """Returns a window of the given `type` and `length`

    Parameters
    ----------
    length : int
        The size of the window, in number of samples
    type : {'povey', 'hanning', 'hamming', 'rectangular', 'blackman'}
        The type of the window, default is 'povey' (like hamming but
        goes to zero at edges)
    blackman_coeff : float, optional
        The constant coefficient for generalized Blackman window, used
        only when `type` is 'blackman'

    Returns
    -------
    window : array, shape = [length, 1]
        The window with the given `type` and `length`

    Raises
    ------
    ValueError
        If the `type` is not valid or `length <= 1`

    """
    if int(length) <= 0:
        raise ValueError(
            'length must be strictly positive but is {}'.format(length))

    if type not in types():
        raise ValueError(
            'type must be in {} but is {}'.format(types, type))

    # special case where the window has only one sample (in kaldi this
    # returns nan)
    if length == 1:
        return np.ones((1,))

    # special case where the window has only two samples, for povey,
    # hanning and blackman this leads to a zeros only window.
    if length == 2 and type in ('povey', 'blackman', 'hanning'):
        return np.ones((2,))

    opt = kaldi.feat.window.FrameExtractionOptions()
    opt.samp_freq = 1000
    opt.frame_length_ms = length  # samp_freq * 0.001 * length
    opt.window_type = type
    opt.blackman_coeff = blackman_coeff

    window = kaldi.feat.window.FeatureWindowFunction().from_options(opt).window
    return kaldi.matrix.SubVector(window).numpy()
