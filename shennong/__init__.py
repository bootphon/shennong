"""Unsupervised speech recognition toolbox for Python
==================================================

shennong is a Python module integrating unsupervised learning
algorithms applied to speech processing and recognition.

See https://coml.lscp.ens.fr/shennong for a complete documentation.

"""

import logging

from .utils import get_logger


log = get_logger(__name__, level=logging.INFO)

__version__ = '0.0.1-dev'
