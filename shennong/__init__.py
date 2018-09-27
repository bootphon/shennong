"""Unsupervised speech recognition toolbox for Python
==================================================

shennong is a Python module integrating unsupervised learning
algorithms applied to speech processing and recognition.

See https://coml.lscp.ens.fr/shennong for a complete documentation.

"""

import logging


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


__version__ = '0.0.dev1'
