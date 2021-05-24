#!/usr/bin/env python
"""Setup script for the shennong Python package"""

import builtins
import codecs
import setuptools

# This is a bit hackish: we are setting a global variable so that the main
# shennong __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.
builtins.__SHENNONG_SETUP__ = True

import shennong


setuptools.setup(
    # general description
    name='shennong',
    description='A toolbox for speech features extraction',
    version=shennong.version(),

    # python package dependencies
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    # packages for code and data
    packages=setuptools.find_packages(),
    package_data={'shennong': ['share/bottleneck/*', 'share/crepe/*']},

    # binaries
    entry_points={'console_scripts': [
        'speech-features = bin.speech_features:main']},

    # metadata for upload to PyPI
    author='Mathieu Bernard, INRIA',
    author_email='mathieu.a.bernard@inria.fr',
    license='GPL3',
    url=shennong.url(),
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    zip_safe=True,
)
