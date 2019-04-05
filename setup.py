#!/usr/bin/env python
"""Setup script for the shennong toolbox package"""

import codecs
import setuptools
import shennong


setuptools.setup(
    # general description
    name='shennong',
    description='A toolbox for unsupervised speech recognition',
    version=shennong.version(),

    # python package dependancies
    setup_requires=['pytest-runner', 'numpy', 'scipy'],
    tests_require=['pytest'],

    # packages for code and data
    packages=setuptools.find_packages(),
    package_data={'shennong': ['share/bottleneck/*']},

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
