#!/usr/bin/env python

"""Setup script for the shennong-features package"""

from setuptools import setup, find_packages


VERSION = open('VERSION', 'r').read().strip()


setup(
    name='shennong',
    description='A toolbox for unsupervised speech recognition',
    version=VERSION,
    packages=find_packages(),
    zip_safe=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='test'
)
