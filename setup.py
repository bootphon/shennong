#!/usr/bin/env python

"""Setup script for the shennong-features package"""

from setuptools import setup, find_packages


VERSION = open('VERSION', 'r').read().strip()


setup(
    name='shennong-features',
    version=VERSION,
    packages=find_packages,
    zip_safe=True,
)
