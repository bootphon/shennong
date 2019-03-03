# Shennong

[![Build Status](https://travis-ci.org/bootphon/shennong.svg?branch=master)](https://travis-ci.org/bootphon/shennong)
[![codecov](https://codecov.io/gh/bootphon/shennong/branch/master/graph/badge.svg)](https://codecov.io/gh/bootphon/shennong)

* A Python toolbox for unsupervised speech recognition.

* See the complete documentation at https://coml.lscp.ens.fr/shennong.


## Installation

This installation process is documented for Ubuntu-18.04, but it can
be adapted to other Linux distributions easilly.

* Create a `shennong` Python virtual environment with
  [conda](https://conda.io/miniconda.html):

        conda env create --name shennong -f environment.yml
        conda activate shennong

* Then install the shennong package:

        make install

* Test the installation is working (this executes all the unit tests
  stored in the `test/` folder):

        make test

* To update the `shennong` dependencies, use:

        conda env update --name shennong -f environment.yml

## Docker

You can use `shennong` from within [docker](https://docs.docker.com)
using the provided `Dockerfile`.

* Build the docker image with:

        [sudo] docker build -t shennong .

* Run an interactive session with:

        [sudo] docker run -it shennong /bin/bash

Look for more advanced usage on the official Docker documentation.


## Documentation

To build the documentation under the `doc/build` folder, please follow those steps.

* Install the required dependencies:

        sudo apt install texlive texlive-latex-extra dvipng

* Build the docs:

        make doc

* The documentation is now available at `doc/build/index.html`
