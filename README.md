# shennong

A toolbox for unsupervised speech recognition


## Installation

* First create a `shennong` Python virtual environment with conda:

        conda create --name shennong python=3.7
        conda activate shennong

* Install some dependencies:

        sudo apt install libatlas-base-dev
        conda install numpy pytest pytest-runner h5py scipy cython pyparsing
        pip install h5features

* Install `pykaldi` following the instructions
  [here](https://github.com/pykaldi/pykaldi#installation). Do **not**
  create a new virtual environment but install it in the `shennong`
  environment instead.

* Then install the `shennong` package:

        python setup.py install

* Tests the installation is working (this executes all the unit tests
  stored in the `test/` folder):

        python setup.py test

# Documentation

To build the documentation under the `doc` folder, please follow those steps.

* Install the required dependencies:

        sudo apt install texlive texlive-latex-extra dvipng
        conda install sphinx shpinx_rtd_theme
