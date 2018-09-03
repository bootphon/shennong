# shennong-features

Speech features extraction package of the **shennong toolbox**.

## Installation

* First create a `shennong` Python virtual environment with conda:

        conda create --name shennong python=3.5
        conda activate shennong

* Install some dependencies:

        conda install numpy pytest pytest-runner h5py
        pip install h5features

* Install `pykaldi` following the instructions
  [here](https://github.com/pykaldi/pykaldi#installation). Do **not**
  create a new virtual environment but install it in the `shennong`
  environment instead.

* Then install the `shennong-features` package:

        python setup.py install

* Tests the installation is working (this executes all the unit tests
  stored in the `test/` folder):

        python setup.py test
