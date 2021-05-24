.. _installation:


Installation
============

.. note::

   Shennong is developed for Python3 and is **not compatible with
   Python2**.


Installation on Linux
---------------------


Installation from conda
~~~~~~~~~~~~~~~~~~~~~~~

This is the recommended installation procedure, compatible with python
3.6 and 3.7.

* Install `conda <https://conda.io/miniconda.html>`_ on your machine,

* Install ``shennong`` from its `conda package
  <https://anaconda.org/coml/shennong>`_::

    conda install -c coml -c conda-forge shennong


Installation from sources
~~~~~~~~~~~~~~~~~~~~~~~~~

* Install `conda <https://conda.io/miniconda.html>`_ on your machine,

* Cloned the `shennong repository
  <https://github.com/bootphon/shennong>`_ and move at its root
  directory::

     git clone https://github.com/bootphon/shennong.git
     cd ./shennong

* Install ``shennong`` in a new virtual environment::

    conda env create --name shennong -f environment.yml
    conda activate shennong
    make install
    make test

* **Or**, if you need to work in an existing environment, add
  ``shennong`` in it::

    conda env update --name $MYENV -f environment.yml
    conda activate $MYENV
    make install
    make test

* To upgrade your installation to the latest version available use::

    git pull origin master
    conda env update --name shennong -f environment.yml
    make install
    make test

.. note::

   To use ``shennong``, do not forget to activate the virtual environment::

     conda activate shennong


Installation on MacOS
---------------------

As `pykaldi <https://github.com/bootphon/shennong-pykaldi>`_ does not provide a
conda image for macos, you must install from sources.

* Install `conda <https://conda.io/miniconda.html>`_ and `brew
  <https://brew.sh/>`_ on your machine,

* Cloned the `shennong repository
  <https://github.com/bootphon/shennong>`_ and move at its root
  directory::

     git clone https://github.com/bootphon/shennong.git
     cd ./shennong

* Install ``pykaldi`` and ``shennong`` in a virtual environment::

    # install shennong dependencies (all excepted pykaldi)
    sed '/pykaldi/d' environment.yml > environment.macos.yml
    conda env create --name shennong -f environment.macos.yml
    conda activate shennong

    # install the shennong fork of pykaldi
    # (see https://github.com/bootphon/shennong-pykaldi#from-source)
    git clone https://github.com/bootphon/shennong-pykaldi.git ./pykaldi
    cd pykaldi
    brew install automake cmake git graphviz libtool pkg-config wget
    pip install --upgrade pip setuptools numpy pyparsing ninja
    cd tools
    ./check_dependencies.sh
    ./install_protobuf.sh
    ./install_clif.sh
    ./install_kaldi.sh
    cd ..
    python setup.py install
    cd ..

    # install shennong
    make install
    make test


Installation on Windows
-----------------------

* As `pykaldi <https://github.com/bootphon/shennong-pykaldi>`_ is **not
  supported on Windows** we do not provide installation instructions.

* Windows users can instead install shennong in a `Windows subsystem
  for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_ or
  within a docker container as documented :ref:`below
  <install_docker>`.


.. _install_docker:

Installation on Docker
----------------------

You can use ``shennong`` from within `docker
<https://docs.docker.com>`_ using the provided ``Dockerfile``.

* Build the docker image with::

    [sudo] docker build -t shennong .

* Run an interactive session with::

    [sudo] docker run -it shennong /bin/bash

* Or you can open a jupyter notebook as follow::

    [sudo] docker run -p 9000:9000 shennong \
        jupyter notebook --no-browser --ip=0.0.0.0 --port=9000 --allow-root

  Then open ``http://localhost:9000`` in your usual web browser.

Look for more advanced usage on the official `Docker documentation
<https://docs.docker.com>`_.


Build the documentation
-----------------------

To build the documentation under the ``doc/build`` folder, follow
those steps.

* Install the required dependencies::

        sudo apt install texlive texlive-latex-extra dvipng

* Build the docs::

        make doc

* The documentation is now available at ``doc/build/html/index.html``
