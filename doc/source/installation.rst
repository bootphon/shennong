.. _installation:


Installation
============

.. note::

   Shennong is developed for Python3 and is **not compatible with
   Python2**.

.. note::

   In this section you are supposed to have cloned the `shennong
   repository <https://github.com/bootphon/shennong>`_ and be located
   at its root directory::

     git clone https://github.com/bootphon/shennong.git
     cd ./shennong


Installation on Linux
---------------------

The recommended installation procedure is using the `Anaconda Python
distribution <https://www.anaconda.com>`_.

* Install `conda <https://conda.io/miniconda.html>`_ on your machine,

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

As `pykaldi <https://github.com/pykaldi/pykaldi>`_ does not provide a
conda image for macos, you must install it manually.

* Install `conda <https://conda.io/miniconda.html>`_ on your machine,

* Install ``shennong`` dependencies (pykaldi excepted) in a virtual environment::

    sed 'd/pykaldi/' environment.yml > environment.macos.yml
    conda env create --name shennong -f environment.macos.yml
    conda activate shennong

* Install ``pykaldi`` from source following `those instructions
  <https://github.com/pykaldi/pykaldi#from-source>`_,

* Finally install ``shennong``::

    make install
    make test


Installation on Windows
-----------------------

* As `pykaldi <https://github.com/pykaldi/pykaldi>`_ is **not yet
  officially supported on Windows** we do not provide installation
  instructions.

* Windows users can instead install shennong in a `Windows subsystem
  for Linux <https://docs.microsoft.com/en-us/windows/wsl/about>`_ or
  within a docker container as documented :ref:`below
  <install_docker>`.


.. _install_docker:

Use in a Docker container
-------------------------

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
