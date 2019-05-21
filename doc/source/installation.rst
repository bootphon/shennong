.. _installation:


Installation
============

* This installation process is documented for Linux Ubuntu, but it can
  be adapted to other Linux/MacOS distributions easilly.

* Shennong is developed for Python3 and is **not compatible with
  Python2**.

* Windows users can install shennong in a `Windows subsystem for Linux
  <https://docs.microsoft.com/en-us/windows/wsl/about>`_ or within a
  docker container as documented :ref:`below <install_docker>`.

.. note::

   In this section you are supposed to have cloned the `shennong
   repository <https://github.com/bootphon/shennong>`_ and be located
   at its root directory::

     git clone https://github.com/bootphon/shennong.git
     cd ./shennong


Installation with conda
-----------------------

The recommended installation procedure is using the `Anaconda Python
distribution <https://www.anaconda.com>`_.

* Install `conda <https://conda.io/miniconda.html>`_ on your machine,

* Setup a Python virtual environment with the dependencies:

  * Create a fresh ``shennong`` Python virtual environment::

      conda env create --name shennong -f environment.yml
      conda activate shennong

  * **Or**, if you need to work in an existing environment, update it::

      conda env update --name MYENV -f environment

* Then install the ``shennong`` package::

    make install

* Finally make sure the installation is correct by executing the tests
  (this executes all the unit tests stored in the ``test/`` folder)::

    make test


.. note::

   To use ``shennong``, do not forget to activate the virtual environment::

     conda activate shennong


Installation upgrade
--------------------

This documents how to upgrade your ``shennong`` installation to the
latest version available.

* Update the ``shennong`` code::

    git pull origin master

* Update the dependencies::

    conda env update --name shennong -f environment.yml

* Reinstall and test it::

    make install
    make test


.. _install_docker:

Use in a Docker container
-------------------------

You can use ``shennong`` from within `docker
<https://docs.docker.com>`_ using the provided ``Dockerfile``.

* Build the docker image with::

    [sudo] docker build -t shennong .

* Run an interactive session with::

    [sudo] docker run -it shennong /bin/bash

* Open a jupyter notebook as follow::

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
