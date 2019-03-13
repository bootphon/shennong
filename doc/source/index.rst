Shennong's documentation
========================

.. image:: https://api.travis-ci.org/bootphon/shennong.svg?branch=master
    :target: https://travis-ci.org/bootphon/shennong
.. image:: https://codecov.io/gh/bootphon/shennong/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/bootphon/shennong


* Shennong is a **toolbox for unsupervised speech recognition**.

* It provides both a **Python API** and a **command line interface**.

* It aims to provide three blocks of processing, inspired by the tasks
  and results of the `Zero Resource Speech challenges
  <http://www.zerospeech.com/>`_:

  * Speech features extraction,
  * Subword modeling,
  * Spoken term discovery.

.. note::

   The package is still in development phase (current version is
   |version|):

    * Release *0.1* will include the *features* block,
    * Release 0.2 will include the *subword modeling* block,
    * Release *0.3* will include the *term discovery* block,
    * Further versions will include complete recipes, replication of
      published results and additional models.


.. toctree::
   :maxdepth: 3
   :caption: Contents

   installation
   python/index


Licence and copyright
---------------------

.. figure:: static/inria.jpg
   :align: left
   :figwidth: 190px
   :alt: Inria
   :target: https://inria.fr/en

Copyright |copyright|

This work is founded by the grant *ADT-193* from `Inria
<https://inria.fr/en>`_ and developed within the `Cognitive Machine
Learning <https://coml.lscp.ens.fr>`_ research team.

-----------------------

shennong is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

shennong is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with shennong. If not, see http://www.gnu.org/licenses/.
