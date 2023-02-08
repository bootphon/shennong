Shennong's documentation
========================

.. image:: https://api.travis-ci.org/bootphon/shennong.svg?branch=master
  :target: https://travis-ci.org/bootphon/shennong
.. image:: https://codecov.io/gh/bootphon/shennong/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/bootphon/shennong


* Shennong is a **toolbox for speech features extraction** which provide a wide
  range of extraction and post-processing algorithms.

* It provides both a **Python API** and a **command line interface**.

* It is mainly based on `Kaldi <https://kaldi-asr.org>`_ using the `pykaldi
  <https://github.com/pykaldi/pykaldi>`_ Python adapter.


.. note::

   Please use the `following paper <https://doi.org/10.3758/s13428-022-02029-6>`_,
   which is also available on `arXiv <https://arxiv.org/pdf/2112.05555.pdf>`_, to
   cite ``shennong``::

    @article{bernard2023shennong,
      title = {Shennong: {{A Python}} Toolbox for Audio Speech Features Extraction},
      author = {Bernard, Mathieu and Poli, Maxime and Karadayi, Julien and Dupoux, Emmanuel},
      year = {2023},
      journal = {Behavior Research Methods},
      url = {https://doi.org/10.3758/s13428-022-02029-6},
      doi = {10.3758/s13428-022-02029-6},
    }


.. toctree::
   :maxdepth: 3
   :caption: Contents

   installation
   intro_features
   cli
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
