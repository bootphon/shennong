.. _intro_features:

Introduction to speech features
===============================

Implemented models
------------------

.. note::

  All the models implemented from Kaldi are using `pykaldi
  <https://github.com/pykaldi/pykaldi>`_, see [Can2018]_.

* The following features extraction models are implemented in
  ``shennong``, the detailed documentation is available :ref:`here
  <features.processor>`:

  =============== ==============
  Features        Implementation
  =============== ==============
  Spectrogram     from Kaldi_
  Filterbank      from Kaldi_
  MFCC            from Kaldi_
  PLP             from Kaldi_
  RASTA-PLP       from rastapy_, after labrosa_
  Bottleneck      from BUTspeech_
  One Hot Vectors shennong
  Pitch           from Kaldi_ and CREPE_
  Energy          from Kaldi_
  =============== ==============

* The following post-processing pipelines are implemented in
  ``shennong``, the detailed documentation is available :ref:`here
  <features.postprocessor>`:

  ======================================= ==============
  Post-processing                         Implementation
  ======================================= ==============
  Delta / Delta-delta                     from Kaldi_
  Mean Variance Normalization (CMVN)      from Kaldi_
  Voice Activity Detection                from Kaldi_
  Vocal Tract Length Normalization (VTLN) from Kaldi_
  ======================================= ==============

* Here is an illustration of the features (without post-processing)
  computed on the example wav file provided as
  ``shennong/test/data/test.wav``, this picture can be reproduced
  using the example script ``shennong/examples/plot_features.py``:

  .. image:: static/features.png
     :width: 70%


Features comparison
-------------------

This section details a phone discrimination task based on the features
available in ``shennong``. It reproduces the track 1 of the `Zero
Speech Challenge 2015 <https://zerospeech.com/2015/track_1.html>`_
using the same datasets and setup. The recipe to replicate this
experiment is available at ``shennong/examples/features_abx``.


* Setup:

  * Two languages are tested:

    * English (`Buckeye corpus <https://buckeyecorpus.osu.edu/>`_, 12
      speakers for a duration of 10:34:44)

    * Xitsonga (`NCHLT corpus
      <http://rma.nwu.ac.za/index.php/nchlt-speech-corpus-ts.html>`_,
      24 speakers for a duration of 4:24:37)

  * The considered features extraction algorithms are:

    * bottleneck
    * filterbanks
    * MFCC
    * PLP
    * RASTA PLP
    * spectrogram

  * Each is tested with 3 distinct parameters sets:

    * **only**: just the raw features,
    * **nocmvn**: raw features with delta, delta-delta and pitch,
    * **full**: raw features with CMVN normalization by speaker, with
      delta, delta-delta and pitch.

  * The considered ABX tasks are the same as in the `ZRC2015 track1
    <https://zerospeech.com/2015/track_1.html>`_, namely a phone
    discrimination task within and across speakers.

  * This gives us 2 corpora * 2 tasks * 6 features * 3 parameters sets
    = 72 scores.

.. note::

   The results below are ABX error rates on phone discrimination
   (given in %).

* Results on English:

  +-------------+--------------------------------------------------------------+---------------------------------------------------------------+
  |             |                    across                                    |                            within                             |
  |  features   +-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  |             | only  | nocmvn  | full | only-vtln | nocmvn-vtln | full-vtln |  only  | nocmvn |  full | only-vtln | nocmvn-vtln | full-vtln |
  +=============+=======+=========+======+===========+=============+===========+========+========+=======+===========+=============+===========+
  | bottleneck  |  12.5 |  12.5   | 12.5 |           |             |           |   8.5  |    8.5 |   8.6 |           |             |           |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | filterbank  |  24.9 |  22.1   | 26.5 |    23.2   |    20.7     |    25.4   |  12.8  |   11.6 |  18.2 |   12.6    |     11.4    |   18.1    |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | mfcc        |  27.2 |  26.4   | 24.0 |    23.4   |    22.7     |    20.0   |  13.0  |   12.5 |  12.4 |   12.8    |     12.3    |   12.0    |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | plp         |  28.0 |  26.6   | 23.8 |    24.8   |    23.5     |    19.7   |  12.5  |   12.4 |  12.0 |   12.5    |     12.4    |   11.9    |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | rastaplp    |  26.8 |  30.0   | 22.7 |           |             |           |  18.1  |   23.0 |  13.1 |           |             |           |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | spectrogram |  30.3 |  27.9   | 29.7 |    30.3   |    27.9     |    29.7   |  16.7  |   15.2 |  20.2 |   16.6    |     15.2    |   20.2    |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+

* Results on Xitsonga:

  +-------------+--------------------------------------------------------------+---------------------------------------------------------------+
  |             |                    across                                    |                            within                             |
  |  features   +-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  |             | only  | nocmvn  | full | only-vtln | nocmvn-vtln | full-vtln |  only  | nocmvn |  full | only-vtln | nocmvn-vtln | full-vtln |
  +=============+=======+=========+======+===========+=============+===========+========+========+=======+===========+=============+===========+
  | bottleneck  |  9.5  |   9.6   |  9.6 |           |             |           |   6.9  |    7.0 |   7.3 |           |             |           |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | filterbank  |  28.1 |  25.1   | 21.5 |    26.9   |    24.0     |    20.7   |  13.8  |   11.7 |  15.2 |   13.6    |     11.4    |    15.2   |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | mfcc        |  33.6 |  32.8   | 26.0 |    31.4   |    30.6     |    22.7   |  17.1  |   16.2 |  14.6 |   17.5    |     16.5    |    14.6   |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | plp         |  33.5 |  31.2   | 26.2 |    31.8   |    29.5     |    22.3   |  16.2  |   14.6 |  14.0 |   16.3    |     14.7    |    14.2   |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | rastaplp    |  27.1 |  25.6   | 21.3 |           |             |           |  19.5  |   20.1 |  12.6 |           |             |           |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+
  | spectrogram |  34.6 |  32.0   | 26.5 |    34.6   |    32.0     |    26.6   |  19.2  |   16.8 |  19.2 |   19.1    |     16.8    |    19.3   |
  +-------------+-------+---------+------+-----------+-------------+-----------+--------+--------+-------+-----------+-------------+-----------+

* Comparison with the `ZRC2015 baseline
  <https://zerospeech.com/2015/results.html>`_ (on MFCC only), see
  [Versteegh2015]_:

  +--------------------+-----------------+-----------------+
  |                    |     English     |      Xitsonga   |
  |                    +--------+--------+--------+--------+
  |                    | across | within | across | within |
  +====================+========+========+========+========+
  |   ZRC2015          |  28.1  |  15.6  |  33.8  | 19.1   |
  +--------------------+--------+--------+--------+--------+
  | shennong-only      |  27.2  |  13.0  |  33.6  | 17.1   |
  +--------------------+--------+--------+--------+--------+
  | shennong-full      |  24.0  |  12.4  |  26.0  | 14.6   |
  +--------------------+--------+--------+--------+--------+
  | ZRC2015-vtln       |  24.0  |  14.6  |        |        |
  +--------------------+--------+--------+--------+--------+
  | shennong-only-vtln |  23.4  |  12.8  |  31.4  | 17.5   |
  +--------------------+--------+--------+--------+--------+
  | shennong-full-vtln |  20.0  |  12.0  |  22.7  | 14.2   |
  +--------------------+--------+--------+--------+--------+

.. _Kaldi: https://kaldi-asr.org
.. _CREPE: https://github.com/marl/crepe
.. _rastapy: https://github.com/mystlee/rasta_py
.. _labrosa: https://labrosa.ee.columbia.edu/matlab/rastamat/
.. _BUTspeech: https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor


---------------------------------------------

.. [Versteegh2015] *The zero resource speech challenge 2015*, Maarten
   Versteegh, Roland Thiollière, Thomas Schatz, Xuan-Nga Cao, Xavier
   Anguera, Aren Jansen, and Emmanuel Dupoux. In
   INTERSPEECH-2015. 2015.

.. [Can2018] *PyKaldi: A Python Wrapper for Kaldi*, Dogan Can and
   Victor R. Martinez and Pavlos Papadopoulos and
   Shrikanth S. Narayanan, in IEEE International Conference on
   Acoustics Speech and Signal Processing (ICASSP), 2018.
