.. _intro_features:

Introduction to speech features
===============================

Implemented models
------------------

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
  Pitch           from Kaldi_
  Energy          from Kaldi_
  =============== ==============

* The following post-processing pipelines are implemented in
  ``shennong``, the detailed documentation is available :ref:`here
  <features.postprocessor>`:

  ===================================== ==============
  Post-processing                       Implementation
  ===================================== ==============
  Delta / Delta-delta                   from Kaldi_
  Mean Variance Normalization (CMVN)    from Kaldi_
  Voice Activity Detection              from Kaldi_
  ===================================== ==============

* Here is an illustration of the features (no post-processing)
  computed on the example wav file provided as
  ``shennong/test/data/test.wav``, this picture can be reproduced
  using the example script ``shennong/examples/plot_features.py``:

  .. image:: static/features.png
     :width: 70%


Features comparison
-------------------

**coming soon**

.. _Kaldi: https://kaldi-asr.org
.. _rastapy: https://github.com/mystlee/rasta_py
.. _labrosa: https://labrosa.ee.columbia.edu/matlab/rastamat/
.. _BUTspeech: https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor
