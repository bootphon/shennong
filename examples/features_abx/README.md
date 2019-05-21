This directory contains the code for an experiment comparing the ABX
discrimination score on the [Zero Resource Challenge
2015](https://zerosspeech.com/2015) datasets (English and Xitsonga)
using the features extraction pipeline implemented in *shennong*.


Datasets
--------

Download the English and Xitsonga datasets from the links available
[here](https://github.com/bootphon/Zerospeech2015#zerospeech-challenge-2015).


Considered features and tasks
-----------------------------

* The considered features extraction algorithms are:

  - *spectrogram*
  - *filterbanks*
  - *MFCC*
  - *PLP*
  - *RASTA PLP*
  - *bottleneck*

* Each is tested with 3 distinct parameters sets:

  - *only*: just the raw features
  - *nocmvn*: raw features with delta, delta-delta and pitch
  - *full* raw features with CMVN normalization by speaker, with
    delta, delta-delta and pitch

* The considered ABX tasks are the same as in the
  [ZRC2015 track1](https://zerospeech.com/2015/track_1.html), namely a
  phone discrimination task within and across speakers.

* This gives us 2 corpora * 2 tasks * 6 features * 3 parameters sets =
  72 scores.


Recipe
------

This recipe is divided in 3 steps:

- `1_setup_features.sh` takes the 2 datasets as input. It downloads
  some auxiliary files, compute the ABX tasks and extract the
  features.

- `2_abx_score.sh` computes the average ABX phone discrimination score
  for all extracted features in both languages and two conditions
  (within or across speakers).

- `3_publish_results.sh` creates result tables ready to be included in
  the shennong documentation.

It must be used with a cluster running SLURM (if not you must adapt
the recipe). It assumes *shennong* is installed in a virtual
environment called *shennong* and *ABXpy* is installed in an
environment called *abx*.
