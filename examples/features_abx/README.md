This directory contains the code for an experiment comparing the ABX
discrimination score on the [Zero Resource Challenge
2015](https://zerosspeech.com/2015) datasets (English and Xitsonga)
using the features extraction pipeline implemented in *shennong*.

Datasets
--------

Download the English and Xitsonga datasets from the links available
[here](https://github.com/bootphon/Zerospeech2015#zerospeech-challenge-2015).


Recipe
------

Simply adapt the parameters in `run.sh` and launch it.

It must be used with a cluster running SLURM (if not you must adapt
the recipe). It assumes *shennong* and *ABXpy* are already installed
in virtual environments.

Results
-------

The results are available at
https://docs.cognitive-ml.fr/shennong/intro_features.html#features-comparison
