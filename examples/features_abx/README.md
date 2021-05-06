## Phone discriminability

This directory contains the code for an experiment comparing the ABX
discrimination score on the [Zero Resource Challenge
2015](https://zerosspeech.com/2015) datasets (English and Xitsonga) using
features extraction pipelines implemented in Shennong.

The results of this experiment are available at
https://docs.cognitive-ml.fr/shennong/intro_features.html#features-comparison.

It requires the English and Xitsonga datasets which can be dowloaded from the
links provided
[here](https://github.com/bootphon/Zerospeech2015#zerospeech-challenge-2015).

To run the experiment, simply adapt the parameters in `run.sh` and launch it.

* It must be used with a cluster running SLURM (if not you must adapt the
  recipe).

* It assumes *shennong* and *ABXpy* are already installed in separated virtual
  environments. To install *ABXpy* in a dedicated conda environement just have a
  ``conda create --name abx -c coml abx``.

* The final results are available as `data/final_scores.txt`.
