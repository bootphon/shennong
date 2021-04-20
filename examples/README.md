# Examples of Shennong usage

This introduces the different examples provided with Shennong.

## Phone discrimination task

The ``features_abx`` directory contains the code for an experiment comparing the
ABX discrimination score on the [Zero Resource Challenge
2015](https://zerosspeech.com/2015) datasets (English and Xitsonga) using the
features extraction pipeline implemented in Shennong.

The results are available at
https://docs.cognitive-ml.fr/shennong/intro_features.html#features-comparison.

It requires the English and Xitsonga datasets which can be dowloaded from the
links provided
[here](https://github.com/bootphon/Zerospeech2015#zerospeech-challenge-2015).

To run the experiment, simply adapt the parameters in `run.sh` and launch it.
It must be used with a cluster running SLURM (if not you must adapt the recipe).
It assumes *shennong* and *ABXpy* are already installed in virtual environments.

To install *ABXpy* in a new conda environement just have a ``conda create --name
abx -c coml abx``.


## Features serializers

The script ``compare_features_serializers.py`` compares the performances of the
features file formats supported by Shennong in terms of file size, read and
write times.


## Features plots

The script ``plot_features.py`` simply plot the features computed from an audio
file. This required ``matplotlib`` to be installed.


## VTLN

The script ``vtln_simple.py`` is an example of use of Vocal Tract Length
Normalization. It shows how to train a VTLN model and apply it to normalize
deatures.


## Pitch algorithms comparison

Need few packages along with shennong:

    pip install amfm_decompy matplotlib praat-parselmouth tqdm

Then simply run `./pitch.py ./data`

On Linux in order to use LaTeX rendering for the plots you may need to `apt
install cm-super` (see
[here](https://github.com/matplotlib/matplotlib/issues/16911)).
