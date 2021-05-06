# Pitch algorithms comparison

Comparison of 4 pitch estimation algorithms (CREPE and Kaldi bundled with
shennong, PRAAT and YAAPT) under various noise conditions.

## Setup

Need few packages along with shennong:

    pip install amfm_decompy matplotlib praat-parselmouth tqdm

On Linux in order to use LaTeX rendering for the plots you may need to `apt
install cm-super` (see
[here](https://github.com/matplotlib/matplotlib/issues/16911)).


## Run

Simply run `./pitch.py ./data`. The script will dowload the dataset, prepare the
data, estimate the pitches and plot the results in `./data/plots`.
