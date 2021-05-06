# Simple examples of Shennong usage

## Features file formats

The script ``features_file_formats.py`` compares the performances of the
features file formats supported by Shennong in terms of file size, read and
write times. The resulting table can be found in the documentation for
`shennong/features_collection.py`.


## Features plots

The script ``features_plot.py`` plot the features computed from an audio file
using various algorithms. This requires ``matplotlib`` to be installed. The
resulting plot is displayed in the *Introduction to speech features* section of
the documentation.


## MFCC with VTLN

The script ``mfcc_vtln.py`` is an example of use of Vocal Tract Length
Normalization. It shows how to train a VTLN model and apply it to normalize
MFCC features.
