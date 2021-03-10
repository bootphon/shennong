# ChangeLog

Version numbers follow [semantic versioning](https://semver.org).


## not yet released

### new functionnalities

* Vocal Tract Length Normalization (VTLN) implemented after Kaldi.

### breaking changes

* renamed `shennong.features.*` to `shennong.*`.


## shennong-0.1.1

### bugfixes

* correctly load utterances from a file in `speech-features` (ignore empty lines)

* fixed a harmless warning when resampling audio with sox

### environment

* now compatible with (and depends on) `pytest>=5.0`

* improved installation instructions for MacOS

* new releases are automatically deployed on conda

## shennong-0.1

First public release
