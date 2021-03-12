# ChangeLog

Version numbers follow [semantic versioning](https://semver.org).


## not yet released

* Vocal Tract Length Normalization (VTLN) implemented using Kaldi:
  `shennong.processor.vtln`.

* CREPE pitch extraction: `shennong.processor.crepepitch`.

* Features serialization supported in CSV (replace JSON format).

* Optionnally ignore features properties when saving them.

* Code reorganization and improvments (renamed `shennong.features.*` to
  `shennong.*`, new `shennong.logger`, etc...)


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
