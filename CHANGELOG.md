# ChangeLog

Version numbers follow [semantic versioning](https://semver.org).


## shennong-1.0

### breaking changes

* Deletion of `processor.rastaplp.RastaPlpProcessor`. Rasta filtering is now an
  option of the standard `PlpProcessor`, so as to use both Rasta and VTLN.

* Pitch, delta and CMVN are now deactivated by default in pipeline configuration
  generation (concerns both `speech-features` binary and the `shennong.pipeline`
  module).

* Features serialization in JSON is no more supported (replaced by CSV)

* Code reorganization (renamed `shennong.features.*` to `shennong.*`, new
  `shennong.logger`, import processors directly from
  `shennong.processor`, renamed `PitchProcessor` to `KaldiPitchProcessor`,
  etc...)

* When defining utterances for use with a pipeline, the format `<audio-file>`
  is no more supported, it must be superseeded by `<utterance-id> <audio-file>`.

* `processor.process_all()` now takes a `shennong.Utterances` instead of a
  `dict(name, audio)`.

### new models

* Vocal Tract Length Normalization (VTLN) implemented using Kaldi:
  `shennong.processor.vtln`.

* CREPE pitch extraction: `shennong.processor.pitch_crepe`.

### improvments

* new `shennong.Utterances` class to encapsulate and manage utterances to be
  feeded to a pipeline.

* `shennong.Audio` can now read/write more than wav files: flac, mp3, etc...
  (anything supported by pydub/ffmpeg).

* Optionnally ignore features properties when saving them.

* New example code in `examples`.


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
