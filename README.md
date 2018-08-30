# shennong-features

Speech features extraction package of the **shennong toolbox**.

## Models to implement in Shennong

* Features
  * MFCC
  * Filterbanks
  * PLP
  * BUT Bottleneck features
  * 1 hot

* Pre/post processing
  * delta and delta2
  * pitch
  * energy
  * VAD
  * MVN
  * VTLN

## Ideas for shennong-features implementation

* all possible from Kaldi, using PyKaldi
  * MFCC, filterbanks, PLP
  * deltas, pitch, energy, MVN, VTLN

* adapt from BUT for bottleneck features and VAD

## Existing features extraction packages

### features_extraction

* https://github.com/bootphon/features_extraction

* A wrapper on other packages (spectral and Matlab toolboxes)

* Implemented models
    * Implements **MFCC**, **Mel filterbanks**, **RASTA**, **lyon** and **drnl**
    * Depends on *spectral* for mel and MFCC
    * Depends on matlab/octave for RASTA, lyon and drnl


### spectral

* https://github.com/bootphon/spectral

* Implement **MFCC** and **Mel filterbanks** (Python implementation)
* With **delta** and **delta-delta**


### Kaldi

* https://github.com/kaldi-asr/kaldi

* Implement **MFCC**, **Mel filterbanks** and **PLP** (C++ implementation)
* With **delta**, **CMVN** and **pitch**.


### BUT/Phonexia Bottleneck feature extractor

* http://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor

* bottleneck, stacked bottleneck features and phoneme/senones posteriors
* also includes a **VAD**


### HTK

* http://htk.eng.cam.ac.uk/

* Custom licence incompatible with GPL3, so we cannot use HTK in Shennong.
* Implement roughly the same features as Kaldi does.
