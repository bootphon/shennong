Data for testing
================

The `test/data` directory contains data used in the `shennong` tests
suite.

* `alignment.txt` is a forced alignment of a fragment of the Buckeye
  corpus.

* `test.wav` is a speech fragment in Czech from the [Bottleneck
  package](https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor),
  `test.8k.wav` is a downsampling (orignal file is sampled at 16 kHz).

* `test.bottleneck.fea` contains features extracted from the original
  [Bottleneck
  package](https://speech.fit.vutbr.cz/software/but-phonexia-bottleneck-feature-extractor)
  on `test.8k.wav`, it is used to test replicability of the bottleneck
  features implementation in `shennong`.

* `test.rastaplp.npy` contains features extracted from the original
  [rastapy package](https://github.com/mystlee/rasta_py) on
  `test.wav`, it is used to test the replicability of the RASTA-PLP
  implmementation in `shennong`.
