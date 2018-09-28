"""Extract frames from raw signals"""

import kaldi.feat.window
import numpy as np


class Frame:
    def __init__(self, sample_rate=16000,
                 frame_shift=0.01, frame_length=0.025,
                 snip_edges=True):
        self._options = kaldi.feat.window.FrameExtractionOptions()
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length

    @property
    def sample_rate(self):
        """Waveform sample frequency in Hertz

        Must match the sample rate of the signal specified in
        `process`

        """
        return self._options.samp_freq

    @sample_rate.setter
    def sample_rate(self, value):
        self._options.samp_freq = value

    @property
    def frame_shift(self):
        """Frame shift in seconds"""
        return self._options.frame_shift_ms / 1000.0

    @frame_shift.setter
    def frame_shift(self, value):
        self._options.frame_shift_ms = value * 1000.0

    @property
    def frame_length(self):
        """Frame length in seconds"""
        return self._options.frame_length_ms / 1000.0

    @frame_length.setter
    def frame_length(self, value):
        self._options.frame_length_ms = value * 1000.0

    @property
    def snip_edges(self):
        """If true, output only frames that completely fit in the file

        When True the number of frames depends on the `frame_length`.
        If False, the number of frames depends only on the
        `frame_shift`, and we reflect the data at the ends.

        """
        return self._options.snip_edges

    @snip_edges.setter
    def snip_edges(self, value):
        self._options.snip_edges = value

    @property
    def samples_per_frame(self):
        """The number of samples in one frame"""
        return self.frame_length * self.sample_rate

    @property
    def samples_per_shift(self):
        """The number of samples between two shifts"""
        return self.frame_shift * self.sample_rate

    def nframes(self, nsamples):
        """Returns the number of frames extracted from `nframes`

        This function returns the number of frames that we can extract
        from a wave file with the given number of samples in it
        (assumed to have the same sampling rate as specified in init).

        Parameters
        ----------
        nsamples : int
            The number of samples in the input

        Returns
        -------
        nframes : int
            The number of frames extracted from `nsamples`

        """
        return int(kaldi.feat.window.num_frames(
            nsamples, self._options, flush=True))

    def first_sample_of_frame(self, frame):
        """Returns the index of the first sample of the frame indexed `frame`

        If `snip_edges` is True, it just returns :math:`frame *
        shift`. If `snip_edges` is False, the formula is a little more
        complicated and the result may be negative.

        """
        return int(kaldi.feat.window.first_sample_of_frame(
            frame, self._options))

    def boundaries(self, nsamples):
        """Returns an array of (istart, istop) index boundaires of frames

        Parameters
        ----------
        nsamples : int
            The number of samples in the input

        Returns
        -------
        boundaries : array, shape = [nframes, 2]
            The start and stop indices of each frame extracted from
            `nsamples` samples.

        """
        nframes = self.nframes(nsamples)
        first = np.asarray(
            [self.first_sample_of_frame(i) for i in range(nframes)])
        return (first.repeat(2).reshape(nframes, 2)
                + (0, self.samples_per_frame)).astype(np.int)

    def framed_array(self, array):
        """Returns an `array` divided in frames

        Parameters
        ----------
        array : array, shape = [x, ...]
            The array to be divided in frames

        Returns
        -------
        frames : array, shape = [nframes(x), samples_per_frame, ...]

        """
        nsamples = array.shape[0]
        shape = (
            self.nframes(nsamples), self.samples_per_frame) + array.shape[1:]

        strides = (
            array.strides[0] * self.samples_per_shift,
            array.strides[0]) + array.strides[1:]

        return np.lib.stride_tricks.as_strided(
            array, shape=shape, strides=strides)
