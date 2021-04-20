"""Provides the Frames class to extract frames from raw signals

Extracts overlapping frames from raw (sampled) signals::

    array ---> Frames ---> array

Examples
--------

>>> import numpy as np
>>> from shennong.frames import Frames

Build a discrete signal

>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Computes frames of 3s with a shift of 1s (here we assume fs=1Hz
for simplicity)

>>> f = Frames(sample_rate=1, frame_shift=1, frame_length=3)
>>> b = f.make_frames(a)
>>> b
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4],
       [3, 4, 5],
       [4, 5, 6],
       [5, 6, 7],
       [6, 7, 8],
       [7, 8, 9]])

"""

import kaldi.feat.window
import numpy as np

from shennong.base import BaseProcessor


class Frames(BaseProcessor):
    """Extract frames from raw signals"""
    def __init__(self, sample_rate=16000,
                 frame_shift=0.01, frame_length=0.025,
                 snip_edges=True):
        self._options = kaldi.feat.window.FrameExtractionOptions()
        self.sample_rate = sample_rate
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.snip_edges = snip_edges

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
        return int(self.frame_length * self.sample_rate)

    @property
    def samples_per_shift(self):
        """The number of samples between two shifts"""
        return int(self.frame_shift * self.sample_rate)

    def nframes(self, nsamples):
        """Returns the number of frames extracted from `nsamples`

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

        Raises
        ------
        ValueError
            If ``samples_per_shift == 0``, meaning the sample rate is
            to low w.r.t the frame shift.

        """
        if self.samples_per_shift == 0:
            raise ValueError('cannot compute nframes: sample rate too low')

        return int(kaldi.feat.window.num_frames(
            nsamples, self._options, flush=True))

    def first_sample_of_frame(self, frame):
        """Returns the index of the first sample of frame indexed `frame`"""
        return int(frame * self.samples_per_shift)

    def last_sample_of_frame(self, frame):
        """Returns the index+1 of the last sample of frame indexed `frame`"""
        return int(self.first_sample_of_frame(frame) + self.samples_per_frame)

    def times(self, nsamples):
        """Returns an array of (tstart, tstop) times of each frames of a signal

        Parameters
        ----------
        nsamples : int
            The number of frames of the considered signal

        Returns
        -------
        times : array, shape = [nframes, 2]
            The start and stop times of each frame extracted from
            `nsamples` samples.

        """
        nframes = self.nframes(nsamples)
        return np.vstack((
            np.arange(nframes) * self.frame_shift,
            np.arange(nframes) * self.frame_shift + self.frame_length)).T

    def boundaries(self, nframes):
        """Returns an array of (istart, istop) index boundaries of frames

        Parameters
        ----------
        nframes : int
            The number of frames to generate

        Returns
        -------
        boundaries : array, shape = [nframes, 2]
            The start and stop indices of each frame extracted from
            `nsamples` samples.

        """
        first = [self.first_sample_of_frame(i) for i in range(nframes)]
        return (np.asarray(first).repeat(2).reshape(nframes, 2)
                + (0, self.samples_per_frame)).astype(np.int)

    def make_frames(self, array, writeable=False):
        """Returns an `array` divided in frames

        Parameters
        ----------
        array : array, shape = [x, ...]
            The array to be divided in frames
        writeable : bool, optional
            Default to False. When True, the returned array is
            writable but the frames are made of copies of the original
            `array`. When False, the result is read-only but this
            optimizes the process: no explicit copy is made of the
            orignal `array`, only views are used. (see
            https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/
            numpy.lib.stride_tricks.as_strided.html)

        Returns
        -------
        frames : array, shape = [nframes(x), samples_per_frame, ...]
            The frames computed from the original `array`

        """
        nframes = self.nframes(array.shape[0])

        # special case when not sniping edges: mirror the data in the
        # last frames
        if not self.snip_edges:
            n = self.last_sample_of_frame(nframes-1) - array.shape[0]
            array = np.concatenate((array, array[-n-1:-1][::-1]))

        if writeable is True:
            return self._make_frames_by_copy(array, nframes)
        else:
            return self._make_frames_by_view(array, nframes)

    def _make_frames_by_view(self, array, nframes):
        # shape of the frames, concatenate the shape for supplementary
        # dimensions
        shape = (nframes, self.samples_per_frame) + array.shape[1:]

        # strides for the framed array, don't touch the strides for
        # the additional dimensions
        strides = (array.strides[0] * self.samples_per_shift,
                   array.strides[0]) + array.strides[1:]

        return np.lib.stride_tricks.as_strided(
            array, shape=shape, strides=strides, writeable=False)

    def _make_frames_by_copy(self, array, nframes):
        # the frames boundaries
        boundaries = self.boundaries(nframes)
        nsamples = self.samples_per_frame

        # allocate the framed array
        framed = np.empty(
            (nframes, nsamples) + array.shape[1:],
            dtype=array.dtype)

        # build the frames
        for i, (start, stop) in enumerate(boundaries):
            assert stop - start == nsamples
            framed[i] = array[start:stop]
        return framed
